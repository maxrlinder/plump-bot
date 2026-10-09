"""Fused Metal implementation of the decoder trunk for MPS training.

The PyTorch trunk spends almost all of its update time outside the matrix
multiplications: SDPA's grouped-query path expands K/V and reduces their
gradients back with slow strided sums, autocast inserts a cast around every
LayerNorm, and each residual add, GELU and LayerNorm is its own pass over a
[rows, d_model] fp32 tensor. Measured on the rl-v6 actor update (156k positions,
16 layers, bf16) the trunk ran at about 1.3 TFLOPS against the ~20 TFLOPS MPS
reaches on the same GEMMs.

This module keeps the GEMMs on MPS and replaces everything between them:

    forward, per layer   add+LN | QKV | attention | out | add+LN | up | GELU | down
    backward, per layer  down/up/out/qkv data GEMMs and split-K weight GEMMs,
                         one GELU-backward kernel (+ up bias), two LayerNorm
                         backward kernels (+ down bias, per-layer-embedding
                         gradient, the bf16 copy the next GEMM reads), and one
                         attention-backward kernel

Attention runs one threadgroup per (sequence, KV head): that head's keys and
values fit in threadgroup memory (sequences are at most 121 tokens) and the
query heads sharing it are threads of the same group, so the GQA reduction of
dK/dV never leaves the threadgroup.

The backward accepts several cotangents at once, interleaved per row. The PPO
actor needs the auxiliary and policy gradients separately (each gets its own
norm budget), which the autograd path pays for with two full trunk backwards;
here both streams share every saved-activation read and every GEMM launch.

Precision follows PyTorch bf16 autocast wherever it rounds: fp32 residual
stream and LayerNorm statistics, bf16 GEMM operands and outputs, the value
embedding added to v in fp32 and rounded once to bf16. Results are equivalent
to the autocast path within bf16 rounding, not bit-identical.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook

from .config import SeqModelConfig

ENV_FLAG = "PLUMP_METAL_TRUNK"
# Threadgroup memory on Apple GPUs is 32 KB; attention keeps K and V for a
# whole sequence there as fp32.
_THREADGROUP_BYTES = 32 * 1024
MAX_LENGTH = 128
_LN_THREADGROUPS = 128
_LN_SIMDGROUPS = 8
_GELU_ROWS = 32
_WGRAD_SPLITS = 8


def metal_trunk_requested() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def unsupported_reason(config: SeqModelConfig) -> str | None:
    """Why this config cannot use the fused trunk, or None when it can."""

    if config.dropout:
        return "dropout must be 0"
    if config.per_layer_embedding_dim not in (0, 64):
        return "per-layer embeddings must be 64 wide (or off)"
    if config.per_layer_embedding_dim and config.d_model < 64:
        return "d_model must be at least the per-layer embedding width"
    group = config.n_heads // config.kv_heads
    if group * MAX_LENGTH > 1024:
        return "query heads per KV head exceed one threadgroup"
    attention_bytes = 4 * (2 * MAX_LENGTH * config.head_dim + group * MAX_LENGTH)
    if attention_bytes > _THREADGROUP_BYTES:
        return "head_dim too large for threadgroup-resident K/V"
    if max(config.max_seq_len, config.oracle_max_seq_len) > MAX_LENGTH:
        return f"sequences longer than {MAX_LENGTH}"
    return None


def _source(dm: int, n_heads: int, kv_heads: int, head_dim: int, ple: int) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int DM = {dm};
constant constexpr int PLE = {max(ple, 1)};
constant constexpr int PER_LANE = (DM + 31) / 32;
constant constexpr int D = {head_dim};
constant constexpr int KVH = {kv_heads};
constant constexpr int NH = {n_heads};
constant constexpr int G = NH / KVH;
constant constexpr int QKV_W = (NH + 2 * KVH) * D;
constant constexpr int MAXL = {MAX_LENGTH};
constant constexpr int K_OFF = NH * D;
constant constexpr int V_OFF = NH * D + KVH * D;

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7: Metal has no erf.
inline float erf_approx(float x) {{
    const float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return copysign(y, x);
}}

// Residual add + LayerNorm, one simdgroup per row.
// flags: 1 add x, 2 add delta (bf16), 4 add ple, 8 write x_out, 16 fp32 output.
kernel void add_ln_fwd(
    device const float* x [[buffer(0)]],
    device const bfloat* delta [[buffer(1)]],
    device const float* ple [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* x_out [[buffer(5)]],
    device bfloat* y_bf [[buffer(6)]],
    device float* y_f [[buffer(7)]],
    device float* stats [[buffer(8)]],
    constant int& R [[buffer(9)]],
    constant int& flags [[buffer(10)]],
    constant int& ple_stride [[buffer(11)]],
    constant int& ple_off [[buffer(12)]],
    constant float& ple_scale [[buffer(13)]],
    constant float& eps [[buffer(14)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const int row = gid / 32;
    if (row >= R) return;
    const long off = (long)row * DM;
    float v[PER_LANE];
    float sum = 0.0f;
    for (int i = 0; i < PER_LANE; ++i) {{
        const int e = lane + 32 * i;
        float val = 0.0f;
        if (e < DM) {{
            if (flags & 1) val = x[off + e];
            if (flags & 2) val += float(delta[off + e]);
            if (flags & 4) val += ple_scale * ple[(long)row * ple_stride + ple_off + (e % PLE)];
            if (flags & 8) x_out[off + e] = val;
        }}
        v[i] = val;
        sum += val;
    }}
    const float mean = simd_sum(sum) / DM;
    float sq = 0.0f;
    for (int i = 0; i < PER_LANE; ++i) {{
        const int e = lane + 32 * i;
        if (e < DM) {{ const float c = v[i] - mean; sq += c * c; }}
    }}
    const float rstd = rsqrt(simd_sum(sq) / DM + eps);
    for (int i = 0; i < PER_LANE; ++i) {{
        const int e = lane + 32 * i;
        if (e < DM) {{
            const float y = (v[i] - mean) * rstd * gamma[e] + beta[e];
            if (flags & 16) y_f[off + e] = y; else y_bf[off + e] = bfloat(y);
        }}
    }}
    if (lane == 0) {{ stats[2 * row] = mean; stats[2 * row + 1] = rstd; }}
}}

// LayerNorm backward over S interleaved streams: row r of stream s sits at
// r * S + s in dy/dres/dx/dx_bf. One simdgroup per (row, stream); the grid
// stride is a multiple of S, so each simdgroup keeps one stream throughout.
// Per-threadgroup partials per stream: dgamma, dbeta, and the column sum of
// the bf16 dx copy (the bias gradient of the GEMM that produced the residual
// this gradient flows back into). flags: 1 dy bf16, 2 add dres, 4 write
// dx_bf (and its column sum), 8 write the per-layer-embedding gradient.
kernel void ln_bwd(
    device const bfloat* dy_bf [[buffer(0)]],
    device const float* dy_f [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device const float* stats [[buffer(3)]],
    device const float* gamma [[buffer(4)]],
    device const float* dres [[buffer(5)]],
    device float* dx [[buffer(6)]],
    device bfloat* dx_bf [[buffer(7)]],
    device float* dple [[buffer(8)]],
    device float* partial [[buffer(9)]],
    constant int& R [[buffer(10)]],
    constant int& S [[buffer(11)]],
    constant int& flags [[buffer(12)]],
    constant int& ple_stride [[buffer(13)]],
    constant int& ple_off [[buffer(14)]],
    constant float& ple_scale [[buffer(15)]],
    constant int& n_tg [[buffer(16)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    threadgroup float acc[2 * 3 * DM];
    const int gsg = tg * n_sg + sg;
    const int stream = gsg % S;
    float dg[PER_LANE], db[PER_LANE], dc[PER_LANE];
    for (int i = 0; i < PER_LANE; ++i) {{ dg[i] = 0.0f; db[i] = 0.0f; dc[i] = 0.0f; }}
    const int total = R * S;
    for (int idx = gsg; idx < total; idx += n_tg * n_sg) {{
        const int row = idx / S;
        const long off = (long)row * DM;
        const long soff = (long)idx * DM;
        const float mean = stats[2 * row], rstd = stats[2 * row + 1];
        float xh[PER_LANE], g[PER_LANE];
        float s1 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < PER_LANE; ++i) {{
            const int e = lane + 32 * i;
            xh[i] = 0.0f; g[i] = 0.0f;
            if (e < DM) {{
                const float dyv = (flags & 1) ? float(dy_bf[soff + e]) : dy_f[soff + e];
                xh[i] = (x[off + e] - mean) * rstd;
                g[i] = dyv * gamma[e];
                dg[i] += dyv * xh[i];
                db[i] += dyv;
                s1 += g[i];
                s2 += g[i] * xh[i];
            }}
        }}
        s1 = simd_sum(s1) / DM;
        s2 = simd_sum(s2) / DM;
        // With PLE = 64, column e feeds embedding column e % 64, which is
        // `lane` for even i and `lane + 32` for odd i.
        float pa = 0.0f, pb = 0.0f;
        for (int i = 0; i < PER_LANE; ++i) {{
            const int e = lane + 32 * i;
            if (e < DM) {{
                float val = rstd * (g[i] - s1 - xh[i] * s2);
                if (flags & 2) val += dres[soff + e];
                dx[soff + e] = val;
                if (flags & 4) {{
                    const bfloat vb = bfloat(val);
                    dx_bf[soff + e] = vb;
                    dc[i] += float(vb);
                }}
                if (i % 2 == 0) pa += val; else pb += val;
            }}
        }}
        if (flags & 8) {{
            const long poff = ((long)stream * R + row) * ple_stride + ple_off;
            dple[poff + lane] = ple_scale * pa;
            dple[poff + lane + 32] = ple_scale * pb;
        }}
    }}
    for (int i = sg * 32 + lane; i < S * 3 * DM; i += n_sg * 32) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < n_sg; ++k) {{
        if (sg == k) {{
            threadgroup float* a = acc + stream * 3 * DM;
            for (int i = 0; i < PER_LANE; ++i) {{
                const int e = lane + 32 * i;
                if (e < DM) {{ a[e] += dg[i]; a[DM + e] += db[i]; a[2 * DM + e] += dc[i]; }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (int i = sg * 32 + lane; i < S * 3 * DM; i += n_sg * 32) {{
        partial[(long)tg * S * 3 * DM + i] = acc[i];
    }}
}}

// du = dgl * gelu'(u) (exact erf GELU) over S <= 2 interleaved streams, plus
// per-threadgroup column sums of the bf16 du (the up-projection bias
// gradient). dgl/du [R, S, F] bf16, u [R, F] bf16, partial [n_tg, S, F].
kernel void gelu_bwd(
    device const bfloat* dgl [[buffer(0)]],
    device const bfloat* u [[buffer(1)]],
    device bfloat* du [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant int& R [[buffer(4)]],
    constant int& S [[buffer(5)]],
    constant int& F [[buffer(6)]],
    constant int& rows_per_tg [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{{
    const int r0 = tg * rows_per_tg;
    const int r1 = min(R, r0 + rows_per_tg);
    for (int c = tid; c < F; c += tgsize) {{
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int r = r0; r < r1; ++r) {{
            const float x = float(u[(long)r * F + c]);
            const float cdf = 0.5f * (1.0f + erf_approx(x * M_SQRT1_2_F));
            const float pdf = exp(-0.5f * x * x) * 0.3989422804014327f;
            const float gp = cdf + x * pdf;
            const long i0 = ((long)r * S) * F + c;
            const bfloat v0 = bfloat(float(dgl[i0]) * gp);
            du[i0] = v0;
            acc0 += float(v0);
            if (S > 1) {{
                const bfloat v1 = bfloat(float(dgl[i0 + F]) * gp);
                du[i0 + F] = v1;
                acc1 += float(v1);
            }}
        }}
        partial[((long)tg * S) * F + c] = acc0;
        if (S > 1) partial[((long)tg * S + 1) * F + c] = acc1;
    }}
}}

// Causal GQA attention forward. One threadgroup per (sequence, KV head);
// thread (g, t) is query head j * G + g at position t.
// qkv [B, L, QKV_W] bf16; vemb [B, L, ve_stride] fp32 sliced at ve_off
// (ve_off < 0: no value embedding); out [B, L, NH*D] bf16;
// lse [B, NH, L] fp32 in log2 units.
kernel void attn_fwd(
    device const bfloat* qkv [[buffer(0)]],
    device const float* vemb [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    device float* lse [[buffer(3)]],
    constant int& L [[buffer(4)]],
    constant int& ve_stride [[buffer(5)]],
    constant int& ve_off [[buffer(6)]],
    constant float& scale_log2 [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{{
    threadgroup float Ks[MAXL * D];
    threadgroup float Vs[MAXL * D];
    const int b = tg / KVH;
    const int j = tg % KVH;
    const long base = (long)b * L;
    for (int i = tid; i < L * D; i += tgsize) {{
        const int s = i / D, d = i % D;
        const long row = (base + s) * QKV_W;
        Ks[i] = float(qkv[row + K_OFF + j * D + d]);
        float v = float(qkv[row + V_OFF + j * D + d]);
        if (ve_off >= 0) {{
            // Autocast adds the fp32 embedding to the bf16 projection and
            // SDPA then reads v in bf16: round exactly once, here.
            v = float(bfloat(v + vemb[(base + s) * ve_stride + ve_off + j * D + d]));
        }}
        Vs[i] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int g = tid / L;
    const int t = tid % L;
    if (g >= G) return;
    const int h = j * G + g;
    float q[D];
    const long qrow = (base + t) * QKV_W + h * D;
    for (int d = 0; d < D; ++d) q[d] = float(qkv[qrow + d]) * scale_log2;
    float m = -INFINITY, l = 0.0f;
    float acc[D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    for (int s = 0; s <= t; ++s) {{
        float score = 0.0f;
        for (int d = 0; d < D; ++d) score = fma(q[d], Ks[s * D + d], score);
        const float m_new = max(m, score);
        const float corr = exp2(m - m_new);
        const float p = exp2(score - m_new);
        l = l * corr + p;
        for (int d = 0; d < D; ++d) acc[d] = fma(p, Vs[s * D + d], acc[d] * corr);
        m = m_new;
    }}
    const float inv = 1.0f / l;
    const long orow = (base + t) * (NH * D) + h * D;
    for (int d = 0; d < D; ++d) out[orow + d] = bfloat(acc[d] * inv);
    lse[((long)b * NH + h) * L + t] = m + log2(l);
}}

// Attention backward for one cotangent stream per threadgroup; SL streams
// are interleaved per row in dout [B, L, SL, NH*D] and dqkv [B, L, SL, QKV_W].
// The v gradient also lands in dvemb [B, L, SL, dve_stride] at ve_off, which
// is the value-embedding table's gradient. Two streams in one threadgroup
// would share the probabilities but double the per-thread accumulators, and
// measured slower: the spills cost more than the shared exp2.
kernel void attn_bwd(
    device const bfloat* qkv [[buffer(0)]],
    device const float* vemb [[buffer(1)]],
    device const bfloat* out [[buffer(2)]],
    device const float* lse [[buffer(3)]],
    device const bfloat* dout [[buffer(4)]],
    device bfloat* dqkv [[buffer(5)]],
    device float* dvemb [[buffer(6)]],
    constant int& L [[buffer(7)]],
    constant int& ve_stride [[buffer(8)]],
    constant int& ve_off [[buffer(9)]],
    constant int& dve_stride [[buffer(10)]],
    constant float& scale_log2 [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant int& B [[buffer(13)]],
    constant int& SL [[buffer(14)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{{
    threadgroup float Ks[MAXL * D];
    threadgroup float Vs[MAXL * D];
    threadgroup float Dl[G * MAXL];
    const int so = tg / (B * KVH);
    const int b = (tg / KVH) % B;
    const int j = tg % KVH;
    dout += so * (NH * D);
    dqkv += so * QKV_W;
    dvemb += so * dve_stride;
    const long base = (long)b * L;
    for (int i = tid; i < L * D; i += tgsize) {{
        const int s = i / D, d = i % D;
        const long row = (base + s) * QKV_W;
        Ks[i] = float(qkv[row + K_OFF + j * D + d]);
        float v = float(qkv[row + V_OFF + j * D + d]);
        if (ve_off >= 0) {{
            v = float(bfloat(v + vemb[(base + s) * ve_stride + ve_off + j * D + d]));
        }}
        Vs[i] = v;
    }}
    const bool active = tid < (uint)(G * L);
    // Inactive lanes alias the last real (g, t) so every address stays
    // valid; nothing they compute is stored.
    const int lin = active ? (int)tid : G * L - 1;
    const int g = lin / L;
    const int t = lin % L;
    const int h = j * G + g;
    const float lse_t = lse[((long)b * NH + h) * L + t];
    {{
        const long orow = (base + t) * (NH * D) + h * D;
        const long drow = ((base + t) * SL) * (NH * D) + h * D;
        float delta = 0.0f;
        for (int d = 0; d < D; ++d) delta = fma(float(dout[drow + d]), float(out[orow + d]), delta);
        if (active) Dl[g * L + t] = delta;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: dq for query (g, t).
    {{
        float q[D], dov[D], dq[D];
        const long qrow = (base + t) * QKV_W + h * D;
        const long drow = ((base + t) * SL) * (NH * D) + h * D;
        for (int d = 0; d < D; ++d) {{
            q[d] = float(qkv[qrow + d]) * scale_log2;
            dov[d] = float(dout[drow + d]);
            dq[d] = 0.0f;
        }}
        const float delta = Dl[g * L + t];
        for (int s = 0; s <= t; ++s) {{
            float score = 0.0f, dp = 0.0f;
            for (int d = 0; d < D; ++d) {{
                score = fma(q[d], Ks[s * D + d], score);
                dp = fma(dov[d], Vs[s * D + d], dp);
            }}
            const float ds = exp2(score - lse_t) * (dp - delta);
            for (int d = 0; d < D; ++d) dq[d] = fma(ds, Ks[s * D + d], dq[d]);
        }}
        if (active) {{
            const long dqrow = ((base + t) * SL) * QKV_W + h * D;
            for (int d = 0; d < D; ++d) dqkv[dqrow + d] = bfloat(dq[d] * scale);
        }}
    }}

    // Phase 2: this query head's share of dk/dv for key s = t.
    const int s = t;
    float k[D], v[D], dk[D], dv[D];
    for (int d = 0; d < D; ++d) {{
        k[d] = Ks[s * D + d];
        v[d] = Vs[s * D + d];
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }}
    for (int tq = s; tq < L; ++tq) {{
        const long qrow = (base + tq) * QKV_W + h * D;
        const long drow = ((base + tq) * SL) * (NH * D) + h * D;
        float qv[D], dot[D];
        float score = 0.0f, dp = 0.0f;
        for (int d = 0; d < D; ++d) {{
            qv[d] = float(qkv[qrow + d]) * scale_log2;
            dot[d] = float(dout[drow + d]);
            score = fma(qv[d], k[d], score);
            dp = fma(dot[d], v[d], dp);
        }}
        const float p = exp2(score - lse[((long)b * NH + h) * L + tq]);
        const float ds = p * (dp - Dl[g * L + tq]);
        for (int d = 0; d < D; ++d) {{
            dv[d] = fma(p, dot[d], dv[d]);
            dk[d] = fma(ds, qv[d], dk[d]);
        }}
    }}
    // Reduce over the G query heads in the (now free) K/V tiles.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < L * D; i += tgsize) {{ Ks[i] = 0.0f; Vs[i] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int gg = 0; gg < G; ++gg) {{
        if (active && g == gg) {{
            for (int d = 0; d < D; ++d) {{
                Ks[s * D + d] += dk[d];
                Vs[s * D + d] += dv[d];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    // dk was accumulated against q * scale * log2(e); its true factor is scale.
    const float k_scale = scale / scale_log2;
    for (int i = tid; i < L * D; i += tgsize) {{
        const int ss = i / D, d = i % D;
        const long row = ((base + ss) * SL) * QKV_W;
        dqkv[row + K_OFF + j * D + d] = bfloat(Ks[i] * k_scale);
        const float dvv = Vs[i];
        dqkv[row + V_OFF + j * D + d] = bfloat(dvv);
        if (ve_off >= 0) {{
            dvemb[((base + ss) * SL) * dve_stride + ve_off + j * D + d] = dvv;
        }}
    }}
}}

// Cached decode attention for a merged wave: every row appends T tokens at
// its own position and attends to its own prefix. One thread per (row, KV
// head, query head in group, new token). Keys before the row's position come
// from the fp16 cache; the T new ones come straight from qkv, rounded to fp16
// exactly as the cache would hold them, and the g == 0 thread writes them
// back. No thread reads a slot another thread of this dispatch writes.
// qkv [B*T, QKV_W] bf16; vemb [B*T, ve_stride] fp32 at ve_off (< 0: none);
// kc/vc [capacity, KVH, max_len, D] half (one layer); positions [B] int64;
// out [B*T, NH*D] bf16.
kernel void attn_decode(
    device const bfloat* qkv [[buffer(0)]],
    device const float* vemb [[buffer(1)]],
    device half* kc [[buffer(2)]],
    device half* vc [[buffer(3)]],
    device const long* positions [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant int& B [[buffer(6)]],
    constant int& T [[buffer(7)]],
    constant int& max_len [[buffer(8)]],
    constant int& ve_stride [[buffer(9)]],
    constant int& ve_off [[buffer(10)]],
    constant float& scale_log2 [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{{
    const int t = gid % T;
    const int g = (gid / T) % G;
    const int j = (gid / (T * G)) % KVH;
    const int b = gid / (T * G * KVH);
    if (b >= B) return;
    const int pos0 = (int)positions[b];
    const long cache_row = ((long)b * KVH + j) * max_len;
    const long tok0 = (long)b * T;
    if (g == 0) {{
        const long src = (tok0 + t) * QKV_W;
        const int slot = min(pos0 + t, max_len - 1);
        for (int d = 0; d < D; ++d) {{
            kc[(cache_row + slot) * D + d] = half(float(qkv[src + K_OFF + j * D + d]));
            float v = float(qkv[src + V_OFF + j * D + d]);
            if (ve_off >= 0) v += vemb[(tok0 + t) * ve_stride + ve_off + j * D + d];
            vc[(cache_row + slot) * D + d] = half(v);
        }}
    }}
    const int h = j * G + g;
    float q[D];
    const long qrow = (tok0 + t) * QKV_W + h * D;
    for (int d = 0; d < D; ++d) q[d] = float(qkv[qrow + d]) * scale_log2;
    float m = -INFINITY, l = 0.0f;
    float acc[D];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    const int keys = pos0 + t + 1;
    for (int s = 0; s < keys; ++s) {{
        float k[D], v[D];
        if (s < pos0) {{
            const long at = (cache_row + s) * D;
            for (int d = 0; d < D; ++d) {{ k[d] = float(kc[at + d]); v[d] = float(vc[at + d]); }}
        }} else {{
            const long src = (tok0 + (s - pos0)) * QKV_W;
            for (int d = 0; d < D; ++d) {{
                k[d] = float(half(float(qkv[src + K_OFF + j * D + d])));
                float vv = float(qkv[src + V_OFF + j * D + d]);
                if (ve_off >= 0) vv += vemb[(tok0 + (s - pos0)) * ve_stride + ve_off + j * D + d];
                v[d] = float(half(vv));
            }}
        }}
        float score = 0.0f;
        for (int d = 0; d < D; ++d) score = fma(q[d], k[d], score);
        const float m_new = max(m, score);
        const float corr = exp2(m - m_new);
        const float p = exp2(score - m_new);
        l = l * corr + p;
        for (int d = 0; d < D; ++d) acc[d] = fma(p, v[d], acc[d] * corr);
        m = m_new;
    }}
    const float inv = 1.0f / l;
    const long orow = (tok0 + t) * (NH * D) + h * D;
    for (int d = 0; d < D; ++d) out[orow + d] = bfloat(acc[d] * inv);
}}
"""


_LIBRARIES: dict[tuple[int, ...], object] = {}

# Every optimizer step in the process, counted by a global post-step hook
# installed with the first MetalTrunk; see MetalTrunk._bf16_weights.
_OPTIMIZER_STEPS = [0]
_STEP_HOOK = None


def _count_optimizer_steps() -> None:
    global _STEP_HOOK
    if _STEP_HOOK is None:
        def bump(optimizer, args, kwargs):
            _OPTIMIZER_STEPS[0] += 1

        _STEP_HOOK = register_optimizer_step_post_hook(bump)


def _library(key: tuple[int, ...]):
    library = _LIBRARIES.get(key)
    if library is None:
        library = torch.mps.compile_shader(_source(*key))
        _LIBRARIES[key] = library
    return library


def _wgrad(dy: torch.Tensor, x: torch.Tensor, streams: int) -> torch.Tensor:
    """dy [R, S*N] bf16, x [R, K] bf16 -> [S, N, K] fp32.

    MPS runs a long-K GEMM at a ~0.7 ms floor whatever the output size; a
    split-K batch restores the parallelism (2.5-4.7x at these shapes).
    """

    rows, width = dy.shape
    depth = x.shape[1]
    main = (rows // _WGRAD_SPLITS) * _WGRAD_SPLITS
    out = torch.bmm(
        dy[:main].view(_WGRAD_SPLITS, main // _WGRAD_SPLITS, width).transpose(1, 2),
        x[:main].view(_WGRAD_SPLITS, main // _WGRAD_SPLITS, depth),
    ).sum(0, dtype=torch.float32)
    if main < rows:
        out += (dy[main:].t() @ x[main:]).float()
    return out.view(streams, width // streams, depth)


# Parameter order within one layer of trunk_parameters().
_LN1_W, _LN1_B, _WQKV, _WOUT, _LN2_W, _LN2_B, _W1, _B1, _W2, _B2 = range(10)
PER_LAYER = 10


def trunk_parameters(model) -> list[torch.nn.Parameter]:
    """Every parameter the trunk reads, in the order MetalTrunk expects."""

    params: list[torch.nn.Parameter] = []
    for block in model.blocks:
        params += [
            block.ln_attn.weight, block.ln_attn.bias,
            block.qkv_proj.weight, block.out_proj.weight,
            block.ln_mlp.weight, block.ln_mlp.bias,
            block.mlp[0].weight, block.mlp[0].bias,
            block.mlp[2].weight, block.mlp[2].bias,
        ]
    params += [model.final_norm.weight, model.final_norm.bias]
    return params


@dataclass
class _Saved:
    batch: int
    length: int
    layers: list[dict]
    weights: list[tuple[torch.Tensor, ...]]
    x_final: torch.Tensor
    stats_final: torch.Tensor
    values: torch.Tensor | None
    value_width: int
    ple_width: int
    has_ple: bool


class MetalTrunk:
    """The fused trunk for one model configuration."""

    def __init__(self, config: SeqModelConfig, eps: float = 1e-5) -> None:
        reason = unsupported_reason(config)
        if reason is not None:
            raise ValueError(f"Metal trunk unsupported: {reason}.")
        self.config = config
        self.eps = eps
        self.layers = config.n_layers
        self.dm = config.d_model
        self.head_dim = config.head_dim
        self.kv_heads = config.kv_heads
        self.n_heads = config.n_heads
        self.ple_dim = config.per_layer_embedding_dim
        self.ple_scale = (
            1.0 / math.sqrt(math.ceil(self.dm / self.ple_dim)) if self.ple_dim else 0.0
        )
        self.value_index = {
            layer: index for index, layer in enumerate(config.value_embedding_layers)
        }
        self.kv_width = self.kv_heads * self.head_dim
        self.key = (self.dm, self.n_heads, self.kv_heads, self.head_dim, self.ple_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self._weight_key: tuple | None = None
        self._weights: list[tuple[torch.Tensor, ...]] = []
        _count_optimizer_steps()

    @property
    def lib(self):
        return _library(self.key)

    def _bf16_weights(self, params: Sequence[torch.Tensor]) -> list[tuple[torch.Tensor, ...]]:
        """Per layer (wqkv, wout, w1, w2, b1, b2) in bf16, reused while valid.

        Casting is ~1.25 ms of dispatch per call, paid on every rollout wave
        and every microbatch, so the copy is cached. The key needs both parts:
        version counters catch ``load_state_dict`` and any ordinary in-place
        write, but the fused MPS Adam step does not bump them, so every
        optimizer step also advances ``_OPTIMIZER_STEPS``. (A ``.data`` write
        would evade both; nothing touches trunk weights that way after
        construction.)
        """

        layer_params = [
            params[PER_LAYER * layer + i]
            for layer in range(self.layers)
            for i in (_WQKV, _WOUT, _W1, _W2, _B1, _B2)
        ]
        key = (_OPTIMIZER_STEPS[0], tuple((id(p), p._version) for p in layer_params))
        if self._weight_key != key:
            cast = [p.detach().to(torch.bfloat16) for p in layer_params]
            self._weights = [
                tuple(cast[6 * layer: 6 * layer + 6]) for layer in range(self.layers)
            ]
            self._weight_key = key
        return self._weights

    def _attention_group(self, length: int) -> int:
        group = self.n_heads // self.kv_heads
        return ((group * length + 31) // 32) * 32

    # -- kernels ---------------------------------------------------------

    def _add_ln(self, rows, gamma, beta, *, x=None, delta=None, ple=None,
                layer=0, write_x=False, fp32_out=False):
        dev = gamma.device
        flags = 0
        dummy_b = torch.empty(1, device=dev, dtype=torch.bfloat16)
        if x is not None:
            flags |= 1
        if delta is not None:
            flags |= 2
        if ple is not None:
            flags |= 4
        x_out = torch.empty(rows, self.dm, device=dev) if write_x else gamma
        if write_x:
            flags |= 8
        if fp32_out:
            flags |= 16
            y = torch.empty(rows, self.dm, device=dev)
            y_bf, y_f = dummy_b, y
        else:
            y = torch.empty(rows, self.dm, device=dev, dtype=torch.bfloat16)
            y_bf, y_f = y, gamma
        stats = torch.empty(rows, 2, device=dev)
        self.lib.add_ln_fwd(
            x if x is not None else gamma,
            delta if delta is not None else dummy_b,
            ple if ple is not None else gamma,
            gamma, beta, x_out, y_bf, y_f, stats,
            rows, flags, ple.shape[-1] if ple is not None else 0,
            layer * self.ple_dim, float(self.ple_scale), float(self.eps),
            threads=rows * 32, group_size=256,
        )
        return (x_out if write_x else None), y, stats

    def _ln_backward(self, dy, x, stats, gamma, streams, *, dres=None,
                     want_bf=False, dple=None, layer=0):
        total = dy.shape[0]
        rows = total // streams
        dev = x.device
        flags = 0
        dummy_b = torch.empty(1, device=dev, dtype=torch.bfloat16)
        if dy.dtype == torch.bfloat16:
            flags |= 1
            dy_bf, dy_f = dy, stats
        else:
            dy_bf, dy_f = dummy_b, dy
        if dres is not None:
            flags |= 2
        dx = torch.empty(total, self.dm, device=dev)
        dx_bf = (
            torch.empty(total, self.dm, device=dev, dtype=torch.bfloat16)
            if want_bf
            else dummy_b
        )
        if want_bf:
            flags |= 4
        if dple is not None:
            flags |= 8
        n_tg = min(_LN_THREADGROUPS, max(1, (total + _LN_SIMDGROUPS - 1) // _LN_SIMDGROUPS))
        partial = torch.empty(n_tg, streams, 3, self.dm, device=dev)
        self.lib.ln_bwd(
            dy_bf, dy_f, x, stats, gamma,
            dres if dres is not None else stats,
            dx, dx_bf, dple if dple is not None else stats, partial,
            rows, streams, flags, dple.shape[-1] if dple is not None else 0,
            layer * self.ple_dim, float(self.ple_scale), n_tg,
            threads=n_tg * _LN_SIMDGROUPS * 32, group_size=_LN_SIMDGROUPS * 32,
        )
        return dx, (dx_bf if want_bf else None), partial.sum(0)

    def _gelu_backward(self, dgl, u, streams):
        rows, width = u.shape
        du = torch.empty_like(dgl)
        n_tg = (rows + _GELU_ROWS - 1) // _GELU_ROWS
        partial = torch.empty(n_tg, streams, width, device=u.device)
        self.lib.gelu_bwd(
            dgl, u, du, partial, rows, streams, width, _GELU_ROWS,
            threads=n_tg * 256, group_size=256,
        )
        return du, partial.sum(0)

    def _attention(self, qkv, values, layer):
        batch, length, _ = qkv.shape
        out = torch.empty(batch, length, self.n_heads * self.head_dim,
                          device=qkv.device, dtype=torch.bfloat16)
        lse = torch.empty(batch, self.n_heads, length, device=qkv.device)
        index = self.value_index.get(layer)
        ve_off = -1 if index is None or values is None else index * self.kv_width
        group = self._attention_group(length)
        self.lib.attn_fwd(
            qkv, values if ve_off >= 0 else lse, out, lse, length,
            values.shape[-1] if ve_off >= 0 else 0, ve_off,
            self.scale * math.log2(math.e),
            threads=batch * self.kv_heads * group, group_size=group,
        )
        return out, lse, ve_off

    def _attention_backward(self, qkv, out, lse, dout, values, ve_off, dvalues):
        batch, length, streams, _ = dout.shape
        dqkv = torch.empty(batch, length, streams, qkv.shape[-1],
                           device=qkv.device, dtype=torch.bfloat16)
        use_values = ve_off >= 0
        group = self._attention_group(length)
        self.lib.attn_bwd(
            qkv, values if use_values else lse, out, lse, dout, dqkv,
            dvalues if use_values else lse, length,
            values.shape[-1] if use_values else 0, ve_off,
            dvalues.shape[-1] if use_values else 0,
            self.scale * math.log2(math.e), self.scale, batch, streams,
            threads=streams * batch * self.kv_heads * group, group_size=group,
        )
        return dqkv

    # -- trunk -----------------------------------------------------------

    def forward(self, x0, ple, values, params: Sequence[torch.Tensor], *, save: bool):
        """x0 [B, L, DM], ple [B, L, layers, 64] | None, values [B, L, nv, kv] | None.

        Returns the final-norm output [B, L, DM] fp32 and, with ``save``, the
        context ``backward`` needs.
        """

        batch, length, _ = x0.shape
        if length > MAX_LENGTH:
            raise ValueError(f"Metal trunk sequences are limited to {MAX_LENGTH}.")
        rows = batch * length
        x = x0.reshape(rows, self.dm).float().contiguous()
        ple_flat = (
            ple.reshape(rows, -1).float().contiguous() if ple is not None and self.ple_dim else None
        )
        ve = (
            values.reshape(batch, length, -1).float().contiguous()
            if values is not None and self.value_index
            else None
        )
        layers: list[dict] = []
        weights: list[tuple[torch.Tensor, ...]] = []
        cast = self._bf16_weights(params)
        _, a, stats1 = self._add_ln(rows, params[_LN1_W], params[_LN1_B], x=x)
        for layer in range(self.layers):
            p = params[PER_LAYER * layer: PER_LAYER * (layer + 1)]
            wqkv, wout, w1, w2, b1, b2 = cast[layer]
            qkv = F.linear(a, wqkv)
            o, lse, ve_off = self._attention(qkv.view(batch, length, -1), ve, layer)
            o = o.view(rows, -1)
            x1, m, stats2 = self._add_ln(
                rows, p[_LN2_W], p[_LN2_B], x=x, delta=F.linear(o, wout), write_x=True
            )
            u = F.linear(m, w1, b1)
            gl = F.gelu(u)
            down = F.linear(gl, w2, b2)
            last = layer == self.layers - 1
            base = PER_LAYER * (self.layers if last else layer + 1)
            x_next, a_next, stats_next = self._add_ln(
                rows, params[base], params[base + 1], x=x1, delta=down,
                ple=ple_flat, layer=layer, write_x=True, fp32_out=last,
            )
            if save:
                layers.append(dict(x=x, a=a, stats1=stats1, qkv=qkv, o=o, lse=lse,
                                   ve_off=ve_off, x1=x1, m=m, stats2=stats2, u=u, gl=gl))
                weights.append((wqkv, wout, w1, w2))
            x, a, stats1 = x_next, a_next, stats_next
        hidden = a.view(batch, length, self.dm)
        if not save:
            return hidden, None
        saved = _Saved(
            batch=batch, length=length, layers=layers, weights=weights,
            x_final=x, stats_final=stats1, values=ve,
            value_width=ve.shape[-1] if ve is not None else 0,
            ple_width=ple_flat.shape[-1] if ple_flat is not None else 0,
            has_ple=ple_flat is not None,
        )
        return hidden, saved

    def backward(self, saved: _Saved, dh: torch.Tensor, params: Sequence[torch.Tensor]):
        """dh [B, L, S, DM] (S cotangent streams interleaved per row).

        Returns dx0 [B, L, S, DM], dple [S, B, L, layers, 64] | None,
        dvalues [B, L, S, nv, kv] | None, and per-parameter [S, ...] fp32
        gradients aligned with ``params``.
        """

        batch, length, streams, _ = dh.shape
        rows = batch * length
        dev = dh.device
        grads: list[torch.Tensor | None] = [None] * len(params)
        dple = (
            torch.empty(streams, rows, saved.ple_width, device=dev) if saved.has_ple else None
        )
        dvalues = (
            torch.zeros(batch, length, streams, saved.value_width, device=dev)
            if saved.values is not None
            else None
        )
        final = PER_LAYER * self.layers
        dx, dx_bf, sums = self._ln_backward(
            dh.reshape(rows * streams, self.dm).float().contiguous(),
            saved.x_final, saved.stats_final, params[final], streams,
            want_bf=True, dple=dple, layer=self.layers - 1,
        )
        grads[final], grads[final + 1] = sums[:, 0], sums[:, 1]
        down_bias = sums[:, 2]
        for layer in reversed(range(self.layers)):
            sv = saved.layers[layer]
            wqkv, wout, w1, w2 = saved.weights[layer]
            p0 = PER_LAYER * layer
            # MLP: dx_bf is bf16(d residual), the down projection's output grad.
            dgl = dx_bf @ w2
            grads[p0 + _W2] = _wgrad(dx_bf.view(rows, -1), sv["gl"], streams)
            grads[p0 + _B2] = down_bias
            du, grads[p0 + _B1] = self._gelu_backward(dgl, sv["u"], streams)
            dm = du @ w1
            grads[p0 + _W1] = _wgrad(du.view(rows, -1), sv["m"], streams)
            dx1, dx1_bf, sums = self._ln_backward(
                dm, sv["x1"], sv["stats2"], params[p0 + _LN2_W], streams,
                dres=dx, want_bf=True,
            )
            grads[p0 + _LN2_W], grads[p0 + _LN2_B] = sums[:, 0], sums[:, 1]
            # Attention.
            do = dx1_bf @ wout
            grads[p0 + _WOUT] = _wgrad(dx1_bf.view(rows, -1), sv["o"], streams)
            dqkv = self._attention_backward(
                sv["qkv"].view(batch, length, -1), sv["o"].view(batch, length, -1),
                sv["lse"], do.view(batch, length, streams, -1), saved.values,
                sv["ve_off"], dvalues,
            ).view(rows * streams, -1)
            da = dqkv @ wqkv
            grads[p0 + _WQKV] = _wgrad(dqkv.view(rows, -1), sv["a"], streams)
            first = layer == 0
            dx, dx_bf, sums = self._ln_backward(
                da, sv["x"], sv["stats1"], params[p0 + _LN1_W], streams,
                dres=dx1, want_bf=not first,
                dple=None if first else dple, layer=layer - 1,
            )
            grads[p0 + _LN1_W], grads[p0 + _LN1_B] = sums[:, 0], sums[:, 1]
            down_bias = sums[:, 2]
        return (
            dx.view(batch, length, streams, self.dm),
            None if dple is None else dple.view(streams, batch, length, self.layers, self.ple_dim),
            None if dvalues is None else dvalues.view(
                batch, length, streams, len(self.value_index), self.kv_width
            ),
            grads,
        )


    # -- cached decode (rollout) ----------------------------------------

    def decode(self, x0, ple, values, cache, positions, params: Sequence[torch.Tensor]):
        """One merged-wave decode step through every layer.

        x0 [B, T, DM] embeddings of the appended tokens, ple/values their
        conditioning, ``positions`` [B] each row's write position for token 0.
        Writes the new K/V into ``cache`` (fp16, dense rows 0..B-1) and returns
        the final-norm output for every appended token, [B, T, DM] fp32.
        """

        batch, steps, _ = x0.shape
        rows = batch * steps
        x = x0.reshape(rows, self.dm).float().contiguous()
        ple_flat = (
            ple.reshape(rows, -1).float().contiguous() if ple is not None and self.ple_dim else None
        )
        ve = (
            values.reshape(rows, -1).float().contiguous()
            if values is not None and self.value_index
            else None
        )
        positions = positions.to(torch.int64).contiguous()
        cast = self._bf16_weights(params)
        _, a, _ = self._add_ln(rows, params[_LN1_W], params[_LN1_B], x=x)
        for layer in range(self.layers):
            p = params[PER_LAYER * layer: PER_LAYER * (layer + 1)]
            wqkv, wout, w1, w2, b1, b2 = cast[layer]
            qkv = F.linear(a, wqkv)
            index = self.value_index.get(layer)
            ve_off = -1 if index is None or ve is None else index * self.kv_width
            o = torch.empty(rows, self.n_heads * self.head_dim, device=x.device,
                            dtype=torch.bfloat16)
            group = self.n_heads // self.kv_heads
            self.lib.attn_decode(
                qkv, ve if ve_off >= 0 else x, cache.k[layer], cache.v[layer],
                positions, o, batch, steps, cache.max_len,
                ve.shape[-1] if ve_off >= 0 else 0, ve_off,
                self.scale * math.log2(math.e),
                threads=batch * self.kv_heads * group * steps, group_size=256,
            )
            x1, m, _ = self._add_ln(
                rows, p[_LN2_W], p[_LN2_B], x=x, delta=F.linear(o, wout), write_x=True
            )
            down = F.linear(F.gelu(F.linear(m, w1, b1)), w2, b2)
            last = layer == self.layers - 1
            base = PER_LAYER * (self.layers if last else layer + 1)
            x, a, _ = self._add_ln(
                rows, params[base], params[base + 1], x=x1, delta=down,
                ple=ple_flat, layer=layer, write_x=True, fp32_out=last,
            )
        return a.view(batch, steps, self.dm)

    def decode_supported(self, cache) -> bool:
        return (
            cache.dtype == torch.float16
            and cache.max_len <= MAX_LENGTH + 1
            and all(k.is_contiguous() for k in cache.k)
            and all(v.is_contiguous() for v in cache.v)
        )


class _MetalTrunkFunction(torch.autograd.Function):
    """Single-cotangent autograd wrapper: a drop-in for the PyTorch trunk."""

    @staticmethod
    def forward(ctx, trunk: MetalTrunk, x0, ple, values, *params):
        hidden, saved = trunk.forward(x0, ple, values, params, save=True)
        ctx.trunk = trunk
        ctx.saved = saved
        ctx.params = params
        ctx.has_ple = ple is not None
        ctx.has_values = values is not None
        return hidden

    @staticmethod
    def backward(ctx, dh):
        dx0, dple, dvalues, grads = ctx.trunk.backward(
            ctx.saved, dh.unsqueeze(2), ctx.params
        )
        ctx.saved = None
        return (
            None,
            dx0[:, :, 0],
            dple[0] if ctx.has_ple and dple is not None else None,
            dvalues[:, :, 0] if ctx.has_values and dvalues is not None else None,
            *(g[0] for g in grads),
        )


def run_trunk(model, trunk: MetalTrunk, tokens: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """forward_hidden's fused branch: embeddings in PyTorch, trunk in Metal."""

    ple, values = model._prepare_layer_conditioning(tokens, x0, 0)
    params = trunk_parameters(model)
    if torch.is_grad_enabled() and (
        x0.requires_grad or any(p.requires_grad for p in params)
    ):
        return _MetalTrunkFunction.apply(trunk, x0, ple, values, *params)
    with torch.no_grad():
        hidden, _ = trunk.forward(x0, ple, values, params, save=False)
    return hidden


class DualTrunkPass:
    """One trunk forward whose backward serves several losses at once.

    ``hidden`` is a leaf: losses built on it reach the heads through ordinary
    autograd, the trunk backward runs once for every loss together, and the
    embedding graph below the trunk is replayed per loss. ``gradients``
    returns exactly what one ``torch.autograd.grad(loss, parameters,
    allow_unused=True)`` per loss would, up to bf16 rounding.
    """

    def __init__(self, model, trunk: MetalTrunk, tokens: torch.Tensor) -> None:
        self.model = model
        self.trunk = trunk
        self.x0 = model.embed(tokens)
        self.ple, self.values = model._prepare_layer_conditioning(tokens, self.x0, 0)
        self.params = trunk_parameters(model)
        with torch.no_grad():
            hidden, self.saved = trunk.forward(
                self.x0, self.ple, self.values, self.params, save=True
            )
        self.hidden = hidden.requires_grad_()

    def gradients(
        self,
        losses: Sequence[torch.Tensor | None],
        parameters: Sequence[torch.Tensor],
    ) -> list[list[torch.Tensor | None] | None]:
        trunk_slot = {id(p): index for index, p in enumerate(self.params)}
        others = [p for p in parameters if id(p) not in trunk_slot]
        active = [
            index for index, loss in enumerate(losses)
            if loss is not None and loss.requires_grad
        ]
        results: list[list[torch.Tensor | None] | None] = [None] * len(losses)
        if not active:
            return results
        head = []
        for position, index in enumerate(active):
            head.append(torch.autograd.grad(
                losses[index], [self.hidden, *others],
                retain_graph=position + 1 < len(active), allow_unused=True,
            ))
        dh = torch.stack(
            [g[0] if g[0] is not None else torch.zeros_like(self.hidden) for g in head],
            dim=2,
        )
        dx0, dple, dvalues, trunk_grads = self.trunk.backward(self.saved, dh, self.params)
        self.saved = None
        outputs = [self.x0]
        if self.ple is not None and dple is not None:
            outputs.append(self.ple)
        if self.values is not None and dvalues is not None:
            outputs.append(self.values)
        reachable = [o.requires_grad for o in outputs]
        for position, index in enumerate(active):
            cotangents = [dx0[:, :, position]]
            if self.ple is not None and dple is not None:
                cotangents.append(dple[position])
            if self.values is not None and dvalues is not None:
                cotangents.append(dvalues[:, :, position])
            pairs = [(o, c) for o, c, r in zip(outputs, cotangents, reachable) if r]
            embedding = (
                torch.autograd.grad(
                    [o for o, _ in pairs], others, [c for _, c in pairs],
                    retain_graph=position + 1 < len(active), allow_unused=True,
                )
                if pairs and others
                else (None,) * len(others)
            )
            other_grads = {}
            for p, from_head, from_embedding in zip(others, head[position][1:], embedding):
                if from_head is None:
                    other_grads[id(p)] = from_embedding
                elif from_embedding is None:
                    other_grads[id(p)] = from_head
                else:
                    other_grads[id(p)] = from_head + from_embedding
            row: list[torch.Tensor | None] = []
            for p in parameters:
                slot = trunk_slot.get(id(p))
                row.append(
                    trunk_grads[slot][position] if slot is not None else other_grads[id(p)]
                )
            results[index] = row
        return results
