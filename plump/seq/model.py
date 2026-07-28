"""Schema-v6 causal sequence model with per-position heads and KV caching."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import NUM_CARDS, TOKEN_WIDTH, SeqModelConfig
from .kv import KVCache

# Auxiliary heads, selectable one by one in forward_full. They differ by an order
# of magnitude in cost -- "owner" is a d -> 5d projection plus a [B, L, 5, d] x
# [52, d] einsum, the other three are single small matmuls on [B, L, d] -- so a
# caller whose loss weights owner at zero should not be paying for it.
AUX_HEADS = ("trick", "suit", "owner", "bid_hit")


@dataclass
class SeqStepOutput:
    """Single-position readout used on the rollout hot path."""

    hidden: torch.Tensor      # [B, d_model]
    bid_logits: torch.Tensor  # [B, bid_count]
    card_logits: torch.Tensor  # [B, NUM_CARDS]
    value: torch.Tensor       # [B]


@dataclass
class SeqOutput:
    """Per-position readout over a full causal forward."""

    hidden: torch.Tensor       # [B, L, d_model]
    bid_logits: torch.Tensor   # [B, L, bid_count]
    card_logits: torch.Tensor  # [B, L, NUM_CARDS]
    value: torch.Tensor        # [B, L]
    trick_logits: Optional[torch.Tensor] = None  # [B, L, max_players, bid_count]
    suit_logits: Optional[torch.Tensor] = None   # [B, L, max_players, 4]
    owner_logits: Optional[torch.Tensor] = None  # [B, L, NUM_CARDS, owner_classes]
    bid_hit_logits: Optional[torch.Tensor] = None  # [B, L, max_players]


class DecoderBlock(nn.Module):
    def __init__(self, config: SeqModelConfig):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.head_group = config.n_heads // config.kv_heads
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.ln_attn = nn.LayerNorm(d)
        # One fused projection instead of three: the rollout hot path issues a
        # forward per game event, so kernel-launch count dominates its cost.
        self.qkv_proj = nn.Linear(
            d, (self.n_heads + 2 * self.kv_heads) * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.ln_mlp = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, d),
        )
        self.dropout = nn.Dropout(config.dropout)

    def _qkv(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = h.shape
        fused = self.qkv_proj(h)
        q, k, v = fused.split(
            [
                self.n_heads * self.head_dim,
                self.kv_heads * self.head_dim,
                self.kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    @property
    def _gqa(self) -> bool:
        # SDPA broadcasts the KV heads internally; materialising the expanded
        # K/V would undo the memory saving GQA exists for.
        return self.kv_heads != self.n_heads

    def _decode_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Attention for ``T`` new query positions, KV heads left grouped.

        Written out rather than delegated to SDPA because on MPS the fused
        kernel is markedly slower at short query lengths, and its GQA path
        slower still -- which would make grouped KV a memory saving paid for in
        wall time. Grouping the *query* heads instead keeps K/V at
        ``kv_heads``, so GQA wins on both axes. Softmax runs in fp32 (as SDPA
        does) so an fp16 cache does not cost accuracy.

        With ``T > 1`` the new queries sit at the end of the cached prefix, so
        query ``i`` may see keys up to ``len(k) - T + i``: a rectangular causal
        mask over the tail, not a square one.
        """

        batch, _, length, _ = q.shape
        # Merge the query's head-group and time axes so this stays a 4-D
        # batched matmul against un-expanded K/V. Broadcasting a group axis
        # against [B, kv_heads, L, D] instead makes MPS materialise the
        # expanded K/V -- measured 3x slower and 2x the memory, which is the
        # whole point of not using SDPA's GQA path here.
        grouped = q.reshape(
            batch, self.kv_heads, self.head_group * length, self.head_dim
        )
        scores = (grouped @ k.transpose(-1, -2)) * self.scale
        if length > 1:
            total = k.shape[2]
            offsets = torch.arange(length, device=q.device) + (total - length)
            blocked = torch.arange(total, device=q.device) > offsets[:, None]
            # Row g * length + t of the merged axis is query t of group g.
            scores = scores.masked_fill(
                blocked.repeat(self.head_group, 1), float("-inf")
            )
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        return (weights @ v).view(batch, self.n_heads, length, self.head_dim)

    def _finish(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = attn.shape
        merged = attn.transpose(1, 2).reshape(batch, length, -1)
        x = x + self.dropout(self.out_proj(merged))
        return x + self.dropout(self.mlp(self.ln_mlp(x)))

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._qkv(self.ln_attn(x))
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self._gqa
        )
        return self._finish(x, attn)

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: KVCache,
        layer: int,
        slots: torch.Tensor | None,
        start: int,
    ) -> torch.Tensor:
        """Causal attention against the cached prefix; writes new K/V.

        ``slots=None`` uses the dense rows 0..B-1 fast path (zero-copy reads).
        Attention runs in the cache dtype so fp16 caches avoid a cast copy.
        """

        length = x.shape[1]
        q, k, v = self._qkv(self.ln_attn(x))
        cache.write_range(layer, slots, start, k, v)
        if start == 0 and length > 1:
            # Prefill: no prefix to attend to, so the square causal kernel is
            # exactly right and SDPA is worth using at this length.
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self._gqa
            )
        else:
            k_all, v_all = cache.read(
                layer, slots, start + length, count=x.shape[0]
            )
            attn = self._decode_attention(q.to(cache.dtype), k_all, v_all).to(
                x.dtype
            )
        return self._finish(x, attn)


class SeqPlumpModel(nn.Module):
    def __init__(self, config: SeqModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        # All twelve token slots share one embedding table with per-slot id
        # offsets. Twelve separate lookups plus eleven adds would be ~23
        # kernels per forward, and the rollout runs one forward per game
        # event, so that overhead dominates at small batch sizes.
        sizes = config.slot_vocab_sizes
        offsets = torch.tensor(
            [sum(sizes[:index]) for index in range(len(sizes))], dtype=torch.long
        )
        self.register_buffer("slot_offsets", offsets, persistent=False)
        self.slot_embedding = nn.Embedding(int(sum(sizes)), d)
        self.pos_embedding = nn.Embedding(config.max_seq_len, d)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(d)

        self.bid_head = nn.Linear(d, config.bid_count)
        self.card_head = nn.Linear(d, NUM_CARDS)
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.trick_count_head = nn.Linear(d, config.max_players * config.bid_count)
        # Both belief heads emit one logit per (relative seat, class) and let the
        # loss mask seats a shape does not have. The alternative -- 4 (resp. 1)
        # logits plus a seat embedding, evaluated once per seat -- would turn a
        # single d x 20 matmul into P passes over [B, L, d] for at most five
        # seats that are already distinguishable by their relative index.
        self.suit_presence_head = nn.Linear(d, config.max_players * 4)
        # An MLP, where suit presence is linear, because the two ask structurally
        # different questions of the trunk. "Does seat p still hold a spade" is
        # monotone in an accumulated feature, so a linear readout suffices. "Did
        # seat p win exactly its bid" is a *bump*: measured on completed rounds,
        # a linear head learns tricks_won >= k essentially perfectly and
        # tricks_won == k barely at all, because equality needs two decision
        # boundaries and one hidden layer to bracket them.
        self.bid_hit_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.max_players)
        )
        self.owner_class_proj = nn.Linear(d, config.owner_class_count * d)
        self.owner_card_emb = nn.Embedding(NUM_CARDS, d)

        self.apply(self._init_module)
        residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            block.out_proj.weight.data.mul_(residual_scale)
            block.mlp[2].weight.data.mul_(residual_scale)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.pos_embedding.weight.device

    def new_cache(
        self,
        capacity: int,
        dtype: torch.dtype = torch.float32,
        max_len: int | None = None,
    ) -> KVCache:
        return KVCache(self.config, capacity, self.device, dtype, max_len)

    def embed(self, tokens: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Sum slot embeddings + absolute positions for tokens [B, T, WIDTH]."""

        if tokens.shape[-1] != TOKEN_WIDTH:
            raise ValueError(f"Tokens must have width {TOKEN_WIDTH}.")
        x = self.slot_embedding(tokens + self.slot_offsets).sum(dim=-2)
        positions = torch.arange(
            start, start + tokens.shape[1], device=tokens.device
        )
        return x + self.pos_embedding(positions)

    def _step_heads(self, hidden: torch.Tensor) -> SeqStepOutput:
        return SeqStepOutput(
            hidden=hidden,
            bid_logits=self.bid_head(hidden),
            card_logits=self.card_head(hidden),
            value=self.value_head(hidden).squeeze(-1),
        )

    def forward_full(
        self,
        tokens: torch.Tensor,
        *,
        aux_heads: bool | Collection[str] = True,
    ) -> SeqOutput:
        """Full causal forward over [B, L, WIDTH] with per-position heads.

        ``aux_heads`` selects which auxiliary heads to run: True for all of
        AUX_HEADS, False for none, or any subset by name. Heads left out are
        None on the returned SeqOutput rather than zeros, so a loss that reads
        one it did not ask for fails loudly instead of training on nothing.
        """

        if aux_heads is True:
            wanted: Collection[str] = AUX_HEADS
        elif aux_heads is False:
            wanted = ()
        else:
            wanted = frozenset(aux_heads)
            unknown = sorted(set(wanted) - set(AUX_HEADS))
            if unknown:
                raise ValueError(f"Unknown auxiliary heads: {unknown}")

        x = self.embed(tokens)
        for block in self.blocks:
            x = block.forward_full(x)
        hidden = self.final_norm(x)

        config = self.config
        output = SeqOutput(
            hidden=hidden,
            bid_logits=self.bid_head(hidden),
            card_logits=self.card_head(hidden),
            value=self.value_head(hidden).squeeze(-1),
        )
        batch, length, _ = hidden.shape
        if "trick" in wanted:
            output.trick_logits = self.trick_count_head(hidden).view(
                batch, length, config.max_players, config.bid_count
            )
        if "suit" in wanted:
            output.suit_logits = self.suit_presence_head(hidden).view(
                batch, length, config.max_players, 4
            )
        if "bid_hit" in wanted:
            output.bid_hit_logits = self.bid_hit_head(hidden)
        if "owner" in wanted:
            class_states = self.owner_class_proj(hidden).view(
                batch, length, config.owner_class_count, config.d_model
            )
            output.owner_logits = torch.einsum(
                "blkd,cd->blck", class_states, self.owner_card_emb.weight
            ) / math.sqrt(config.d_model)
        return output

    def forward_prefill(
        self,
        tokens: torch.Tensor,
        cache: KVCache,
        slots: torch.Tensor | None,
    ) -> SeqStepOutput:
        """Encode a fresh prefix [B, T, WIDTH] into empty cache slots."""

        x = self.embed(tokens)
        for layer, block in enumerate(self.blocks):
            x = block.forward_cached(x, cache, layer, slots, start=0)
        hidden = self.final_norm(x[:, -1])
        return self._step_heads(hidden)

    def forward_prefix(self, tokens: torch.Tensor) -> SeqStepOutput:
        """Cache-free decode: re-encode [B, T, WIDTH] and read the last step.

        Heads run only on the final position so this is a like-for-like
        comparison against ``forward_step``: the difference measured is
        exactly the trunk recompute the KV cache avoids.
        """

        x = self.embed(tokens)
        for block in self.blocks:
            x = block.forward_full(x)
        return self._step_heads(self.final_norm(x[:, -1]))

    def forward_step(
        self,
        tokens: torch.Tensor,
        position: int,
        cache: KVCache,
        slots: torch.Tensor | None,
    ) -> SeqStepOutput:
        """Append tokens at ``position`` and read the heads at the last one.

        ``tokens`` is [B, WIDTH] for a single append, or [B, T, WIDTH] to
        append a run of events in one call. Only the last position produces a
        readout, which is why a run is worth merging: the events before it in
        the run exist purely to advance the cache.
        """

        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        x = self.embed(tokens, start=position)
        for layer, block in enumerate(self.blocks):
            x = block.forward_cached(x, cache, layer, slots, start=position)
        hidden = self.final_norm(x[:, -1])
        return self._step_heads(hidden)
