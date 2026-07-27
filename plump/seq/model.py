"""Schema-v6 causal sequence model with per-position heads and KV caching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import NUM_CARDS, TOKEN_WIDTH, SeqModelConfig
from .kv import KVCache


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
        """Attention for a single new query position, KV heads left grouped.

        Written out rather than delegated to SDPA because on MPS the fused
        kernel is markedly slower at query length 1, and its GQA path slower
        still -- which would make grouped KV a memory saving paid for in wall
        time. Grouping the *query* heads instead keeps K/V at ``kv_heads``,
        so GQA wins on both axes. Softmax runs in fp32 (as SDPA does) so an
        fp16 cache does not cost accuracy.
        """

        batch = q.shape[0]
        grouped = q.reshape(batch, self.kv_heads, self.head_group, self.head_dim)
        scores = (grouped @ k.transpose(-1, -2)) * self.scale
        weights = scores.float().softmax(dim=-1).to(v.dtype)
        return (weights @ v).view(batch, self.n_heads, 1, self.head_dim)

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
        if length == 1:
            k_all, v_all = cache.read(
                layer, slots, start + 1, count=x.shape[0]
            )
            attn = self._decode_attention(q.to(cache.dtype), k_all, v_all).to(
                x.dtype
            )
        else:
            if start != 0:
                raise NotImplementedError("Multi-token append requires an empty prefix.")
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self._gqa
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
        self.suit_presence_head = nn.Linear(d, config.max_players * 4)
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

    def forward_full(self, tokens: torch.Tensor, *, aux_heads: bool = True) -> SeqOutput:
        """Full causal forward over [B, L, WIDTH] with per-position heads."""

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
        if aux_heads:
            batch, length, _ = hidden.shape
            output.trick_logits = self.trick_count_head(hidden).view(
                batch, length, config.max_players, config.bid_count
            )
            output.suit_logits = self.suit_presence_head(hidden).view(
                batch, length, config.max_players, 4
            )
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
        """Append one token [B, WIDTH] at ``position`` and read the heads."""

        x = self.embed(tokens.unsqueeze(1), start=position)
        for layer, block in enumerate(self.blocks):
            x = block.forward_cached(x, cache, layer, slots, start=position)
        hidden = self.final_norm(x[:, 0])
        return self._step_heads(hidden)
