"""Schema-v6 causal sequence model with per-position heads and KV caching."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import (
    BASE_TOKEN_WIDTH,
    CORE_TOKEN_WIDTH,
    NA_PATTERN_MASKS,
    NUM_CARDS,
    NUM_RANKS,
    NUM_SUITS,
    SLOT_CARD,
    SLOT_RANK,
    SLOT_REMAINING_HAND_START,
    SLOT_NA_PATTERN,
    SLOT_SUIT,
    SLOT_TYPE,
    TOKEN_WIDTH,
    TOKEN_TRICK_WIN,
    SeqModelConfig,
)
from .kv import KVCache
from .metal_trunk import (
    MAX_LENGTH as METAL_TRUNK_MAX_LENGTH,
    MetalTrunk,
    run_trunk as run_metal_trunk,
    trunk_parameters as metal_trunk_parameters,
)

# Auxiliary heads, selectable one by one in forward_full: a caller whose loss
# weights one at zero should not pay for its forward or its backward.
AUX_HEADS = ("trick", "suit", "bid_hit", "rank_boundary", "next_winner")
SEQ_MODEL_FORMAT_VERSION = 6
STRUCTURED_CARD_OUTPUT_KEYS = frozenset(
    {
        "card_rank_output_embedding.weight",
        "card_suit_output_embedding.weight",
    }
)
EMBEDDING_NORM_KEYS = frozenset({"embedding_norm.weight"})
RANK_BOUNDARY_HEAD_KEYS = frozenset(
    {
        "rank_boundary_head.weight",
        "rank_boundary_head.bias",
    }
)
PLAYER_VALUE_HEAD_KEYS = frozenset(
    {
        "player_value_head.0.weight",
        "player_value_head.0.bias",
        "player_value_head.2.weight",
        "player_value_head.2.bias",
    }
)
NEXT_WINNER_HEAD_KEYS = frozenset(
    {
        "next_winner_head.weight",
        "next_winner_head.bias",
    }
)
ORACLE_CARD_ORDER_HEAD_KEYS = frozenset(
    {
        "card_order_head.weight",
        "card_order_head.bias",
    }
)
ADDITIVE_CONDITIONING_KEYS = frozenset(
    {
        "bid_pressure_embedding.weight",
        "na_pattern_embedding.weight",
        "per_layer_slot_embedding.weight",
        "per_layer_pos_embedding.weight",
        "per_layer_context_projection.weight",
        "per_layer_na_pattern_embedding.weight",
        "value_slot_embedding.weight",
        "value_pos_embedding.weight",
        "value_na_pattern_embedding.weight",
    }
)
NA_PATTERN_EMBEDDING_KEYS = frozenset(
    {
        "na_pattern_embedding.weight",
        "per_layer_na_pattern_embedding.weight",
        "value_na_pattern_embedding.weight",
    }
)


def _absolute_positions(
    start: int | torch.Tensor, length: int, device: torch.device
) -> torch.Tensor:
    """Token positions for a forward, shared or per-row.

    An ``int`` start gives [T] and broadcasts across the batch, which is the
    lockstep case every homogeneous wave uses. A [B] tensor gives [B, T], so a
    merged wave can hold rows sitting at different lengths in one call, and a
    [B, T] tensor passes through for a caller that has already clamped the
    positions its padding would otherwise send off the end of a table.
    """

    if isinstance(start, torch.Tensor):
        if start.dim() == 2:
            if start.shape[1] != length:
                raise ValueError("Explicit positions must cover every step.")
            return start
        return start[:, None] + torch.arange(length, device=device)[None, :]
    return torch.arange(start, start + length, device=device)


class EmbeddingRMSNorm(nn.Module):
    """RMSNorm with a gain only, applied to the summed input embedding."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).to(x.dtype) * self.weight


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
    suit_logits: Optional[torch.Tensor] = None   # [B, L, belief_opponents, 4]
    bid_hit_logits: Optional[torch.Tensor] = None  # [B, L, max_players]
    next_winner_logits: Optional[torch.Tensor] = None  # [B, L, max_players]
    # Opponent rows followed by an any-opponent row; suit; lower/higher.
    rank_boundary_logits: Optional[torch.Tensor] = None  # [B,L,O+1,4,2]


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

    def _qkv(
        self,
        h: torch.Tensor,
        value_embedding: torch.Tensor | None = None,
        value_gate_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        v = v.view(batch, length, self.kv_heads, self.head_dim)
        if value_embedding is not None:
            embedded = value_embedding.view_as(v)
            if value_gate_weight is not None:
                gate_width = value_gate_weight.shape[-1]
                gate = 2.0 * torch.sigmoid(
                    F.linear(h[..., :gate_width], value_gate_weight)
                )
                embedded = gate.unsqueeze(-1) * embedded
            v = v + embedded
        v = v.transpose(1, 2)
        return q, k, v

    @property
    def _gqa(self) -> bool:
        # SDPA broadcasts the KV heads internally; materialising the expanded
        # K/V would undo the memory saving GQA exists for.
        return self.kv_heads != self.n_heads

    def _decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention for ``T`` new query positions, KV heads left grouped.

        Written out rather than delegated to SDPA because on MPS the fused
        kernel is markedly slower at short query lengths, and its GQA path
        slower still -- which would make grouped KV a memory saving paid for in
        wall time. Grouping the *query* heads instead keeps K/V at
        ``kv_heads``, so GQA wins on both axes. Softmax runs in fp32 (as SDPA
        does), limiting sensitivity to reduced-precision cache storage.

        With ``T > 1`` the new queries sit at the end of the cached prefix, so
        query ``i`` may see keys up to ``len(k) - T + i``: a rectangular causal
        mask over the tail, not a square one. Ragged batches place that ramp
        per row instead, starting from each row's own length.

        ``key_lengths`` makes the batch ragged: row ``b`` attends only to its
        own first ``key_lengths[b]`` keys, so rows holding different games at
        different lengths can share one call. K/V are read to the batch maximum
        and the surplus is masked off rather than gathered per row -- measured
        at 1.14x going from length 30 to 73 and free from 73 to 146, which is
        far cheaper than splitting the batch by length.
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
        total = k.shape[2]
        keys = torch.arange(total, device=q.device)
        if key_lengths is not None:
            # The ragged mask already encodes causality: row b's query t sits
            # at key_lengths[b] - 1 + t, so anything at or past that plus one
            # is in its future. That subsumes the rectangular mask below, which
            # only holds when every row ends at the same key.
            if length == 1:
                blocked = keys[None, :] >= key_lengths[:, None]
                scores = scores.masked_fill(
                    blocked[:, None, None, :], float("-inf")
                )
            else:
                limits = (
                    key_lengths[:, None]
                    + torch.arange(length, device=q.device)[None, :]
                )
                blocked = keys[None, None, :] >= limits[:, :, None]
                # Tiling [B, T, K] by head_group along the time axis puts query
                # t of group g at row g * length + t, matching ``grouped``.
                scores = scores.masked_fill(
                    blocked.repeat(1, self.head_group, 1)[:, None], float("-inf")
                )
        elif length > 1:
            offsets = torch.arange(length, device=q.device) + (total - length)
            blocked = keys > offsets[:, None]
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

    def forward_full(
        self,
        x: torch.Tensor,
        *,
        value_embedding: torch.Tensor | None = None,
        value_gate_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self._qkv(
            self.ln_attn(x), value_embedding, value_gate_weight
        )
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
        *,
        value_embedding: torch.Tensor | None = None,
        value_gate_weight: torch.Tensor | None = None,
        row_positions: torch.Tensor | None = None,
        key_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal attention against the cached prefix; writes new K/V.

        ``slots=None`` uses the dense rows 0..B-1 fast path (zero-copy reads).
        Attention runs in the cache dtype. Matching it to the autocast dtype
        avoids a query cast while retaining two-byte cache storage.

        ``row_positions``/``key_lengths`` switch to the merged path: each row
        writes at its own offset and attends to its own prefix, so one call can
        carry games of different shapes and lengths. ``start`` is then the
        batch maximum, used only to size the K/V read.
        """

        length = x.shape[1]
        q, k, v = self._qkv(
            self.ln_attn(x), value_embedding, value_gate_weight
        )
        if row_positions is not None:
            rows = (
                torch.arange(x.shape[0], device=x.device)
                if slots is None
                else slots
            )
            write_positions = row_positions
            if row_positions.dim() == 1 and length > 1:
                write_positions = (
                    row_positions[:, None]
                    + torch.arange(length, device=x.device)[None, :]
                )
            cache.write_step(layer, rows, write_positions, k, v)
        else:
            cache.write_range(layer, slots, start, k, v)
        if row_positions is None and start == 0 and length > 1:
            # Prefill: no prefix to attend to, so the square causal kernel is
            # exactly right and SDPA is worth using at this length.
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self._gqa
            )
        else:
            k_all, v_all = cache.read(
                layer, slots, start + length, count=x.shape[0]
            )
            attn = self._decode_attention(
                q.to(cache.dtype), k_all, v_all, key_lengths=key_lengths
            ).to(x.dtype)
        return self._finish(x, attn)


class FactoredCardHead(nn.Module):
    """Card scoring with no per-card row of any kind.

    ``logit(c) = h . (rank[r_c] + suit[s_c]) + rank_bias[r_c] + suit_bias[s_c]``

    The directions themselves live on the backbone as
    ``card_rank_output_embedding`` / ``card_suit_output_embedding`` (shared with
    the dense path, where they are an additive correction); this module holds
    only the factored bias so that ``card_head.bias`` keeps working for callers.
    ``weight`` is ``None``, which is how the backbone detects the factored path.
    """

    weight: None = None

    def __init__(self) -> None:
        super().__init__()
        self.rank_bias = nn.Parameter(torch.zeros(NUM_RANKS))
        self.suit_bias = nn.Parameter(torch.zeros(NUM_SUITS))
        self.register_buffer(
            "_rank_ids",
            torch.arange(NUM_CARDS, dtype=torch.long) % NUM_RANKS,
            persistent=False,
        )
        self.register_buffer(
            "_suit_ids",
            torch.arange(NUM_CARDS, dtype=torch.long) // NUM_RANKS,
            persistent=False,
        )

    @property
    def bias(self) -> torch.Tensor:
        return self.rank_bias[self._rank_ids] + self.suit_bias[self._suit_ids]


class SeqPlumpModel(nn.Module):
    def __init__(self, config: SeqModelConfig):
        super().__init__()
        self._card_input_weight_cache: torch.Tensor | None = None
        self._card_output_weight_cache: torch.Tensor | None = None
        # Optional fused Metal trunk (plump.seq.metal_trunk), switched on by
        # the trainer. A plain attribute: it holds no state of its own.
        self.metal_trunk = None
        config.validate()
        self.config = config
        d = config.d_model
        # All twelve token slots share one embedding table with per-slot id
        # offsets. Twelve separate lookups plus eleven adds would be ~23
        # kernels per forward, and the rollout runs one forward per game
        # event, so that overhead dominates at small batch sizes.
        sizes = config.embedding_core_slot_vocab_sizes
        offsets = torch.tensor(
            [sum(sizes[:index]) for index in range(len(sizes))], dtype=torch.long
        )
        self.register_buffer("slot_offsets", offsets, persistent=False)
        self.slot_embedding = nn.Embedding(int(sum(sizes)), d)
        self.pos_embedding: nn.Embedding | None = (
            nn.Embedding(config.max_seq_len, d)
            if config.absolute_position_embedding
            else None
        )
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(d)

        self.bid_head = nn.Linear(d, config.bid_count)
        # Factored mode removes the 52 x d matrix entirely; the rank and suit
        # output embeddings below become the whole head rather than a
        # correction on top of it.
        self.card_head: nn.Module = (
            FactoredCardHead()
            if config.card_embedding_factored
            else nn.Linear(d, NUM_CARDS)
        )
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        # Every seat including the observer's own (relative seat 0): how many
        # tricks the observer itself ends up taking is a genuine prediction
        # about how the rest of the round plays out, not a readback of its hand.
        self.trick_count_head = nn.Linear(d, config.max_players * config.bid_count)
        # The belief heads emit one logit per (relative seat, class) and let the
        # loss mask seats a shape does not have. The alternative -- 4 (resp. 1)
        # logits plus a seat embedding, evaluated once per seat -- would turn a
        # single d x 16 matmul into P passes over [B, L, d] for at most five
        # seats that are already distinguishable by their relative index.
        # Opponents only: see SeqModelConfig.belief_opponents.
        self.suit_presence_head = nn.Linear(d, config.belief_opponents * 4)
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
        # Output-side transfer terms. Card c is scored with one effective row:
        #
        #   W_exact[c] + W_rank[rank(c)] + W_suit[suit(c)]
        #
        # The exact-card residual retains full expressivity, while rank rows
        # share value knowledge across all four suits and suit rows share
        # suit-level behavior across ranks. Register these after the existing
        # heads so a pre-format-2 optimizer's original parameter order remains
        # a prefix during checkpoint migration.
        # Module constructors normally consume the global Torch RNG before
        # ``self.apply`` initializes the complete model. Preserve that RNG
        # point so adding these parameters does not silently change every
        # pre-existing cold-start weight (and therefore the rollout stream).
        with torch.random.fork_rng(devices=[]):
            self.card_rank_output_embedding = nn.Embedding(NUM_RANKS, d)
            self.card_suit_output_embedding = nn.Embedding(NUM_SUITS, d)
            # For each opponent (plus one aggregate row) and suit, predict
            # whether a held card lies below the observer's current minimum or
            # above its current maximum. It is deliberately registered last:
            # format-2 parameter/optimizer ordering remains a strict prefix,
            # and old checkpoints can leave this readout randomly initialized.
            self.rank_boundary_head = nn.Linear(
                d, (config.belief_opponents + 1) * NUM_SUITS * 2
            )
            # One logit per seat on whichever seat axis the stream uses --
            # observer-relative for an actor, absolute for the oracle, since
            # the oracle's canonical observer is seat 0. Linear like suit
            # presence rather than an MLP like bid hit: "who takes this trick"
            # is monotone in the accumulated trick state, not an equality bump.
            self.next_winner_head = nn.Linear(d, config.max_players)
        self.register_buffer(
            "_card_output_rank_ids",
            torch.arange(NUM_CARDS, dtype=torch.long) % NUM_RANKS,
            persistent=False,
        )
        self.register_buffer(
            "_card_output_suit_ids",
            torch.arange(NUM_CARDS, dtype=torch.long) // NUM_RANKS,
            persistent=False,
        )
        card_ids = torch.arange(NUM_CARDS, dtype=torch.long)
        card_input_rows = [
            self._card_output_rank_ids + offsets[SLOT_RANK],
            self._card_output_suit_ids + offsets[SLOT_SUIT],
        ]
        if not config.card_embedding_factored:
            card_input_rows.insert(0, card_ids + offsets[SLOT_CARD])
        self.register_buffer(
            "_card_input_slot_ids",
            torch.stack(tuple(card_input_rows), dim=-1),
            persistent=False,
        )
        # Zero for any base column the model does not embed. Multiplying the
        # token column by this keeps the id inside its (collapsed) block and
        # the matching bag weight is forced off, so the column contributes
        # exactly nothing without perturbing offsets for the columns after it.
        active = torch.ones(BASE_TOKEN_WIDTH, dtype=torch.long)
        if config.card_embedding_factored:
            active[SLOT_CARD] = 0
        self.register_buffer("base_slot_active", active, persistent=False)

        # Additive schema/capacity extensions are registered after every
        # pre-existing actor parameter.  Old Adam parameter ids therefore
        # remain a strict prefix when a checkpoint first enables them.
        pressure_sizes = config.bid_pressure_slot_vocab_sizes
        self.register_buffer(
            "bid_pressure_offsets",
            torch.tensor(
                [sum(pressure_sizes[:i]) for i in range(len(pressure_sizes))],
                dtype=torch.long,
            ),
            persistent=False,
        )
        na_ids = torch.tensor(
            [value if value is not None else -1 for value in config.base_slot_na_ids],
            dtype=torch.long,
        )
        self.register_buffer("base_slot_na_ids", na_ids, persistent=False)
        with torch.random.fork_rng(devices=[]):
            self.bid_pressure_embedding = nn.Embedding(
                int(sum(pressure_sizes)), d
            )

            self.per_layer_slot_embedding: nn.Embedding | None = None
            self.per_layer_pos_embedding: nn.Embedding | None = None
            self.per_layer_context_projection: nn.Linear | None = None
            ple_dim = config.per_layer_embedding_dim
            if ple_dim:
                packed_ple = config.n_layers * ple_dim
                self.per_layer_slot_embedding = nn.Embedding(
                    int(sum(config.embedding_base_slot_vocab_sizes)), packed_ple
                )
                if config.absolute_position_embedding:
                    self.per_layer_pos_embedding = nn.Embedding(
                        config.max_seq_len, packed_ple
                    )
                self.per_layer_context_projection = nn.Linear(
                    d, packed_ple, bias=False
                )

            self.value_slot_embedding: nn.Embedding | None = None
            self.value_pos_embedding: nn.Embedding | None = None
            value_layers = config.value_embedding_layers
            if value_layers:
                packed_value = len(value_layers) * config.kv_heads * config.head_dim
                self.value_slot_embedding = nn.Embedding(
                    int(sum(config.embedding_base_slot_vocab_sizes)), packed_value
                )
                if config.absolute_position_embedding:
                    self.value_pos_embedding = nn.Embedding(
                        config.max_seq_len, packed_value
                    )

            # One row represents the complete set of absent fields on a token.
            # Register these after every legacy module so old optimizer ids
            # remain a strict prefix during checkpoint migration.
            self.embedding_norm: EmbeddingRMSNorm | None = (
                EmbeddingRMSNorm(d) if config.embedding_rms_norm else None
            )
            self.na_pattern_embedding = nn.Embedding(len(NA_PATTERN_MASKS), d)
            self.per_layer_na_pattern_embedding: nn.Embedding | None = None
            if ple_dim:
                self.per_layer_na_pattern_embedding = nn.Embedding(
                    len(NA_PATTERN_MASKS), packed_ple
                )
            self.value_na_pattern_embedding: nn.Embedding | None = None
            if value_layers:
                self.value_na_pattern_embedding = nn.Embedding(
                    len(NA_PATTERN_MASKS), packed_value
                )
        conditioning_sizes = config.embedding_base_slot_vocab_sizes
        self.register_buffer(
            "conditioning_slot_offsets",
            torch.tensor(
                [sum(conditioning_sizes[:i]) for i in range(len(conditioning_sizes))],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self._value_layer_to_index = {
            layer: index
            for index, layer in enumerate(config.value_embedding_layers)
        }

        # Reproduce the pre-extension initialization/RNG stream exactly for
        # all existing parameters.  New modules initialize in a forked stream,
        # so merely enabling extra capacity cannot change rollout sampling
        # before those zero contributions have learned anything.
        for module in (
            self.slot_embedding,
            self.pos_embedding,
            self.blocks,
            self.final_norm,
            self.bid_head,
            self.card_head,
            self.value_head,
            self.trick_count_head,
            self.suit_presence_head,
            self.bid_hit_head,
            self.card_rank_output_embedding,
            self.card_suit_output_embedding,
            self.rank_boundary_head,
        ):
            # pos_embedding is absent when absolute positions are off. The
            # guard cannot change the draw order for a model that has one,
            # so existing cold starts are unaffected.
            if module is not None:
                module.apply(self._init_module)
        with torch.random.fork_rng(devices=[]):
            for module in (
                self.bid_pressure_embedding,
                self.per_layer_slot_embedding,
                self.per_layer_pos_embedding,
                self.per_layer_context_projection,
                self.value_slot_embedding,
                self.value_pos_embedding,
                # Appended last so the draws every module above it receives are
                # the ones it received before this head existed.
                self.next_winner_head,
            ):
                if module is not None:
                    module.apply(self._init_module)
        # A zero additive start preserves the former head exactly—important
        # both for cold starts and for resuming existing checkpoints. Adam gets
        # nonzero gradients for these rows immediately, so zero is a neutral
        # starting contribution rather than a frozen or symmetric dead path.
        # Under ``dense`` these rows are an additive correction on top of the
        # exact-card matrix, so a zero start reproduces the former head
        # exactly. Under ``factored`` they *are* the head: zeroing them would
        # make every card score identically with no way to break the tie on
        # the first step, so the ordinary initialization is kept.
        if not self.config.card_embedding_factored:
            nn.init.zeros_(self.card_rank_output_embedding.weight)
            nn.init.zeros_(self.card_suit_output_embedding.weight)
        nn.init.zeros_(self.bid_pressure_embedding.weight)
        if self.per_layer_slot_embedding is not None:
            assert self.per_layer_context_projection is not None
            nn.init.zeros_(self.per_layer_slot_embedding.weight)
            nn.init.zeros_(self.per_layer_context_projection.weight)
            if self.per_layer_pos_embedding is not None:
                nn.init.zeros_(self.per_layer_pos_embedding.weight)
        if self.value_slot_embedding is not None:
            nn.init.zeros_(self.value_slot_embedding.weight)
            if self.value_pos_embedding is not None:
                nn.init.zeros_(self.value_pos_embedding.weight)
        self.initialize_na_pattern_embeddings()
        residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            block.out_proj.weight.data.mul_(residual_scale)
            block.mlp[2].weight.data.mul_(residual_scale)
        # Per-seat value on the observer-relative axis: entry r is the seat r
        # steps clockwise from the observer, so entry 0 is the observer itself.
        # Deliberately the same module shape as SeqPPOOracleCritic's
        # player_value_head, trained on the same targets with the same zero-sum
        # centering, the same dense position weights and the same
        # normalization, so the two losses are directly comparable and the only
        # difference measured is observer information versus perfect
        # information. The scalar value_head above remains the counterfactual
        # objectives' baseline readout and the oracle's initialization source.
        #
        # Built last and drawn from a forked stream so that adding it leaves
        # every pre-existing parameter bit-identical: an additive head must not
        # silently reinitialize the model around it.
        with torch.random.fork_rng(devices=[]):
            self.player_value_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.max_players)
            )
            self.player_value_head.apply(self._init_module)

    def initialize_na_pattern_embeddings(self) -> None:
        """Fold legacy per-field sentinel rows into one row per NA pattern."""

        def combined(
            weight: torch.Tensor,
            offsets: torch.Tensor,
            mask: int,
            *,
            first_slot: int = 0,
        ) -> torch.Tensor:
            rows = []
            for local_slot, offset in enumerate(offsets):
                slot = first_slot + local_slot
                sentinel = self.config.base_slot_na_ids[slot]
                if sentinel is not None and mask & (1 << slot):
                    rows.append(weight[int(offset) + sentinel])
            if not rows:
                return weight.new_zeros(weight.shape[-1])
            return torch.stack(rows).sum(dim=0)

        with torch.no_grad():
            self.na_pattern_embedding.weight.zero_()
            for pattern_id, mask in enumerate(NA_PATTERN_MASKS[1:], start=1):
                core = combined(
                    self.slot_embedding.weight,
                    self.slot_offsets,
                    mask,
                )
                pressure = combined(
                    self.bid_pressure_embedding.weight,
                    self.bid_pressure_offsets,
                    mask,
                    first_slot=CORE_TOKEN_WIDTH,
                )
                self.na_pattern_embedding.weight[pattern_id].copy_(core + pressure)

            for table, patterns in (
                (
                    self.per_layer_slot_embedding,
                    self.per_layer_na_pattern_embedding,
                ),
                (self.value_slot_embedding, self.value_na_pattern_embedding),
            ):
                if table is None or patterns is None:
                    continue
                patterns.weight.zero_()
                for pattern_id, mask in enumerate(NA_PATTERN_MASKS[1:], start=1):
                    patterns.weight[pattern_id].copy_(
                        combined(
                            table.weight,
                            self.conditioning_slot_offsets,
                            mask,
                        )
                    )

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
        return self.slot_embedding.weight.device

    def clear_card_output_cache(self) -> None:
        self._card_input_weight_cache = None
        self._card_output_weight_cache = None

    def effective_card_input_weight(self) -> torch.Tensor:
        """Return the 52 exact + rank + suit input directions."""

        if (
            not self.training
            and not torch.is_grad_enabled()
            and self._card_input_weight_cache is not None
        ):
            return self._card_input_weight_cache
        weight = self._embedding_bag_sum(
            self._card_input_slot_ids,
            self.slot_embedding.weight,
        )
        if not self.training and not torch.is_grad_enabled():
            self._card_input_weight_cache = weight
        return weight

    def train(self, mode: bool = True):
        # Evaluation caches the 52 effective rows so the autoregressive hot
        # path still issues one card-logit matmul per step. Any mode transition
        # can follow an optimizer/load mutation, so invalidate conservatively.
        self.clear_card_output_cache()
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        self.clear_card_output_cache()
        return super()._apply(fn, recurse=recurse)

    def effective_card_output_weight(self) -> torch.Tensor:
        """Return the 52 exact + rank + suit scoring directions."""

        if (
            not self.training
            and not torch.is_grad_enabled()
            and self._card_output_weight_cache is not None
        ):
            return self._card_output_weight_cache
        weight = self.card_rank_output_embedding(
            self._card_output_rank_ids
        ) + self.card_suit_output_embedding(self._card_output_suit_ids)
        if self.card_head.weight is not None:
            weight = weight + self.card_head.weight
        if not self.training and not torch.is_grad_enabled():
            self._card_output_weight_cache = weight
        return weight

    @staticmethod
    def _embedding_bag_sum(
        indices: torch.Tensor,
        weight: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum fixed-width embedding bags without a wide slot temporary.

        ``F.embedding(indices, weight).sum(-2)`` materializes
        ``[..., slots, embedding_dim]``.  The packed conditioning tables are
        deliberately wide (768 for the active PLE setup), so that temporary
        costs substantially more memory traffic than the transformer matmul
        it feeds.  ``embedding_bag`` performs the identical gather-and-sum
        directly into ``[..., embedding_dim]`` and retains dense gradients for
        the ordinary Adam optimizer.

        ``valid`` supplies zero per-sample weights for padded card ids.  It is
        used only for the repeated remaining-hand fields, where there is no
        single padding row shared by the exact-card, rank, and suit tables.
        """

        shape = indices.shape[:-1]
        flat = indices.reshape(-1, indices.shape[-1])
        sample_weight = None
        if valid is not None:
            sample_weight = valid.reshape_as(flat).to(weight.dtype)
        summed = F.embedding_bag(
            input=flat,
            weight=weight,
            mode="sum",
            per_sample_weights=sample_weight,
        )
        return summed.view(*shape, weight.shape[-1])

    def _card_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # One matmul, including on the one-token rollout path. The effective
        # weight is cached while evaluating and rebuilt with gradients for each
        # training forward.
        return F.linear(
            hidden,
            self.effective_card_output_weight(),
            self.card_head.bias,
        )

    def new_cache(
        self,
        capacity: int,
        dtype: torch.dtype = torch.float32,
        max_len: int | None = None,
        stacked: bool | None = None,
    ) -> KVCache:
        return KVCache(
            self.config, capacity, self.device, dtype, max_len, stacked=stacked
        )

    def embed(
        self, tokens: torch.Tensor, start: int | torch.Tensor = 0
    ) -> torch.Tensor:
        """Embed base slots, cards, and positions, then normalize if enabled."""

        x = self._embed_sum(tokens, start)
        if self.embedding_norm is not None:
            x = self.embedding_norm(x)
        return x

    @torch.no_grad()
    def calibrate_embedding_norm(self, batches) -> float:
        """Set the gain so the norm passes the embedding sum through unchanged.

        RMSNorm divides by each token's own RMS, so the single gain that best
        reproduces the pre-norm function is the mean per-token RMS over real
        traffic. Without this a model that trained without the norm sees its
        input rescaled by ~1/rms and its policy collapses.

        `batches` yields (tokens, valid) pairs. They are pooled rather than
        concatenated because rollout groups have different sequence lengths.
        """

        if self.embedding_norm is None:
            raise ValueError("embedding_rms_norm is disabled for this model.")
        total = 0.0
        count = 0
        for tokens, valid in batches:
            rms = self._embed_sum(tokens).float().pow(2).mean(-1).sqrt()
            if valid is not None:
                rms = rms[valid.to(rms.device).bool()]
            total += float(rms.sum())
            count += int(rms.numel())
        if count == 0:
            raise ValueError("Calibration needs at least one real token.")
        gain = total / count
        self.embedding_norm.weight.fill_(gain)
        return gain

    def _embed_sum(
        self, tokens: torch.Tensor, start: int | torch.Tensor = 0
    ) -> torch.Tensor:
        """Embed base slots, remaining-hand cards, and absolute positions."""

        if tokens.shape[-1] != TOKEN_WIDTH:
            raise ValueError(f"Tokens must have width {TOKEN_WIDTH}.")
        core = tokens[..., :CORE_TOKEN_WIDTH]
        pressure = tokens[..., CORE_TOKEN_WIDTH:BASE_TOKEN_WIDTH]
        core_na = self.base_slot_na_ids[:CORE_TOKEN_WIDTH]
        pressure_na = self.base_slot_na_ids[CORE_TOKEN_WIDTH:BASE_TOKEN_WIDTH]
        core_active = self.base_slot_active[:CORE_TOKEN_WIDTH]
        x = self._embedding_bag_sum(
            core * core_active + self.slot_offsets,
            self.slot_embedding.weight,
            valid=((core_na < 0) | (core != core_na)) & (core_active > 0),
        )
        x = x + self._embedding_bag_sum(
            pressure + self.bid_pressure_offsets,
            self.bid_pressure_embedding.weight,
            valid=(pressure_na < 0) | (pressure != pressure_na),
        )
        x = x + self.na_pattern_embedding(tokens[..., SLOT_NA_PATTERN])

        # Only TRICK_WIN rows contain remaining-card ids. Select those rows
        # before gathering so a normal PLAY/BID/TURN step pays no dense
        # [rows, 10, d_model] temporary. Each card reuses exactly the same
        # exact-card + rank + suit input direction as its HAND/PLAY token.
        # Resolve the mask to explicit indices once and reuse them for both
        # the gather and the scatter. Indexing twice with the same boolean
        # mask has been observed on MPS to return different row counts within
        # one forward -- a 1200-iteration run died with "value tensor of shape
        # [138, 240] cannot be broadcast to indexing result of shape [0, 240]",
        # which is that inconsistency exactly. A materialized index tensor
        # cannot disagree with itself.
        trick_win = (core[..., SLOT_TYPE] == TOKEN_TRICK_WIN).nonzero(as_tuple=True)
        remaining = tokens[..., SLOT_REMAINING_HAND_START:SLOT_NA_PATTERN][trick_win]
        if remaining.numel():
            valid = remaining < NUM_CARDS
            safe_ids = remaining.clamp_max(NUM_CARDS - 1)
            hand_sum = self._embedding_bag_sum(
                safe_ids,
                self.effective_card_input_weight(),
                valid=valid,
            )
            x = x.clone()
            x[trick_win] = x[trick_win] + hand_sum
        if self.pos_embedding is None:
            return x
        positions = _absolute_positions(start, tokens.shape[1], tokens.device)
        return x + self.pos_embedding(positions)

    def _packed_token_conditioning(
        self,
        tokens: torch.Tensor,
        table: nn.Embedding,
        na_patterns: nn.Embedding,
    ) -> torch.Tensor:
        """Sum all categorical fields, including repeated remaining cards."""

        base = tokens[..., :BASE_TOKEN_WIDTH]
        packed = self._embedding_bag_sum(
            base * self.base_slot_active + self.conditioning_slot_offsets,
            table.weight,
            valid=(
                (self.base_slot_na_ids < 0) | (base != self.base_slot_na_ids)
            )
            & (self.base_slot_active > 0),
        )
        packed = packed + na_patterns(tokens[..., SLOT_NA_PATTERN])
        # Same single-materialization rule as embed(); see the note there.
        trick_win = (base[..., SLOT_TYPE] == TOKEN_TRICK_WIN).nonzero(as_tuple=True)
        remaining = tokens[..., SLOT_REMAINING_HAND_START:SLOT_NA_PATTERN][trick_win]
        if remaining.numel():
            valid = remaining < NUM_CARDS
            safe = remaining.clamp_max(NUM_CARDS - 1)
            remaining_rows = [
                safe.remainder(NUM_RANKS)
                + self.conditioning_slot_offsets[SLOT_RANK],
                safe.div(NUM_RANKS, rounding_mode="floor")
                + self.conditioning_slot_offsets[SLOT_SUIT],
            ]
            if not self.config.card_embedding_factored:
                remaining_rows.insert(
                    0, safe + self.conditioning_slot_offsets[SLOT_CARD]
                )
            card_rows = torch.stack(tuple(remaining_rows), dim=-1)
            card_valid = valid.unsqueeze(-1).expand_as(card_rows)
            hand_sum = self._embedding_bag_sum(
                card_rows.flatten(-2),
                table.weight,
                valid=card_valid.flatten(-2),
            )
            packed = packed.clone()
            packed[trick_win] = packed[trick_win] + hand_sum
        return packed

    def _prepare_layer_conditioning(
        self,
        tokens: torch.Tensor,
        initial_embedding: torch.Tensor,
        start: int | torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare packed PLE and attention-value streams once per forward."""

        positions = _absolute_positions(start, tokens.shape[1], tokens.device)
        ple = None
        if self.per_layer_slot_embedding is not None:
            assert self.per_layer_context_projection is not None
            assert self.per_layer_na_pattern_embedding is not None
            dim = self.config.per_layer_embedding_dim
            identity = self._packed_token_conditioning(
                tokens,
                self.per_layer_slot_embedding,
                self.per_layer_na_pattern_embedding,
            )
            if self.per_layer_pos_embedding is not None:
                identity = identity + self.per_layer_pos_embedding(positions)
            context = self.per_layer_context_projection(initial_embedding)
            ple = ((identity + context) * (1.0 / math.sqrt(2.0))).view(
                *tokens.shape[:2], self.config.n_layers, dim
            )

        values = None
        if self.value_slot_embedding is not None:
            assert self.value_na_pattern_embedding is not None
            kv_dim = self.config.kv_heads * self.config.head_dim
            values = (
                self._packed_token_conditioning(
                    tokens,
                    self.value_slot_embedding,
                    self.value_na_pattern_embedding,
                )
                + (
                    self.value_pos_embedding(positions)
                    if self.value_pos_embedding is not None
                    else 0
                )
            ).view(*tokens.shape[:2], len(self._value_layer_to_index), kv_dim)
        return ple, values

    def _apply_per_layer_embedding(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor | None,
        layer: int,
    ) -> torch.Tensor:
        if conditioning is None:
            return x
        update = conditioning[..., layer, :]
        if update.shape[-1] == x.shape[-1]:
            return x + update
        if x.shape[-1] % update.shape[-1] == 0:
            repeats = x.shape[-1] // update.shape[-1]
            # Broadcast in [repeat, PLE] form. Unlike materializing a repeated
            # [layers, d_model] conditioning tensor up front, this keeps dense
            # update memory proportional to the configured PLE width.
            shaped = x.view(*x.shape[:-1], repeats, update.shape[-1])
            shaped = shaped + update.unsqueeze(-2) * (1.0 / math.sqrt(repeats))
            return shaped.reshape_as(x)
        repeats = math.ceil(x.shape[-1] / update.shape[-1])
        expanded = torch.cat([update] * repeats, dim=-1)[..., : x.shape[-1]]
        return x + expanded * (1.0 / math.sqrt(repeats))

    def _run_blocks_full(
        self,
        tokens: torch.Tensor,
        x: torch.Tensor,
        *,
        start: int = 0,
    ) -> torch.Tensor:
        ple, values = self._prepare_layer_conditioning(tokens, x, start)
        for layer, block in enumerate(self.blocks):
            value_index = self._value_layer_to_index.get(layer)
            value = None if value_index is None else values[..., value_index, :]
            x = block.forward_full(
                x, value_embedding=value
            )
            x = self._apply_per_layer_embedding(x, ple, layer)
        return x

    def _run_blocks_cached(
        self,
        tokens: torch.Tensor,
        x: torch.Tensor,
        cache: KVCache,
        slots: torch.Tensor | None,
        *,
        start: int,
        row_positions: torch.Tensor | None = None,
        key_lengths: torch.Tensor | None = None,
        embed_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``embed_positions`` splits the two uses of a position apart: the cache
        # is written at the true offset, while the embedding tables are read at
        # a clamped one. They differ only for a merged wave's padding, whose
        # output is discarded but whose index still has to be in range.
        conditioning_start = start if row_positions is None else row_positions
        if embed_positions is not None:
            conditioning_start = embed_positions
        ple, values = self._prepare_layer_conditioning(
            tokens, x, conditioning_start
        )
        for layer, block in enumerate(self.blocks):
            value_index = self._value_layer_to_index.get(layer)
            value = None if value_index is None else values[..., value_index, :]
            x = block.forward_cached(
                x,
                cache,
                layer,
                slots,
                start,
                value_embedding=value,
                row_positions=row_positions,
                key_lengths=key_lengths,
            )
            x = self._apply_per_layer_embedding(x, ple, layer)
        return x

    def _step_heads(
        self,
        hidden: torch.Tensor,
        *,
        phase: str | None = None,
        with_value: bool = True,
    ) -> SeqStepOutput:
        """Read the heads needed by one decode step.

        Training and general inference keep the default and receive every
        head.  Rollout waves know whether the next action is a bid or a play,
        so they can skip the unused (and relatively wide) action projection.
        Empty logits make an accidental read of the skipped head fail through
        its shape instead of silently supplying plausible values.
        """

        if phase not in (None, "bid", "play"):
            raise ValueError(f"Unknown policy phase {phase!r}.")
        empty = hidden.new_empty((hidden.shape[0], 0))
        return SeqStepOutput(
            hidden=hidden,
            bid_logits=self.bid_head(hidden) if phase != "play" else empty,
            card_logits=self._card_logits(hidden) if phase != "bid" else empty,
            value=(
                self.value_head(hidden).squeeze(-1)
                if with_value
                else hidden.new_empty((hidden.shape[0], 0))
            ),
        )

    def action_logits(self, hidden: torch.Tensor, phase: str) -> torch.Tensor:
        """Policy logits for hidden states already produced by the trunk.

        A merged wave carries bid rows and play rows in one trunk call, so the
        action head is chosen per row group afterwards rather than per call.
        """

        if phase == "bid":
            return self.bid_head(hidden)
        if phase == "play":
            return self._card_logits(hidden)
        raise ValueError(f"Unknown policy phase {phase!r}.")

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full causal trunk without materializing any readout head."""

        x = self.embed(tokens)
        if self.uses_metal_trunk(tokens):
            return run_metal_trunk(self, self.metal_trunk, tokens, x)
        x = self._run_blocks_full(tokens, x)
        return self.final_norm(x)

    def enable_metal_trunk(self) -> None:
        """Route bf16 MPS trunk forwards (and their backwards) through Metal."""

        self.metal_trunk = MetalTrunk(self.config, eps=self.final_norm.eps)

    def uses_metal_trunk(self, tokens: torch.Tensor) -> bool:
        """The fused kernels hard-code bf16 GEMMs, so they only stand in for
        the trunk under bf16 autocast on MPS; anything else (fp32 training,
        CPU evaluation) keeps the PyTorch path."""

        return (
            self.metal_trunk is not None
            and tokens.device.type == "mps"
            and tokens.shape[1] <= METAL_TRUNK_MAX_LENGTH
            and torch.is_autocast_enabled("mps")
            and torch.get_autocast_dtype("mps") == torch.bfloat16
        )

    def policy_logits(self, hidden: torch.Tensor, phase: str) -> torch.Tensor:
        """Evaluate only the requested action head on selected hidden rows."""

        if phase == "bid":
            return self.bid_head(hidden)
        if phase == "play":
            return self._card_logits(hidden)
        raise ValueError(f"Unknown policy phase {phase!r}.")

    def auxiliary_logits(
        self,
        hidden: torch.Tensor,
        heads: Collection[str],
    ) -> dict[str, torch.Tensor]:
        """Evaluate selected linear auxiliaries in one packed projection.

        PPO passes only actual decision-state rows here.  Concatenating the
        three small readouts avoids issuing one accelerator matmul per head;
        autograd scatters the packed weight gradient back to the original
        checkpoint-compatible parameters.
        """

        modules = {
            "suit": self.suit_presence_head,
            "rank_boundary": self.rank_boundary_head,
            "trick": self.trick_count_head,
            "next_winner": self.next_winner_head,
        }
        ordered = tuple(
            name
            for name in ("suit", "rank_boundary", "trick", "next_winner")
            if name in heads
        )
        unknown = sorted(set(heads) - set(modules))
        if unknown:
            raise ValueError(f"Unknown packed auxiliary heads: {unknown}")
        if not ordered:
            return {}
        widths = [modules[name].out_features for name in ordered]
        packed = F.linear(
            hidden,
            torch.cat([modules[name].weight for name in ordered], dim=0),
            torch.cat([modules[name].bias for name in ordered], dim=0),
        )
        return dict(zip(ordered, packed.split(widths, dim=-1)))

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

        hidden = self.forward_hidden(tokens)

        config = self.config
        output = SeqOutput(
            hidden=hidden,
            bid_logits=self.bid_head(hidden),
            card_logits=self._card_logits(hidden),
            value=self.value_head(hidden).squeeze(-1),
        )
        batch, length, _ = hidden.shape
        if "trick" in wanted:
            output.trick_logits = self.trick_count_head(hidden).view(
                batch, length, config.max_players, config.bid_count
            )
        if "suit" in wanted:
            output.suit_logits = self.suit_presence_head(hidden).view(
                batch, length, config.belief_opponents, 4
            )
        if "bid_hit" in wanted:
            output.bid_hit_logits = self.bid_hit_head(hidden)
        if "next_winner" in wanted:
            output.next_winner_logits = self.next_winner_head(hidden)
        if "rank_boundary" in wanted:
            output.rank_boundary_logits = self.rank_boundary_head(hidden).view(
                batch,
                length,
                config.belief_opponents + 1,
                NUM_SUITS,
                2,
            )
        return output

    def forward_prefill(
        self,
        tokens: torch.Tensor,
        cache: KVCache,
        slots: torch.Tensor | None,
        *,
        readout_indices: torch.Tensor | None = None,
        phase: str | None = None,
        with_value: bool = True,
    ) -> SeqStepOutput:
        """Encode a fresh prefix [B, T, WIDTH] into empty cache slots."""

        x = self.embed(tokens)
        x = self._run_blocks_cached(tokens, x, cache, slots, start=0)
        hidden = self.final_norm(x[:, -1])
        if readout_indices is not None:
            hidden = hidden[readout_indices]
        return self._step_heads(hidden, phase=phase, with_value=with_value)

    def forward_prefix(
        self,
        tokens: torch.Tensor,
        *,
        readout_indices: torch.Tensor | None = None,
        phase: str | None = None,
        with_value: bool = True,
    ) -> SeqStepOutput:
        """Cache-free decode: re-encode [B, T, WIDTH] and read the last step.

        Heads run only on the final position so this is a like-for-like
        comparison against ``forward_step``: the difference measured is
        exactly the trunk recompute the KV cache avoids.
        """

        x = self.embed(tokens)
        x = self._run_blocks_full(tokens, x)
        hidden = self.final_norm(x[:, -1])
        if readout_indices is not None:
            hidden = hidden[readout_indices]
        return self._step_heads(hidden, phase=phase, with_value=with_value)

    def forward_step(
        self,
        tokens: torch.Tensor,
        position: int,
        cache: KVCache,
        slots: torch.Tensor | None,
        *,
        readout_indices: torch.Tensor | None = None,
        phase: str | None = None,
        with_value: bool = True,
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
        x = self._run_blocks_cached(
            tokens, x, cache, slots, start=position
        )
        hidden = self.final_norm(x[:, -1])
        if readout_indices is not None:
            hidden = hidden[readout_indices]
        return self._step_heads(hidden, phase=phase, with_value=with_value)

    def forward_step_ragged(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache,
        slots: torch.Tensor | None = None,
        *,
        readout_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append a run of tokens per row, each row at its own cache position.

        This is the merged-wave counterpart of ``forward_step``: the rows in
        one call may hold different games, different table sizes and different
        hand sizes, and each sits at whatever length its own game has reached.

        ``tokens`` is [B, WIDTH] or [B, T, WIDTH]. A merged wave also disagrees
        on *how many* tokens each row appends -- one game closes a trick while
        another does not -- so the caller pads every row to the batch's longest
        run and names the last real token of each row in ``readout_offsets``.
        Padding is safe because a padded row's surplus K/V lands on positions
        that row has not reached yet and will overwrite on a later wave, while
        its own ``key_lengths`` keeps it from ever attending there.

        Returns the final hidden state [B, d_model] rather than a
        ``SeqStepOutput``. A merged wave holds bid rows and play rows at once,
        so which action head to read is a per-row question and the caller
        answers it by grouping rows; the trunk, which is all of the cost, runs
        once for everyone.
        """

        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        if positions.shape[0] != tokens.shape[0]:
            raise ValueError("positions must carry one offset per row.")
        length = tokens.shape[1]
        write_positions = positions
        embed_positions = positions
        if length > 1:
            steps = torch.arange(length, device=positions.device)[None, :]
            # Padding runs a row past the end of its own game. Those writes are
            # never read back -- the row's ``key_lengths`` stops short of them
            # and its game ends first -- but they still have to land inside the
            # pool, and their embedding lookups inside the tables.
            write_positions = (positions[:, None] + steps).clamp(
                max=cache.max_len - 1
            )
            embed_positions = write_positions.clamp(
                max=self.config.max_seq_len - 1
            )
        x = self.embed(tokens, start=embed_positions)
        if (
            slots is None
            and self.uses_metal_trunk(tokens)
            and self.metal_trunk.decode_supported(cache)
        ):
            ple, values = self._prepare_layer_conditioning(
                tokens, x, embed_positions
            )
            # Final-norm output for every appended token; a row's padding
            # past its own run is computed and dropped, as below.
            normed = self.metal_trunk.decode(
                x, ple, values, cache, positions, metal_trunk_parameters(self)
            )
            if readout_offsets is None:
                return normed[:, -1]
            rows = torch.arange(normed.shape[0], device=normed.device)
            return normed[rows, readout_offsets]
        # Every row's own next write is at its current length, so the read has
        # to cover the longest row in the batch plus the whole run.
        span = min(int(positions.max()) + length, cache.max_len)
        x = self._run_blocks_cached(
            tokens,
            x,
            cache,
            slots,
            start=span - length,
            row_positions=write_positions,
            key_lengths=positions + 1,
            embed_positions=embed_positions,
        )
        if readout_offsets is None:
            hidden = x[:, -1]
        else:
            rows = torch.arange(x.shape[0], device=x.device)
            hidden = x[rows, readout_offsets]
        return self.final_norm(hidden)


class SeqPPOCritic(nn.Module):
    """Independent PPO critic with an optional complete-deal side input.

    The public/observer-relative token stream stays byte-for-byte identical to
    the actor's. In privileged mode, the complete initial deal is encoded as a
    fixed-size side tensor and added only to the critic's GAME position. It
    therefore adds no actor tokens, cannot desynchronise rollout cache rows,
    and cannot leak hidden cards into the deployed policy.

    A full ``SeqPlumpModel`` is retained as the trunk so an existing actor can
    initialize every compatible critic weight exactly. Policy and auxiliary
    readouts are frozen and never evaluated; only the independent trunk/value
    head and the private-deal embeddings are optimized.
    """

    def __init__(
        self,
        config: SeqModelConfig,
        *,
        privileged: bool = True,
        initialize_from: SeqPlumpModel | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.privileged = privileged
        self.backbone = SeqPlumpModel(config)
        if initialize_from is not None:
            self.backbone.load_state_dict(initialize_from.state_dict())

        d = config.d_model
        # Pair each card feature with its observer-relative owner. A plain sum
        # of separate seat and card embeddings would lose that association.
        self.private_card_embedding = nn.Embedding(
            config.max_players * NUM_CARDS + 1, d,
            padding_idx=config.max_players * NUM_CARDS,
        )
        self.private_rank_embedding = nn.Embedding(
            config.max_players * NUM_RANKS + 1, d,
            padding_idx=config.max_players * NUM_RANKS,
        )
        self.private_suit_embedding = nn.Embedding(
            config.max_players * NUM_SUITS + 1, d,
            padding_idx=config.max_players * NUM_SUITS,
        )
        # Zero preserves the actor's existing value predictions at migration;
        # embedding rows receive ordinary gradients on the first critic step.
        nn.init.zeros_(self.private_card_embedding.weight)
        nn.init.zeros_(self.private_rank_embedding.weight)
        nn.init.zeros_(self.private_suit_embedding.weight)

        for module in (
            self.backbone.bid_head,
            self.backbone.card_head,
            self.backbone.card_rank_output_embedding,
            self.backbone.card_suit_output_embedding,
            self.backbone.trick_count_head,
            self.backbone.suit_presence_head,
            self.backbone.bid_hit_head,
            self.backbone.rank_boundary_head,
            self.backbone.next_winner_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _private_deal(self, initial_hands: torch.Tensor) -> torch.Tensor:
        """Encode [B, relative-seat, card-slot] padded with NUM_CARDS."""

        if initial_hands.dim() != 3:
            raise ValueError("initial_hands must be [batch, players, cards].")
        batch, players, _ = initial_hands.shape
        if players != self.config.max_players:
            raise ValueError(
                "initial_hands player axis must equal config.max_players."
            )
        valid = initial_hands < NUM_CARDS
        safe = initial_hands.clamp_max(NUM_CARDS - 1)
        relative = torch.arange(players, device=initial_hands.device).view(1, -1, 1)

        exact_pad = self.config.max_players * NUM_CARDS
        rank_pad = self.config.max_players * NUM_RANKS
        suit_pad = self.config.max_players * NUM_SUITS
        exact_ids = torch.where(valid, relative * NUM_CARDS + safe, exact_pad)
        rank_ids = torch.where(
            valid, relative * NUM_RANKS + safe.remainder(NUM_RANKS), rank_pad
        )
        suit_ids = torch.where(
            valid, relative * NUM_SUITS + safe.div(NUM_RANKS, rounding_mode="floor"), suit_pad
        )
        vectors = (
            self.private_card_embedding(exact_ids)
            + self.private_rank_embedding(rank_ids)
            + self.private_suit_embedding(suit_ids)
        )
        count = valid.sum(dim=(1, 2), keepdim=False).clamp_min(1).to(vectors.dtype)
        return vectors.sum(dim=(1, 2)) / count.sqrt().unsqueeze(-1)

    def forward_full(
        self, tokens: torch.Tensor, initial_hands: torch.Tensor
    ) -> torch.Tensor:
        x = self.backbone.embed(tokens)
        if self.privileged:
            x = x.clone()
            x[:, 0] = x[:, 0] + self._private_deal(initial_hands)
        x = self.backbone._run_blocks_full(tokens, x)
        hidden = self.backbone.final_norm(x)
        return self.backbone.value_head(hidden).squeeze(-1)


class SeqPPOOracleCritic(nn.Module):
    """One perfect-information sequence and one value vector per game.

    The prefix contains ``P * N`` separate HAND tokens, ordered first by the
    environment's absolute seat and then by card. Their player field identifies
    the owner. The public suffix uses observer 0, so its player fields use that
    same absolute-seat convention. Output column ``s`` is consequently tied to
    input owner/actor id ``s`` without any observer-relative remapping.

    This critic runs only during the update. Its longer positional table does
    not alter the actor architecture, rollout sequence, or KV cache.
    """

    def __init__(
        self,
        config: SeqModelConfig,
        *,
        initialize_from: SeqPlumpModel | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = SeqPlumpModel(config)

        # The oracle adds (P - 1) * N card tokens. Preserve the actor's learned
        # positional rows and initialize only the critic-only tail.
        actor_positions = self.backbone.pos_embedding
        if actor_positions is not None:
            oracle_positions = nn.Embedding(
                config.oracle_max_seq_len, config.d_model
            )
            SeqPlumpModel._init_module(oracle_positions)
            with torch.no_grad():
                oracle_positions.weight[: actor_positions.num_embeddings].copy_(
                    actor_positions.weight
                )
            self.backbone.pos_embedding = oracle_positions

        # The oracle has P*N distinct all-hand prefix tokens, so every
        # position-keyed additive stream needs the same longer table.  These
        # are independent critic parameters; actor rows are copied below and
        # the critic-only tail begins at the neutral zero embedding.
        for name in ("per_layer_pos_embedding", "value_pos_embedding"):
            actor_table = getattr(self.backbone, name)
            if actor_table is None:
                continue
            oracle_table = nn.Embedding(
                config.oracle_max_seq_len, actor_table.embedding_dim
            )
            nn.init.zeros_(oracle_table.weight)
            with torch.no_grad():
                oracle_table.weight[: actor_table.num_embeddings].copy_(
                    actor_table.weight
                )
            setattr(self.backbone, name, oracle_table)

        d = config.d_model
        self.player_value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.max_players),
        )
        self.player_value_head.apply(SeqPlumpModel._init_module)
        # One contextual score per exact card. The deal partitions cards into
        # disjoint player hands, and the trainer normalizes scores separately
        # within every owner's remaining set. A redundant player x card axis
        # would make the projection five times wider without representing any
        # additional valid comparison.
        self.card_order_head = nn.Linear(d, NUM_CARDS)
        self.card_order_head.apply(SeqPlumpModel._init_module)
        if initialize_from is not None:
            self.initialize_from_actor(initialize_from)

        # These actor readouts are retained so the actor trunk can be copied
        # exactly. Trick count and next-trick winner are deliberately trainable
        # for the oracle: both are useful dense outcome targets even with
        # perfect information -- knowing every hand does not tell you who takes
        # the next trick without simulating the play. Suit
        # presence remains frozen because every held card and owner is already
        # visible in the oracle prefix, making that task a trivial identity.
        for module in (
            self.backbone.bid_head,
            self.backbone.card_head,
            self.backbone.value_head,
            # The critic reads values from its own player_value_head below;
            # the backbone's copy exists only so the actor trunk state dict
            # loads exactly, and must never receive oracle gradients.
            self.backbone.player_value_head,
            self.backbone.card_rank_output_embedding,
            self.backbone.card_suit_output_embedding,
            self.backbone.suit_presence_head,
            self.backbone.bid_hit_head,
            self.backbone.rank_boundary_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def initialize_from_actor(self, actor: SeqPlumpModel) -> None:
        """Copy every compatible actor trunk row and broadcast its value head."""

        source = actor.state_dict()
        target = self.backbone.state_dict()
        with torch.no_grad():
            for name, value in source.items():
                destination = target[name]
                if name in {
                    "pos_embedding.weight",
                    "per_layer_pos_embedding.weight",
                    "value_pos_embedding.weight",
                }:
                    destination[: value.shape[0]].copy_(value)
                elif destination.shape == value.shape:
                    destination.copy_(value)
                else:
                    raise ValueError(
                        f"Cannot initialize oracle critic parameter {name}: "
                        f"{tuple(value.shape)} -> {tuple(destination.shape)}."
                    )

            actor_first = actor.value_head[0]
            actor_last = actor.value_head[2]
            oracle_first = self.player_value_head[0]
            oracle_last = self.player_value_head[2]
            oracle_first.weight.copy_(actor_first.weight)
            oracle_first.bias.copy_(actor_first.bias)
            oracle_last.weight.copy_(
                actor_last.weight.expand(self.config.max_players, -1)
            )
            oracle_last.bias.copy_(
                actor_last.bias.expand(self.config.max_players)
            )

    def _forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.config.oracle_max_seq_len:
            raise ValueError("Oracle token sequence exceeds its position table.")
        return self.backbone.forward_hidden(tokens)

    def forward_value_and_trick(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return absolute-seat value and final-trick logits from one trunk."""

        hidden = self._forward_hidden(tokens)
        batch, length, _ = hidden.shape
        trick_logits = self.backbone.trick_count_head(hidden).view(
            batch,
            length,
            self.config.max_players,
            self.config.bid_count,
        )
        return self.player_value_head(hidden), trick_logits

    def forward_selected(
        self,
        tokens: torch.Tensor,
        seq: torch.Tensor,
        pos: torch.Tensor,
        *,
        value_seq: torch.Tensor | None = None,
        value_pos: torch.Tensor | None = None,
        dense_value_projection: bool = False,
        auxiliary_seq: torch.Tensor | None = None,
        auxiliary_pos: torch.Tensor | None = None,
        dense_auxiliary_projection: bool = False,
        with_trick: bool = False,
        with_card_order: bool = False,
        with_next_winner: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Run one causal trunk and select requested oracle readout addresses.

        ``seq``/``pos`` identify the PPO baseline rows by default.  Dense value
        training instead supplies ``value_seq``/``value_pos`` and projects the
        value MLP over the whole rectangular sequence before removing padding.
        Auxiliary heads follow the analogous path.  Thus every genuine hidden
        state receives every enabled non-policy readout while the transformer
        itself still runs exactly once.
        """

        hidden = self._forward_hidden(tokens)
        selected = hidden[seq, pos]
        if (value_seq is None) != (value_pos is None):
            raise ValueError("value_seq and value_pos must be supplied together.")
        if dense_value_projection:
            if value_seq is None:
                raise ValueError(
                    "Dense value projection requires selected addresses."
                )
            dense_values = self.player_value_head(
                hidden.reshape(-1, hidden.shape[-1])
            )
            values = dense_values[value_seq * hidden.shape[1] + value_pos]
        else:
            value_hidden = (
                selected
                if value_seq is None
                else hidden[value_seq, value_pos]
            )
            values = self.player_value_head(value_hidden)
        modules = []
        if with_trick:
            modules.append(("trick", self.backbone.trick_count_head))
        if with_card_order:
            modules.append(("card_order", self.card_order_head))
        if with_next_winner:
            modules.append(("next_winner", self.backbone.next_winner_head))
        if not modules:
            return values, None, None, None

        if (auxiliary_seq is None) != (auxiliary_pos is None):
            raise ValueError(
                "auxiliary_seq and auxiliary_pos must be supplied together."
            )
        if dense_auxiliary_projection:
            if auxiliary_seq is None:
                raise ValueError(
                    "Dense auxiliary projection requires selected addresses."
                )
            auxiliary_hidden = hidden.reshape(-1, hidden.shape[-1])
        else:
            auxiliary_hidden = (
                selected
                if auxiliary_seq is None
                else hidden[auxiliary_seq, auxiliary_pos]
            )

        # Every oracle auxiliary is a linear readout of the same selected
        # hidden rows.  Packing them issues one accelerator matmul and one
        # autograd path rather than one kernel per small head.
        packed = F.linear(
            auxiliary_hidden,
            torch.cat([module.weight for _, module in modules], dim=0),
            torch.cat([module.bias for _, module in modules], dim=0),
        )
        if dense_auxiliary_projection:
            assert auxiliary_seq is not None and auxiliary_pos is not None
            packed = packed[
                auxiliary_seq * hidden.shape[1] + auxiliary_pos
            ]
        split = dict(
            zip(
                (name for name, _ in modules),
                packed.split(
                    [module.out_features for _, module in modules], dim=-1
                ),
            )
        )
        trick_logits = split.get("trick")
        if trick_logits is not None:
            trick_logits = trick_logits.view(
                trick_logits.shape[0],
                self.config.max_players,
                self.config.bid_count,
            )
        card_order_logits = split.get("card_order")
        next_winner_logits = split.get("next_winner")
        return values, trick_logits, card_order_logits, next_winner_logits

    def forward_full(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, position, absolute_player]`` oracle values."""

        return self.player_value_head(self._forward_hidden(tokens))


def load_seq_model_state_dict(
    model: SeqPlumpModel,
    state_dict: Mapping[str, torch.Tensor],
) -> frozenset[str]:
    """Load current weights or migrate either additive model extension.

    Returns exactly the accepted missing keys. Structured card-output rows are
    neutralized to zero; the rank-boundary readout, the per-seat value head and
    the next-trick-winner readout intentionally retain their random
    initialization. No other mismatch is accepted.
    """

    result = model.load_state_dict(state_dict, strict=False)
    missing = frozenset(result.missing_keys)
    unexpected = frozenset(result.unexpected_keys)
    model_keys = frozenset(model.state_dict())
    additive_keys = ADDITIVE_CONDITIONING_KEYS & model_keys
    allowed = (
        STRUCTURED_CARD_OUTPUT_KEYS
        | RANK_BOUNDARY_HEAD_KEYS
        | EMBEDDING_NORM_KEYS
        | PLAYER_VALUE_HEAD_KEYS
        | NEXT_WINNER_HEAD_KEYS
        | additive_keys
    )
    additive_families = (
        frozenset({"bid_pressure_embedding.weight"}) & model_keys,
        frozenset(
            key
            for key in ADDITIVE_CONDITIONING_KEYS
            if key.startswith("per_layer_")
            and key not in NA_PATTERN_EMBEDDING_KEYS
        ) & model_keys,
        frozenset(
            key
            for key in ADDITIVE_CONDITIONING_KEYS
            if key.startswith("value_")
            and key not in NA_PATTERN_EMBEDDING_KEYS
        ) & model_keys,
        frozenset({"na_pattern_embedding.weight"}) & model_keys,
        frozenset({"per_layer_na_pattern_embedding.weight"}) & model_keys,
        frozenset({"value_na_pattern_embedding.weight"}) & model_keys,
    )
    complete_families = all(
        not (missing & family) or (missing & family) == family
        for family in (
            STRUCTURED_CARD_OUTPUT_KEYS,
            RANK_BOUNDARY_HEAD_KEYS,
            PLAYER_VALUE_HEAD_KEYS,
            NEXT_WINNER_HEAD_KEYS,
            *additive_families,
        )
    )
    migrated = missing <= allowed and complete_families and not unexpected
    if missing or unexpected:
        if not migrated:
            raise RuntimeError(
                "Model checkpoint state mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        if missing & STRUCTURED_CARD_OUTPUT_KEYS:
            with torch.no_grad():
                model.card_rank_output_embedding.weight.zero_()
                model.card_suit_output_embedding.weight.zero_()
        if missing & NA_PATTERN_EMBEDDING_KEYS:
            model.initialize_na_pattern_embeddings()
        if missing & EMBEDDING_NORM_KEYS:
            # A checkpoint from before the norm existed. Gain 1.0 is NOT
            # behaviour-preserving here; the caller must calibrate against real
            # tokens before the model is used or trained.
            with torch.no_grad():
                model.embedding_norm.weight.fill_(1.0)
    model.clear_card_output_cache()
    return missing
