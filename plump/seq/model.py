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
    NUM_CARDS,
    NUM_RANKS,
    NUM_SUITS,
    SLOT_CARD,
    SLOT_RANK,
    SLOT_REMAINING_HAND_START,
    SLOT_SUIT,
    SLOT_TYPE,
    TOKEN_WIDTH,
    TOKEN_TRICK_WIN,
    SeqModelConfig,
)
from .kv import KVCache

# Auxiliary heads, selectable one by one in forward_full: a caller whose loss
# weights one at zero should not pay for its forward or its backward.
AUX_HEADS = ("trick", "suit", "bid_hit")
SEQ_MODEL_FORMAT_VERSION = 2
STRUCTURED_CARD_OUTPUT_KEYS = frozenset(
    {
        "card_rank_output_embedding.weight",
        "card_suit_output_embedding.weight",
    }
)


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
        does), limiting sensitivity to reduced-precision cache storage.

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
        Attention runs in the cache dtype. Matching it to the autocast dtype
        avoids a query cast while retaining two-byte cache storage.
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
        self._card_input_weight_cache: torch.Tensor | None = None
        self._card_output_weight_cache: torch.Tensor | None = None
        config.validate()
        self.config = config
        d = config.d_model
        # All twelve token slots share one embedding table with per-slot id
        # offsets. Twelve separate lookups plus eleven adds would be ~23
        # kernels per forward, and the rollout runs one forward per game
        # event, so that overhead dominates at small batch sizes.
        sizes = config.base_slot_vocab_sizes
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

        self.apply(self._init_module)
        # A zero additive start preserves the former head exactly—important
        # both for cold starts and for resuming existing checkpoints. Adam gets
        # nonzero gradients for these rows immediately, so zero is a neutral
        # starting contribution rather than a frozen or symmetric dead path.
        nn.init.zeros_(self.card_rank_output_embedding.weight)
        nn.init.zeros_(self.card_suit_output_embedding.weight)
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
        card_ids = torch.arange(NUM_CARDS, device=self.slot_embedding.weight.device)
        weight = (
            self.slot_embedding(card_ids + self.slot_offsets[SLOT_CARD])
            + self.slot_embedding(
                self._card_output_rank_ids + self.slot_offsets[SLOT_RANK]
            )
            + self.slot_embedding(
                self._card_output_suit_ids + self.slot_offsets[SLOT_SUIT]
            )
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
        weight = (
            self.card_head.weight
            + self.card_rank_output_embedding(self._card_output_rank_ids)
            + self.card_suit_output_embedding(self._card_output_suit_ids)
        )
        if not self.training and not torch.is_grad_enabled():
            self._card_output_weight_cache = weight
        return weight

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

    def embed(self, tokens: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Embed base slots, remaining-hand cards, and absolute positions."""

        if tokens.shape[-1] != TOKEN_WIDTH:
            raise ValueError(f"Tokens must have width {TOKEN_WIDTH}.")
        base = tokens[..., :BASE_TOKEN_WIDTH]
        x = self.slot_embedding(base + self.slot_offsets).sum(dim=-2)

        # Only TRICK_WIN rows contain remaining-card ids. Select those rows
        # before gathering so a normal PLAY/BID/TURN step pays no dense
        # [rows, 10, d_model] temporary. Each card reuses exactly the same
        # exact-card + rank + suit input direction as its HAND/PLAY token.
        trick_win = base[..., SLOT_TYPE] == TOKEN_TRICK_WIN
        remaining = tokens[..., SLOT_REMAINING_HAND_START:][trick_win]
        if remaining.numel():
            valid = remaining < NUM_CARDS
            safe_ids = remaining.clamp_max(NUM_CARDS - 1)
            card_vectors = F.embedding(
                safe_ids,
                self.effective_card_input_weight(),
            )
            hand_sum = (
                card_vectors * valid.unsqueeze(-1).to(card_vectors.dtype)
            ).sum(dim=-2)
            x = x.clone()
            x[trick_win] = x[trick_win] + hand_sum
        positions = torch.arange(
            start, start + tokens.shape[1], device=tokens.device
        )
        return x + self.pos_embedding(positions)

    def _step_heads(
        self,
        hidden: torch.Tensor,
        *,
        phase: str | None = None,
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
            value=self.value_head(hidden).squeeze(-1),
        )

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full causal trunk without materializing any readout head."""

        x = self.embed(tokens)
        for block in self.blocks:
            x = block.forward_full(x)
        return self.final_norm(x)

    def policy_logits(self, hidden: torch.Tensor, phase: str) -> torch.Tensor:
        """Evaluate only the requested action head on selected hidden rows."""

        if phase == "bid":
            return self.bid_head(hidden)
        if phase == "play":
            return self._card_logits(hidden)
        raise ValueError(f"Unknown policy phase {phase!r}.")

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
        return output

    def forward_prefill(
        self,
        tokens: torch.Tensor,
        cache: KVCache,
        slots: torch.Tensor | None,
        *,
        readout_indices: torch.Tensor | None = None,
        phase: str | None = None,
    ) -> SeqStepOutput:
        """Encode a fresh prefix [B, T, WIDTH] into empty cache slots."""

        x = self.embed(tokens)
        for layer, block in enumerate(self.blocks):
            x = block.forward_cached(x, cache, layer, slots, start=0)
        hidden = self.final_norm(x[:, -1])
        if readout_indices is not None:
            hidden = hidden[readout_indices]
        return self._step_heads(hidden, phase=phase)

    def forward_prefix(
        self,
        tokens: torch.Tensor,
        *,
        readout_indices: torch.Tensor | None = None,
        phase: str | None = None,
    ) -> SeqStepOutput:
        """Cache-free decode: re-encode [B, T, WIDTH] and read the last step.

        Heads run only on the final position so this is a like-for-like
        comparison against ``forward_step``: the difference measured is
        exactly the trunk recompute the KV cache avoids.
        """

        x = self.embed(tokens)
        for block in self.blocks:
            x = block.forward_full(x)
        hidden = self.final_norm(x[:, -1])
        if readout_indices is not None:
            hidden = hidden[readout_indices]
        return self._step_heads(hidden, phase=phase)

    def forward_step(
        self,
        tokens: torch.Tensor,
        position: int,
        cache: KVCache,
        slots: torch.Tensor | None,
        *,
        readout_indices: torch.Tensor | None = None,
        phase: str | None = None,
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
        if readout_indices is not None:
            hidden = hidden[readout_indices]
        return self._step_heads(hidden, phase=phase)


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
        for block in self.backbone.blocks:
            x = block.forward_full(x)
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
        oracle_positions = nn.Embedding(config.oracle_max_seq_len, config.d_model)
        SeqPlumpModel._init_module(oracle_positions)
        with torch.no_grad():
            oracle_positions.weight[: actor_positions.num_embeddings].copy_(
                actor_positions.weight
            )
        self.backbone.pos_embedding = oracle_positions

        d = config.d_model
        self.player_value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.max_players),
        )
        self.player_value_head.apply(SeqPlumpModel._init_module)
        if initialize_from is not None:
            self.initialize_from_actor(initialize_from)

        # These actor readouts are retained so the actor trunk can be copied
        # exactly. Trick count is deliberately trainable for the oracle: it is
        # a useful dense outcome target even with perfect information. Suit
        # presence remains frozen because every held card and owner is already
        # visible in the oracle prefix, making that task a trivial identity.
        for module in (
            self.backbone.bid_head,
            self.backbone.card_head,
            self.backbone.value_head,
            self.backbone.card_rank_output_embedding,
            self.backbone.card_suit_output_embedding,
            self.backbone.suit_presence_head,
            self.backbone.bid_hit_head,
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
                if name == "pos_embedding.weight":
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
        x = self.backbone.embed(tokens)
        for block in self.backbone.blocks:
            x = block.forward_full(x)
        return self.backbone.final_norm(x)

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

    def forward_full(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, position, absolute_player]`` oracle values."""

        return self.player_value_head(self._forward_hidden(tokens))


def load_seq_model_state_dict(
    model: SeqPlumpModel,
    state_dict: Mapping[str, torch.Tensor],
) -> bool:
    """Load current weights or migrate the pre-structured card output.

    Returns whether the two additive output embeddings were absent. No other
    missing or unexpected state is accepted, keeping this narrow migration
    from becoming permissive legacy-checkpoint loading.
    """

    result = model.load_state_dict(state_dict, strict=False)
    missing = frozenset(result.missing_keys)
    unexpected = frozenset(result.unexpected_keys)
    migrated = missing == STRUCTURED_CARD_OUTPUT_KEYS and not unexpected
    if missing or unexpected:
        if not migrated:
            raise RuntimeError(
                "Model checkpoint state mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            model.card_rank_output_embedding.weight.zero_()
            model.card_suit_output_embedding.weight.zero_()
    model.clear_card_output_cache()
    return migrated
