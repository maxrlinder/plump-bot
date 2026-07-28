"""Preallocated KV cache pool with row allocation and branch copying.

One ``KVCache`` instance serves one set of model weights (the current policy
or one frozen opponent snapshot). Leaves x seats map to rows via the rollout
engine's routing table. Branching a leaf copies the parent prefix for every
seat row.

Storage is one stacked ``[n_layers, capacity, ...]`` tensor when that fits
inside the INT_MAX element limit MPSGraph imposes on a single tensor, and one
tensor per layer when it does not. ``self.k[layer]`` is a view either way, so
every read and write path is identical between the two; the only thing the
layout changes is that a stacked pool can branch-copy every layer in one
indexed op instead of ``n_layers`` of them, and grows as one allocation.

The choice is made against ``max_capacity`` rather than the starting capacity,
because ``ensure_capacity`` may grow the pool later and a layout that overflows
after growing is worse than one that was never chosen.
"""

from __future__ import annotations

import torch

from .config import SeqModelConfig


# Elements, not bytes: the limit is on a tensor's element count, so it does not
# move when the cache dtype does.
MAX_TENSOR_ELEMENTS = 2**31 - 1


class KVCache:
    def __init__(
        self,
        config: SeqModelConfig,
        capacity: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        max_len: int | None = None,
        max_capacity: int | None = None,
        poison: bool = False,
        stacked: bool | None = None,
    ):
        # Debug aid: fill unwritten rows with NaN so that reading a row before
        # it has been prefilled or branch-copied raises instead of silently
        # attending to zeros. Tests turn this on; production leaves it off.
        self.poison = poison
        self.capacity = capacity
        self.max_capacity = max(max_capacity or capacity, capacity)
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_len = max_len if max_len is not None else config.max_seq_len
        self.n_layers = config.n_layers
        self._row_shape = (config.kv_heads, self.max_len, config.head_dim)
        # ``stacked`` forces a layout; None picks the one that fits. Tests use
        # the override to exercise both paths at sizes where either would do.
        self.stacked = (
            self.fits_stacked(config, self.max_capacity, self.max_len)
            if stacked is None
            else stacked
        )
        self._base_k: torch.Tensor | None = None
        self._base_v: torch.Tensor | None = None
        self._allocate(capacity)
        self._free: list[int] = list(range(capacity - 1, -1, -1))

    @staticmethod
    def stacked_elements(
        config: SeqModelConfig, capacity: int, max_len: int
    ) -> int:
        return (
            config.n_layers
            * capacity
            * config.kv_heads
            * max_len
            * config.head_dim
        )

    @classmethod
    def fits_stacked(
        cls, config: SeqModelConfig, capacity: int, max_len: int
    ) -> bool:
        return (
            cls.stacked_elements(config, capacity, max_len)
            <= MAX_TENSOR_ELEMENTS
        )

    def fits_stacked_capacity(self, capacity: int) -> bool:
        """Same test against this pool's own row shape, post-construction."""

        elements = (
            self.n_layers
            * capacity
            * self._row_shape[0]
            * self._row_shape[1]
            * self._row_shape[2]
        )
        return elements <= MAX_TENSOR_ELEMENTS

    def _allocate(self, capacity: int) -> None:
        """Point ``self.k`` / ``self.v`` at per-layer storage for ``capacity``."""

        if self.stacked:
            self._base_k = self._empty_stacked(capacity)
            self._base_v = self._empty_stacked(capacity)
            # unbind returns views into the base, so writing through
            # self.k[layer] writes the stacked tensor. That is what lets the
            # rest of this class stay layout-agnostic.
            self.k = list(self._base_k.unbind(0))
            self.v = list(self._base_v.unbind(0))
        else:
            self._base_k = self._base_v = None
            self.k = [self._empty(capacity) for _ in range(self.n_layers)]
            self.v = [self._empty(capacity) for _ in range(self.n_layers)]

    @property
    def _fill(self) -> float:
        # Zero-filled, not uninitialised. Rows are meant to be written before
        # they are read, but the dense fast path reads whole row ranges, so a
        # gap would attend to garbage instead of failing loudly.
        # ``poison=True`` turns those gaps into NaN so tests can find them.
        return float("nan") if self.poison else 0.0

    def _empty(self, capacity: int) -> torch.Tensor:
        return torch.full(
            (capacity, *self._row_shape),
            self._fill,
            device=self.device,
            dtype=self.dtype,
        )

    def _empty_stacked(self, capacity: int) -> torch.Tensor:
        return torch.full(
            (self.n_layers, capacity, *self._row_shape),
            self._fill,
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def free_count(self) -> int:
        return len(self._free)

    def ensure_capacity(self, rows: int) -> None:
        """Grow to hold at least ``rows`` rows, preserving cached content.

        A floor-style leaf budget lets a layer overshoot by its branching
        factor, so the high-water row count is not known ahead of time. This
        copies the live cache, so it grows in large steps and never past
        ``max_capacity`` -- the caller stops branching before that point.
        """

        if rows <= self.capacity:
            return
        if rows > self.max_capacity:
            raise RuntimeError(
                f"KV cache needs {rows} rows, memory budget allows "
                f"{self.max_capacity}."
            )
        # Grow by half rather than doubling: the old and new pools are both
        # resident during the copy, so an aggressive step costs more peak
        # memory than the rows it buys.
        new_capacity = min(rows + rows // 2, self.max_capacity)
        # max_capacity can be raised after construction (the collector reuses a
        # pool across shapes), so a pool that was stacked when it was built can
        # be asked to grow past the limit. Drop to per-layer rather than
        # allocating a tensor that overflows.
        if self.stacked and not self.fits_stacked_capacity(new_capacity):
            self.stacked = False
        if self.stacked:
            for name in ("_base_k", "_base_v"):
                grown = self._empty_stacked(new_capacity)
                grown[:, : self.capacity] = getattr(self, name)
                setattr(self, name, grown)
            self.k = list(self._base_k.unbind(0))
            self.v = list(self._base_v.unbind(0))
        else:
            for store in (self.k, self.v):
                for layer in range(self.n_layers):
                    grown = self._empty(new_capacity)
                    grown[: self.capacity] = store[layer]
                    store[layer] = grown
        self.capacity = new_capacity
        # Return the old pool to the system instead of leaving it in the
        # allocator's free list, where it still counts against the budget.
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def alloc(self, count: int) -> list[int]:
        if count > len(self._free):
            raise RuntimeError(
                f"KV cache exhausted: requested {count}, free {len(self._free)}."
            )
        return [self._free.pop() for _ in range(count)]

    def free(self, slots: list[int]) -> None:
        self._free.extend(slots)
        if len(self._free) > self.capacity:
            raise RuntimeError("KV cache freed more slots than exist.")

    def reset(self) -> None:
        self._free = list(range(self.capacity - 1, -1, -1))

    # Cap on the gather buffer one branch_copy chunk may materialise. The
    # advanced-index read builds a dense [rows, heads, length, dim] temporary,
    # which at a wide layer rivals the pool itself and pushes the allocator
    # into thrashing right when memory is tightest.
    COPY_CHUNK_BYTES = 256 * 1024 * 1024

    def branch_copy(
        self,
        parents: torch.Tensor,
        children: torch.Tensor,
        length: int,
    ) -> None:
        """Copy the first ``length`` cached positions from parents to children."""

        row_bytes = max(
            self._row_shape[0] * length * self._row_shape[2] * self.dtype.itemsize, 1
        )
        # A stacked copy gathers every layer at once, so one chunk materialises
        # n_layers times the temporary. Shrink the chunk to match, or the cap
        # this constant exists to enforce would be exceeded n_layers-fold.
        if self.stacked:
            row_bytes *= self.n_layers
        chunk = max(self.COPY_CHUNK_BYTES // row_bytes, 1)
        for start in range(0, parents.shape[0], chunk):
            src = parents[start : start + chunk]
            dst = children[start : start + chunk]
            if self.stacked:
                self._base_k[:, dst, :, :length] = self._base_k[:, src, :, :length]
                self._base_v[:, dst, :, :length] = self._base_v[:, src, :, :length]
            else:
                for layer in range(self.n_layers):
                    self.k[layer][dst, :, :length] = self.k[layer][src, :, :length]
                    self.v[layer][dst, :, :length] = self.v[layer][src, :, :length]

    def write_range(
        self,
        layer: int,
        slots: torch.Tensor | None,
        start: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Write [B, kv_heads, T, head_dim]; ``slots=None`` means rows 0..B-1."""

        length = k_new.shape[2]
        stop = start + length
        if slots is None:
            count = k_new.shape[0]
            self.k[layer][:count, :, start:stop] = k_new.to(self.dtype)
            self.v[layer][:count, :, start:stop] = v_new.to(self.dtype)
        else:
            self.k[layer][slots, :, start:stop] = k_new.to(self.dtype)
            self.v[layer][slots, :, start:stop] = v_new.to(self.dtype)

    def read(
        self,
        layer: int,
        slots: torch.Tensor | None,
        length: int,
        count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached K/V [B, kv_heads, length, head_dim] in the cache dtype.

        ``slots=None`` returns zero-copy views of rows 0..count-1 — the fast
        path for wave-dense batches.
        """

        if slots is None:
            if count is None:
                raise ValueError("Dense read requires an explicit row count.")
            k = self.k[layer][:count, :, :length]
            v = self.v[layer][:count, :, :length]
            if self.poison and (k.isnan().any() or v.isnan().any()):
                raise RuntimeError(
                    f"Read of unwritten KV rows: layer={layer} "
                    f"length={length} count={count}."
                )
            return k, v
        return (
            self.k[layer][slots, :, :length],
            self.v[layer][slots, :, :length],
        )
