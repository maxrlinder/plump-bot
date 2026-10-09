"""The autorelease helper must be safe to use unconditionally."""

from __future__ import annotations

import pytest

from plump.seq.autorelease import autorelease_pool


def test_pools_nest_and_pop_when_the_block_raises():
    with autorelease_pool():
        with autorelease_pool():
            pass

    with pytest.raises(RuntimeError):
        with autorelease_pool():
            raise RuntimeError("iteration failed")

    # Popping in the failure path left the pool stack balanced.
    with autorelease_pool():
        pass
