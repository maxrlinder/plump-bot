"""Drain Objective-C autorelease pools around accelerator work on macOS.

PyTorch's MPS backend hands back *autoreleased* Metal and MPS objects from
some of its per-operation paths. A Cocoa application drains those on every
run-loop turn; a plain Python training loop has no run loop, so any created
outside an explicit pool are never released. They accumulate as native
``MALLOC_SMALL`` heap -- invisible to tracemalloc and to Python's gc -- at a
constant rate per operation.

Measured on rl-v6 (torch 2.13.0.dev20260518, 16-layer model, MPS): 3.9 MB of
native heap per training iteration, reaching 20 GB of footprint -- 19 GB of
it compressed or swapped -- after ~5,500 iterations. Replaying an identical
rollout leaked at the same rate as fresh ones, so it is not a shape-keyed
cache that saturates. In an A/B/A probe at 96 games per iteration, wrapping
each iteration in a pool cut the growth from +2.7 to +0.46 MB/iteration, and
it returned to +2.6 as soon as the pool was removed.

Elsewhere this is a no-op, so callers need not branch on platform.
"""

from __future__ import annotations

import ctypes
import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator


@lru_cache(maxsize=1)
def _libobjc() -> ctypes.CDLL | None:
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL("/usr/lib/libobjc.dylib")
    except OSError:
        return None
    library.objc_autoreleasePoolPush.argtypes = []
    library.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    library.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
    library.objc_autoreleasePoolPop.restype = None
    return library


@contextmanager
def autorelease_pool() -> Iterator[None]:
    """Release every object autoreleased on this thread inside the block.

    Pools nest, and must be pushed and popped on the same thread, which a
    ``with`` block guarantees. The pop also runs when the block raises, so a
    failed iteration cannot leave its objects behind either.
    """

    library = _libobjc()
    if library is None:
        yield
        return
    token = library.objc_autoreleasePoolPush()
    try:
        yield
    finally:
        library.objc_autoreleasePoolPop(token)
