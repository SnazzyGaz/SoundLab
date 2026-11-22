"""Speed helpers for interactive use (the slider app). None of them changes a single output sample.

* threads()     - how many threads the band loops may use: SOUNDLAB_THREADS, default 1 (so batch jobs and tests that
                  run one process per core stay single-threaded). The slider app raises it to the core count.
* band_map(), ordered_map() - run a per-band function over the bands on those threads and give the results back in
                  band order, so sums keep the serial order and stay bit-identical. numpy's FFTs release the GIL.
* memo_arrays() - a byte-bounded LRU memo for deterministic array functions (keyed on a digest of array arguments),
                  so a slider move that leaves a stem's noise seeds alone reuses its rebuilt noise. Returns copies.
* cancel        - cooperative cancellation: the slider app marks a render stale when a newer one arrives, and the
                  long loops call check() between iterations to stop early.
"""
from __future__ import annotations

import functools
import hashlib
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_threads = None
_pool = None
_pool_n = 0
_pool_lock = threading.Lock()


def threads() -> int:
    if _threads is not None:
        return _threads
    try:
        return max(1, int(os.environ.get("SOUNDLAB_THREADS", "1")))
    except ValueError:
        return 1


def set_threads(n: int | None) -> None:
    """Threads for the band loops in this process (None: back to SOUNDLAB_THREADS / 1)."""
    global _threads
    _threads = None if n is None else max(1, int(n))


def _get_pool(n: int) -> ThreadPoolExecutor:
    global _pool, _pool_n
    with _pool_lock:
        if _pool is None or _pool_n != n:
            if _pool is not None:
                _pool.shutdown(wait=False)
            _pool, _pool_n = ThreadPoolExecutor(n, thread_name_prefix="soundlab-band"), n
        return _pool


def band_map(fn, items):
    """[fn(*it) for it in items], on threads() threads, results in order."""
    items = list(items)
    n = threads()
    if n <= 1 or len(items) < 2:
        return [fn(*it) for it in items]
    return list(_get_pool(n).map(lambda it: fn(*it), items))


def ordered_map(fn, items, ahead: int = 2):
    """Yield fn(*it) for it in items, in item order, computed on threads() threads with at most ahead x threads
    results alive at once; check()s for cancellation between batches."""
    items = list(items)
    n = threads()
    if n <= 1 or len(items) < 2:
        for it in items:
            check()
            yield fn(*it)
        return
    pool = _get_pool(n)
    step = ahead * n
    for i in range(0, len(items), step):
        check()
        yield from pool.map(lambda it: fn(*it), items[i:i + step])


# --------------------------------------------------------------------------------------------- cancellation
class Superseded(Exception):
    """A newer render replaced this one (slider app)."""


_local = threading.local()


class cancel_scope:
    """with cancel_scope(is_stale): ... - inside, check() raises Superseded once is_stale() is true."""

    def __init__(self, is_stale):
        self.is_stale = is_stale

    def __enter__(self):
        self.prev = getattr(_local, "is_stale", None)
        _local.is_stale = self.is_stale
        return self

    def __exit__(self, *exc):
        _local.is_stale = self.prev
        return False


def check() -> None:
    f = getattr(_local, "is_stale", None)
    if f is not None and f():
        raise Superseded("superseded by a newer render")


# --------------------------------------------------------------------------------------------- memo
_block_memo = False


def set_block_memo(on: bool) -> None:
    """Reuse whole recipe blocks between renders (the slider app). Off by default: library renders run every block
    (some models record their last render for inspection)."""
    global _block_memo
    _block_memo = bool(on)


def block_memo_on() -> bool:
    return _block_memo and os.environ.get("SOUNDLAB_MEMO", "1") != "0"


def digest(a: np.ndarray) -> tuple:
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, hashlib.blake2b(a.view(np.uint8).reshape(-1), digest_size=16).digest())


def _limit_bytes() -> int:
    try:
        return int(float(os.environ.get("SOUNDLAB_MEMO_MB", "512")) * 2 ** 20)
    except ValueError:
        return 512 * 2 ** 20


class _Memo:
    def __init__(self):
        self.d: OrderedDict = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self.lock:
            v = self.d.get(key)
            if v is None:
                self.misses += 1
                return None
            self.d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key, val, nbytes):
        lim = _limit_bytes()
        if nbytes > lim:
            return
        with self.lock:
            if key in self.d:
                return
            self.d[key] = (val, nbytes)
            self.bytes += nbytes
            while self.bytes > lim and self.d:
                _, (_, nb) = self.d.popitem(last=False)
                self.bytes -= nb

    def clear(self):
        with self.lock:
            self.d.clear()
            self.bytes = 0
            self.hits = self.misses = 0


MEMO = _Memo()


def _key_part(v):
    if isinstance(v, np.ndarray):
        return ("nd",) + digest(v)
    if isinstance(v, (list, tuple)):
        return (type(v).__name__,) + tuple(_key_part(x) for x in v)
    if isinstance(v, (np.integer, np.floating)):
        return (type(v).__name__, v.item())
    if v is None or isinstance(v, (int, float, str, bool, bytes)):
        return (type(v).__name__, v)
    raise TypeError(f"memo_arrays: unhashable argument {type(v).__name__}")


def _nbytes(v) -> int:
    if isinstance(v, np.ndarray):
        return v.nbytes
    if isinstance(v, (list, tuple)):
        return sum(_nbytes(x) for x in v) + 64
    return 64


def _copy(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, tuple):
        return tuple(_copy(x) for x in v)
    if isinstance(v, list):
        return [_copy(x) for x in v]
    return v


def memo_arrays(fn):
    """Memoise a deterministic function of arrays and plain values. The cache holds its own copy of the result and
    hands out copies, so callers may change what they get."""
    name = f"{fn.__module__}.{fn.__qualname__}"

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if os.environ.get("SOUNDLAB_MEMO", "1") == "0":
            return fn(*args, **kw)
        key = (name, tuple(_key_part(a) for a in args), tuple(sorted((k, _key_part(v)) for k, v in kw.items())))
        hit = MEMO.get(key)
        if hit is not None:
            return _copy(hit[0])
        val = fn(*args, **kw)
        MEMO.put(key, _copy(val), _nbytes(val))
        return val

    wrapper.uncached = fn
    return wrapper
