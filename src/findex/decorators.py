"""Lab 3 - the decorator toolbox: ``@timed`` and the query cache it feeds.

A decorator is a function that takes a function and returns a function:

    @timed
    def build_index(...): ...

is *exactly* ``build_index = timed(build_index)``. Nothing about the compiler is
involved, which is why a decorator can be written in five lines.

``functools.wraps`` is not decoration for its own sake: it copies ``__name__``,
``__qualname__``, ``__doc__``, ``__module__`` and ``__dict__`` onto the wrapper
and sets ``__wrapped__``, so ``help()``, ``repr()``, ``inspect.signature()`` and
every test that introspects the function still see the original. Forget it and
``build_index.__name__`` becomes ``"wrapper"`` - and the log line below starts
lying about which function was slow.

The timing is logged in a ``finally`` block, so a call that *raises* is still
measured and still logged; the exception is not swallowed (no ``return`` in the
``finally``, no ``except``).
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Wrap ``fn`` so every call logs its wall-clock duration in milliseconds."""
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.debug("%s took %.1f ms", fn.__name__, elapsed_ms)

    return wrapper
