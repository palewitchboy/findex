"""Lab 3 - ``@timed``: wraps, logging, and behaviour when the call raises."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from findex.decorators import timed
from findex.index import build_index


def test_timed_returns_the_value_unchanged() -> None:
    @timed
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert add(a=2, b=3) == 5


def test_timed_preserves_the_function_identity() -> None:
    @timed
    def documented(x: int) -> int:
        """A docstring that must survive."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring that must survive."
    assert documented.__wrapped__ is not None  # functools.wraps sets this
    assert documented.__module__ == __name__


def test_timed_logs_the_function_name_and_milliseconds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="findex.decorators")

    @timed
    def slow_enough() -> None:
        time.sleep(0.002)

    slow_enough()
    assert "slow_enough took" in caplog.text
    assert "ms" in caplog.text


def test_timed_logs_even_when_the_call_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="findex.decorators")

    @timed
    def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        boom()
    assert "boom took" in caplog.text


def test_timed_is_silent_below_debug(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="findex.decorators")

    @timed
    def quiet() -> None:
        return None

    quiet()
    assert caplog.text == ""


def test_build_index_is_decorated(
    corpus: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="findex.decorators")
    build_index(corpus)
    assert build_index.__name__ == "build_index"
    assert "build_index took" in caplog.text
