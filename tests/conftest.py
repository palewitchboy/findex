"""Shared fixtures: a tiny on-disk corpus and the index built from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.index import Index, build_index

DOCS = {
    "alpha.txt": (
        "Title: Alpha\n\nThe quick brown fox jumps over the lazy dog. A fox is quick.\n"
    ),
    "beta.txt": (
        "Title: Beta\n\nThe lazy dog sleeps. The dog dreams of a fox and rabbits.\n"
    ),
    "gamma.txt": ("Title: Gamma\n\nRabbits and a fox are quick. Nothing else here.\n"),
}


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def index(corpus: Path) -> Index:
    return build_index(corpus)
