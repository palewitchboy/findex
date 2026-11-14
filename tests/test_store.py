"""Lab 2 - persistence: pickle and json round-trips, format sniffing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.store import (
    load,
    load_json,
    load_pickle,
    save,
    save_json,
    save_pickle,
    storage_of,
)


def _fingerprint(index: Index) -> dict[str, object]:
    return {
        "postings": {term: dict(index.tf_items(term)) for term in index.terms()},
        "doc_lengths": index.doc_lengths,
        "meta": {d: (m.path, m.title, m.n_tokens) for d, m in index.doc_meta.items()},
        "n_tokens": index.n_tokens,
    }


@pytest.mark.parametrize("fmt", ["pickle", "json"])
def test_round_trip_preserves_everything(
    index: Index, tmp_path: Path, fmt: str
) -> None:
    path = tmp_path / f"index.{fmt}"
    save(index, path, fmt=fmt)
    restored = load(path)
    assert _fingerprint(restored) == _fingerprint(index)
    assert restored.positions is None


def test_round_trip_keeps_positions(tmp_path: Path, corpus: Path) -> None:
    index = build_index(corpus, positions=True)
    path = tmp_path / "with_positions.json"
    save_json(index, path)
    restored = load_json(path)
    assert restored.positions is not None
    assert restored.positions["fox"][0] == (5, 12)


def test_load_sniffs_the_format(index: Index, tmp_path: Path) -> None:
    pickled, as_json = tmp_path / "a.bin", tmp_path / "b.json"
    save_pickle(index, pickled)
    save_json(index, as_json)
    assert _fingerprint(load(pickled)) == _fingerprint(index)
    assert _fingerprint(load(as_json)) == _fingerprint(index)
    assert load_pickle(pickled).n_tokens == load_json(as_json).n_tokens


def test_unknown_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"?not an index")
    with pytest.raises(ValueError):
        load(path)
    with pytest.raises(ValueError):
        save(build_index(tmp_path), tmp_path / "x", fmt="csv")


def test_json_file_is_readable_and_stable(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    save_json(index, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["postings"]["fox"] == [[0, 2], [1, 1], [2, 1]]
    assert data["doc_meta"][0]["title"] == "Alpha"
    assert '"fox"' in path.read_text(encoding="utf-8")  # not escaped, not binary


def test_storage_of_reports_the_layout(corpus: Path) -> None:
    assert storage_of(build_index(corpus, storage="objects")) == "objects"
    assert storage_of(build_index(corpus, storage="arrays")) == "arrays"
