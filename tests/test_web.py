"""Lab 7 - the web layer, exercised through ``TestClient``.

What is being tested is the **boundary**, not the search engine: labs 2-3
already own ranking and snippets, and their tests still run. Here every case is
about HTTP semantics - a validated parameter is a 422, an unknown document is a
404, an unloaded index is a 503, and the HTML routes render the same data the
JSON routes return.

The fixtures deliberately do **not** mock the app's internals:

* the normal client gets a real pickle on disk, so the real ``lifespan`` runs
  and ``/health``/``/stats`` read what startup actually loaded;
* the "no index" client points ``INDEX_PATH`` at a file that does not exist,
  which is the deployment accident this lab is supposed to survive;
* one test uses ``dependency_overrides`` the way the brief describes, to prove
  a route can be handed a different index without being touched.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.store import save_pickle
from findex.web.app import create_app, get_index
from findex.web.settings import Settings


def make_client(index: Index, tmp_path: Path, **overrides: object) -> TestClient:
    """An app whose lifespan really loads ``index`` from a real file."""
    path = tmp_path / "index.pkl"
    save_pickle(index, path)
    settings = Settings(index_path=path, **overrides)  # type: ignore[arg-type]
    return TestClient(create_app(settings))


@pytest.fixture()
def client(index: Index, tmp_path: Path) -> Iterator[TestClient]:
    with make_client(index, tmp_path) as test_client:
        yield test_client


@pytest.fixture()
def broken_client(tmp_path: Path) -> Iterator[TestClient]:
    """INDEX_PATH points at nothing: the app must come up and say so."""
    app = create_app(Settings(index_path=tmp_path / "missing.pkl"))
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- search
def test_search_returns_ranked_hits(client: TestClient) -> None:
    response = client.get("/search", params={"q": "fox"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "fox"
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["pages"] == 1
    assert body["k"] == 10
    assert body["scorer"] == "bm25"
    assert body["took_ms"] >= 0
    assert len(body["results"]) == 3
    scores = [hit["score"] for hit in body["results"]]
    assert scores == sorted(scores, reverse=True)
    first = body["results"][0]
    assert set(first) == {"doc_id", "title", "score", "snippet"}
    assert first["title"] == "Alpha"  # two foxes, the best BM25 hit


def test_search_snippet_marks_query_terms(client: TestClient) -> None:
    body = client.get("/search", params={"q": "fox"}).json()
    assert any("[fox]" in hit["snippet"].lower() for hit in body["results"])


def test_search_scorer_tfidf_is_honoured(client: TestClient) -> None:
    body = client.get("/search", params={"q": "dog", "scorer": "tfidf"}).json()
    assert body["scorer"] == "tfidf"
    assert body["total"] > 0


def test_search_async_route_returns_the_same_ranking(client: TestClient) -> None:
    ranked = client.get("/search", params={"q": "dog", "k": 5}).json()
    inline = client.get("/search/async", params={"q": "dog", "k": 5}).json()
    assert [hit["doc_id"] for hit in ranked["results"]] == [
        hit["doc_id"] for hit in inline["results"]
    ]
    assert ranked["total"] == inline["total"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},  # empty
        {"q": "   "},  # whitespace only
        {"q": "fox", "k": 0},  # below 1
        {"q": "fox", "k": "abc"},  # not an int
        {"q": "fox", "k": 1001},  # above the absolute ceiling
        {"q": "fox", "page": 0},
        {"q": "fox", "scorer": "cosine"},
    ],
)
def test_search_bad_parameters_are_422(
    client: TestClient, params: dict[str, object]
) -> None:
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_search_k_above_max_k_is_422(index: Index, tmp_path: Path) -> None:
    """MAX_K comes from settings, so this app's ceiling is 5, not 100."""
    with make_client(index, tmp_path, max_k=5) as test_client:
        ok = test_client.get("/search", params={"q": "fox", "k": 5})
        assert ok.status_code == 200
        response = test_client.get("/search", params={"q": "fox", "k": 6})
    assert response.status_code == 422
    assert "MAX_K" in response.json()["detail"]


def test_search_unparseable_query_is_400(client: TestClient) -> None:
    response = client.get("/search", params={"q": "fox AND ("})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("bad query:")


def test_pagination_slices_one_ranking(client: TestClient) -> None:
    first = client.get("/search", params={"q": "fox", "k": 1, "page": 1}).json()
    second = client.get("/search", params={"q": "fox", "k": 1, "page": 2}).json()
    assert first["total"] == second["total"] == 3
    assert first["pages"] == second["pages"] == 3
    assert len(first["results"]) == len(second["results"]) == 1
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]
    # page 2 is the second-best hit, not a re-ranking of a fresh query
    whole = client.get("/search", params={"q": "fox", "k": 3}).json()
    assert second["results"][0]["doc_id"] == whole["results"][1]["doc_id"]


def test_page_past_the_end_is_empty_but_total_is_not(client: TestClient) -> None:
    body = client.get("/search", params={"q": "fox", "k": 10, "page": 99}).json()
    assert body["total"] == 3
    assert body["results"] == []


def test_search_no_matches(client: TestClient) -> None:
    body = client.get("/search", params={"q": "zeppelin"}).json()
    assert body["total"] == 0
    assert body["results"] == []
    assert body["pages"] == 1


# --------------------------------------------------------------------------- stats
def test_stats_reports_the_loaded_index(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == 3
    assert body["terms"] > 0
    assert body["postings"] > 0
    assert body["tokens"] > 0
    assert body["avg_doc_length"] > 0
    assert body["uptime_s"] >= 0


# --------------------------------------------------------------------------- health
def test_health_is_200_when_the_index_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["documents"] == 3


def test_health_is_503_without_an_index(broken_client: TestClient) -> None:
    response = broken_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "index-not-loaded"
    assert body["index_loaded"] is False
    assert body["documents"] == 0
    assert "missing.pkl" in body["detail"] or "FileNotFoundError" in body["detail"]


def test_search_is_503_without_an_index(broken_client: TestClient) -> None:
    response = broken_client.get("/search", params={"q": "fox"})
    assert response.status_code == 503
    assert "detail" in response.json()


def test_index_is_loaded_once(
    index: Index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five requests, one ``load`` - that is what lifespan is for."""
    calls: list[Path] = []
    # `import findex.web.app` would bind the *FastAPI instance* (the package
    # exports `app`), so the module object is fetched from sys.modules.
    module = importlib.import_module("findex.web.app")
    real_load = module.load

    def counting(path: Path, fmt: str | None = None) -> Index:
        calls.append(Path(path))
        return real_load(path, fmt)

    monkeypatch.setattr(module, "load", counting)
    with make_client(index, tmp_path) as test_client:
        for _ in range(5):
            assert test_client.get("/search", params={"q": "fox"}).status_code == 200
    assert len(calls) == 1


# --------------------------------------------------------------------------- documents
def test_document_json(client: TestClient) -> None:
    body = client.get("/api/doc/0").json()
    assert body["doc_id"] == 0
    assert body["title"] == "Alpha"
    assert body["n_tokens"] > 0
    assert "quick brown fox" in body["text"]
    assert body["truncated"] is False


def test_document_json_404(client: TestClient) -> None:
    response = client.get("/api/doc/999")
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_document_page_highlights_terms(client: TestClient) -> None:
    response = client.get("/docs/0", params={"q": "fox"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<mark>fox</mark>" in response.text
    assert "Alpha" in response.text


def test_document_page_404(client: TestClient) -> None:
    assert client.get("/docs/999").status_code == 404


def test_document_page_without_query_has_no_marks(client: TestClient) -> None:
    response = client.get("/docs/0")
    assert response.status_code == 200
    assert "<mark>" not in response.text


def test_document_text_is_capped(index: Index, tmp_path: Path) -> None:
    """doc_max_chars bounds the page; the cut is reported, not hidden."""
    with make_client(index, tmp_path, doc_max_chars=20) as test_client:
        body = test_client.get("/api/doc/0").json()
    assert body["truncated"] is True
    assert len(body["text"]) == 20


# --------------------------------------------------------------------------- html
def test_search_page_renders_the_form(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<form class="search"' in response.text
    assert 'name="q"' in response.text


def test_search_page_renders_hits_and_marks(client: TestClient) -> None:
    response = client.get("/", params={"q": "fox"})
    assert response.status_code == 200
    assert "<mark>fox</mark>" in response.text.lower()
    assert "Alpha" in response.text


def test_search_page_paginates(client: TestClient) -> None:
    first = client.get("/", params={"q": "fox", "k": 1})
    assert "наступна →" in first.text
    assert "попередня" not in first.text
    second = client.get("/", params={"q": "fox", "k": 1, "page": 2})
    assert "попередня" in second.text


def test_search_page_keeps_the_form_on_a_bad_query(client: TestClient) -> None:
    response = client.get("/", params={"q": "fox AND ("})
    assert response.status_code == 200
    assert "bad query" in response.text


def test_search_page_contains_no_results_message(client: TestClient) -> None:
    response = client.get("/", params={"q": "zeppelin"})
    assert "Нічого не знайдено" in response.text


# --------------------------------------------------------------------------- docs
def test_swagger_moved_to_api_docs(client: TestClient) -> None:
    """``/docs`` belongs to the document view, so Swagger had to move."""
    response = client.get("/api/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_docs_route_is_not_the_schema_browser(client: TestClient) -> None:
    response = client.get("/docs/0")
    assert response.status_code == 200
    assert "swagger-ui" not in response.text.lower()


def test_openapi_schema(client: TestClient) -> None:
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    assert "/search" in paths
    assert "/search/async" in paths
    assert "/stats" in paths
    assert "/health" in paths
    assert paths["/health"]["get"]["responses"].keys() >= {"200"}
    assert "/" not in paths  # the HTML page is not part of the API contract


# --------------------------------------------------------------------------- plumbing
def test_dependency_overrides_swap_the_index(index: Index, tmp_path: Path) -> None:
    """No index file at all; the route never learns the difference."""
    app = create_app(Settings(index_path=tmp_path / "does-not-exist.pkl"))
    app.dependency_overrides[get_index] = lambda: index
    with TestClient(app) as test_client:
        body = test_client.get("/search", params={"q": "fox"}).json()
        assert body["total"] == 3
        # ... while /health still reports the truth about startup
        assert test_client.get("/health").status_code == 503


def test_request_id_header(client: TestClient) -> None:
    response = client.get("/health")
    assert len(response.headers["x-request-id"]) == 12


def test_unhandled_error_is_a_clean_500(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "index.pkl"
    save_pickle(index, path)
    app = create_app(Settings(index_path=path))

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("kaboom")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "internal server error"
    assert body["request_id"] == response.headers["x-request-id"]
    assert "kaboom" not in response.text  # the traceback stays in the log


def test_static_css_is_served(client: TestClient) -> None:
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "--mark" in response.text


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_PATH", "somewhere/index.pkl")
    monkeypatch.setenv("MAX_K", "7")
    monkeypatch.setenv("PAGE_SIZE", "3")
    monkeypatch.setenv("PORT", "9001")
    settings = Settings()
    assert settings.index_path == Path("somewhere/index.pkl")
    assert settings.max_k == 7
    assert settings.page_size == 3
    assert settings.port == 9001
    assert json.dumps(settings.model_dump(mode="json"))  # serialisable for logs
