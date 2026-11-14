"""Lab 7 - the Pydantic models, and *only* at the boundary.

Nothing in ``findex.index``, ``findex.ranking`` or ``findex.snippet`` imports
this module. The internal types stay ``@dataclass(frozen=True, slots=True)``:
they are created millions of times per index build, they never see untrusted
input, and a Rust-validated model per posting would be pure overhead. Pydantic
belongs where data crosses a trust boundary - an HTTP request or response - and
that is exactly here:

    route parameter  -> FastAPI reads the type hint -> 422 on a bad value
    SearchResult     -> HitModel                     -> JSON on the way out

``SearchResponse`` is also the OpenAPI schema: ``response_model`` makes FastAPI
validate what the handler returned before serialising it, so a field renamed
in a refactor fails the tests instead of the clients.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: The scorers the CLI already had; ``Literal`` turns the set into a 422 and
#: into an enum in ``/api/docs`` without a line of validation code.
ScorerName = Literal["bm25", "tfidf"]


class HitModel(BaseModel):
    """One ranked hit, as the API returns it."""

    doc_id: int = Field(description="Stable id: the position in the sorted corpus.")
    title: str
    score: float = Field(description="Scorer output; comparable only within one query.")
    snippet: str = Field(
        default="",
        description="Context window with the query terms marked as [term].",
    )


class SearchResponse(BaseModel):
    """A page of ranked results plus everything a client needs to ask for more."""

    query: str
    total: int = Field(description="Documents matching the query, before pagination.")
    page: int
    pages: int = Field(description="Total pages at this `k`; at least 1.")
    k: int = Field(description="Hits per page.")
    scorer: ScorerName
    took_ms: float = Field(description="Parse + match + score + snippet, milliseconds.")
    results: list[HitModel]


class DocModel(BaseModel):
    """One document, with its stored metadata and (a prefix of) its text."""

    doc_id: int
    title: str
    path: str
    n_tokens: int
    text: str
    truncated: bool = Field(
        default=False, description="True when the text was cut at doc_max_chars."
    )


class StatsModel(BaseModel):
    """Index size, for humans and for a dashboard."""

    documents: int
    terms: int = Field(description="Vocabulary size: distinct indexed terms.")
    postings: int = Field(description="Total (term, document) pairs.")
    tokens: int
    avg_doc_length: float
    uptime_s: float = Field(description="Seconds since the lifespan started.")


class HealthModel(BaseModel):
    """What a platform's health check reads. 200 when loaded, 503 when not."""

    status: Literal["ok", "index-not-loaded"]
    index_loaded: bool
    documents: int
    uptime_s: float
    detail: str = ""


__all__ = [
    "DocModel",
    "HealthModel",
    "HitModel",
    "ScorerName",
    "SearchResponse",
    "StatsModel",
]
