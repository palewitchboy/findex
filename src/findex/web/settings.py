"""Lab 7 - configuration from the environment, validated once, at the edge.

The twelve-factor rule (12factor.net, rule III) says config lives in the
**environment**, not in code: the same image must run in development, in CI and
in production, and the only thing that differs is a handful of variables. This
module is that boundary. ``Settings`` is a ``BaseSettings`` subclass, so
pydantic reads ``INDEX_PATH``, ``LOG_LEVEL``, ``MAX_K``, ``PAGE_SIZE`` and
``PORT`` from the process environment (and from a ``.env`` file when one is
present), coerces each value to its annotated type and raises a precise
``ValidationError`` **before the server binds a socket** if one of them is
wrong.

Two consequences worth naming:

* ``Settings(index_path=...)`` is also a perfectly ordinary constructor - the
  tests build one by hand instead of mutating ``os.environ``, and
  ``create_app(settings)`` takes it as an argument. The environment is the
  *default source*, not the only one.
* Field names are the environment names, case-insensitively: ``max_k`` reads
  ``MAX_K``. No prefix, nothing to remember. ``extra="ignore"`` keeps the
  process environment - which holds hundreds of unrelated variables - from
  turning startup into a validation error.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything the service needs to know that is not code."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    #: The pickled (or JSON) index produced by ``findex index``. A missing file
    #: is **not** a startup crash: the app comes up, ``/health`` answers 503 and
    #: every search route answers 503 with a readable message. That is what a
    #: platform's health check is for - restarting a container that cannot serve
    #: is not helped by the container refusing to start at all.
    index_path: Path = Path("index.pkl")

    #: Root logging level for the ``findex`` logger and uvicorn's.
    log_level: str = "INFO"

    #: Hard ceiling on ``?k=``. The default keeps one request from asking for a
    #: million scored documents; ``le`` on the query parameter is a static
    #: 1000 (OpenAPI cannot read settings) and this value is enforced in the
    #: handler, so ``/api/openapi.json`` stays honest about the absolute limit.
    max_k: int = Field(default=100, ge=1)

    #: Hits per page on the HTML page (the JSON API takes ``k`` explicitly).
    page_size: int = Field(default=10, ge=1, le=100)

    #: Default port for ``findex serve`` and the container's ``EXPOSE``.
    port: int = Field(default=8000, ge=1, le=65535)

    #: How much of a document the ``/docs/{id}`` page renders. A 5 MB book
    #: turned into one HTML page is 5 MB of markup per request; the cut is
    #: visible in the page ("показано перші N символів"), never silent.
    doc_max_chars: int = Field(default=200_000, ge=1)

    #: How many normalised document texts the snippet reader keeps in RAM.
    #: Lab 3 sized that cache at 8 for one CLI query; a server answers thousands
    #: of queries over the same corpus, and re-reading a 5 MB book per hit costs
    #: ~217 ms (measured). 32 covers this 48-document demo (~35 MB of text on
    #: disk); lower it in a small container, raise it for a static corpus.
    text_cache: int = Field(default=32, ge=8)


__all__ = ["Settings"]
