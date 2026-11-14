"""Lab 7 - the web layer.

``uvicorn findex.web:app`` is the whole deployment story, so this package
exports one module-level ``app``. ``create_app(settings)`` exists for the tests
(and for a future second instance): same routes, different index file.

Importing this package touches the environment (``Settings``), not the disk -
the index is only opened when the lifespan runs, i.e. when the server actually
starts. That keeps ``import findex.web`` cheap for tooling such as pyright,
ruff and the OpenAPI exporter.
"""

from __future__ import annotations

from findex.web.app import create_app

#: The ASGI application: ``uvicorn findex.web:app``.
app = create_app()

__all__ = ["app", "create_app"]
