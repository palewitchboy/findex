# findex - lab 7 image.
#
#   docker build -t findex .
#   docker run --rm -e INDEX_PATH=/data/index.pkl -p 8000:8000 findex
#
# Multi-stage, in the shape uv's own Docker guide recommends:
#   * builder  - resolves and installs from `uv.lock`, byte-compiles, keeps the
#                compiler and the cache out of the runtime image;
#   * runtime  - a slim Python with the finished virtualenv, a non-root user,
#                the index mounted (or baked) at /data, uvicorn as PID 1.
#
# NOT BUILT ON THIS MACHINE: Docker is not installed here, so this file is
# reviewed, not executed. See the README's Lab 7 section for the exact gap.

# syntax=docker/dockerfile:1

# --------------------------------------------------------------------------- builder
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml or uv.lock
# changes, so editing a module does not re-install 30 wheels.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Then the code, and the project itself, in a second layer.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --------------------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

# A non-root user owns nothing but the app and the data directory. Running as
# root inside a container is a privilege the process never needs.
RUN groupadd --system --gid 1001 findex \
    && useradd --system --uid 1001 --gid findex --create-home --home-dir /app findex \
    && mkdir -p /data \
    && chown findex:findex /data

WORKDIR /app

COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INDEX_PATH=/data/index.pkl \
    LOG_LEVEL=INFO \
    PORT=8000

# The index is data, not code: mount it at /data, or bake it in at build time
# for a self-contained demo image (drop `bench/_tmp` from .dockerignore first):
#   COPY --chown=findex:findex bench/_tmp/lab7/index.pkl /data/index.pkl

USER findex

EXPOSE 8000

# A health check the orchestrator can read without curl (slim has no curl).
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"]

# `--workers 1` by default: one container, one process, scale with replicas.
# Raise it only if the platform gives the container several cores.
CMD ["uvicorn", "findex.web:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
