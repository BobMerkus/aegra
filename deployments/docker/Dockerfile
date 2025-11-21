FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create non-root user for runtime
RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder stage
# -----------------------------
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

# Copy the project into the intermediate image
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/
COPY aegra.json ./aegra.json
COPY auth.py ./auth.py
COPY graphs/ ./graphs/

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-editable

# -----------------------------
# Final, minimal runtime image
# -----------------------------
FROM base AS final

# Install only minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy runtime assets required by the app and alembic
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/
COPY aegra.json ./aegra.json
COPY auth.py ./auth.py
COPY graphs/ ./graphs/

ENV PATH="/app/.venv/bin:$PATH"

# Copy src to keep compatibility with current compose command using 'src.agent_server.main:app'
# (We can switch compose to 'agent_server.main:app' later and drop this for a smaller image.)
COPY src/ ./src/

EXPOSE 8000

# Run as non-root
USER app

# Leave default CMD empty so docker-compose command can run migrations then start uvicorn
# CMD ["/app/.venv/bin/uvicorn", "src.agent_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
