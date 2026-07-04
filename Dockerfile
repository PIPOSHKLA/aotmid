# =============================================================================
# Multi-stage Docker build for AOT Stock Network thesis application
# =============================================================================

# ── Stage 1: Build environment ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="AOT Stock Network"
LABEL org.opencontainers.image.description="Social Network Analysis of Factors Influencing AOT Stock Price"
LABEL org.opencontainers.image.version="0.1.0"

WORKDIR /app

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code
COPY pyproject.toml .
COPY cli/ cli/
COPY src/ src/
COPY Home.py .
COPY .env.example .env

RUN pip install -e ".[ml]" --no-cache-dir --no-deps

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
