# ── Stage 1: builder ────────────────────────────────────────────────────────
# Installs all Python dependencies into an isolated prefix.
# This stage is only used at build time and is discarded from the final image.
FROM python:3.12-slim AS builder

WORKDIR /install

# System libraries required by OpenCV and MediaPipe at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST so this expensive layer is cached.
# Re-running `pip install` only happens when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# ── Stage 2: final ──────────────────────────────────────────────────────────
# Lean runtime image — no build tools, no pip cache, just what's needed to run.
FROM python:3.12-slim AS final

WORKDIR /app

# Same runtime libs in the final stage (needed by OpenCV/MediaPipe at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY app/ ./app/

# Document the port the app listens on.
# Hugging Face Spaces sets PORT=7860 automatically; docker run -p can remap it.
EXPOSE 7860

# Default environment variables — all overridable via -e or docker-compose
ENV LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    CV_THREAD_POOL_SIZE=2 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=7860

# Use $PORT so the same image works locally (8000) and on HF Spaces (7860)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
