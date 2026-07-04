# Warm, offline image for the Phase 8 Streamlit UI (plan §4, ADR-0013).
#
# Ships everything the UI needs to start *warm*: the processed catalog, the
# leaderboard, the six UI rungs' artifacts + the Map projection, and the MiniLM
# weights baked in. `recommend_similar`/Compare/Map load straight from the baked
# artifacts (no model download); free-text queries encode with the pre-pulled
# MiniLM, so the container needs no network at runtime. The MPNet rung is the one
# exception — its weights download on first use (graceful: slower, needs network).
#
# Build (from the repo root, with a warm data/processed + artifacts present):
#   docker build -t course-recommender-lab .
# Run:
#   docker run --rm -p 8501:8501 course-recommender-lab
# Then open http://localhost:8501 . The image is CPU-only.

FROM python:3.11-slim AS base

# libgomp1: OpenMP runtime that torch + faiss-cpu wheels link against.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face / Streamlit caches live under a writable, predictable home so the
# baked MiniLM weights survive into the final image and the non-root user can
# read them. PIP_NO_CACHE_DIR keeps image layers lean.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# --- dependencies (own layer, cached unless deps change) ---------------------
# Install the CPU build of torch from PyTorch's index first so the heavy default
# (CUDA) wheel is never pulled, then the package extras. Copy only the files the
# editable install resolves against so this layer caches across source edits.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 \
    && pip install -e ".[ui,semantic]"

# --- pre-pull the default model so free-text queries work offline ------------
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# --- application code + warm data/artifacts (see .dockerignore) ---------------
COPY .streamlit/ ./.streamlit/
COPY app/ ./app/
COPY data/ ./data/
COPY results/ ./results/
COPY artifacts/ ./artifacts/

# Run as a non-root user; give it the HF cache and app tree.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app "$HF_HOME"
USER appuser

# $PORT lets PaaS hosts (Cloud Run, HF Spaces) inject their port; defaults to 8501.
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8501'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/_stcore/health').read()==b'ok' else 1)"

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
