# Hosted image for the stress-detection app (Streamlit + MediaPipe + TF CPU).
#
# Build context = the repo root. The app's relative paths (../models, ../data)
# are why the layout under /app mirrors the repo: /app/code and /app/models.
#
#   docker build -t stress-app .
#   docker run --rm -p 8501:8501 --cpus=2 -m 4g \
#       -e SUPABASE_URL=... -e SUPABASE_ANON_KEY=... stress-app
#
# Local, no login, SQLite:  add  -e STRESS_STORAGE_BACKEND=sqlite
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    GLOG_minloglevel=2 \
    # Two users share the CPU: stop each TensorFlow/BLAS pool from grabbing
    # every core and fighting the other (and MediaPipe's own threads).
    OMP_NUM_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    STRESS_CAPTURE_MODE=browser \
    STRESS_STORAGE_BACKEND=supabase

# OpenCV/MediaPipe need these shared libraries even with no display.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached unless requirements.txt changes). Hosted image:
# CPU-only TensorFlow, and no test/evaluation-only packages.
COPY requirements.txt .
RUN sed -E 's/^tensorflow==/tensorflow-cpu==/' requirements.txt \
  | grep -v -E '^(pytest|statsmodels)' > requirements.hosted.txt \
 && pip install -r requirements.hosted.txt

RUN useradd -m -u 1000 appuser \
 && mkdir -p /app/data && chown appuser:appuser /app/data   # writable only for the SQLite fallback mode
COPY --chown=appuser:appuser code ./code
COPY --chown=appuser:appuser models ./models

# Fail the BUILD (not the demo) if the CNN weights are Git-LFS pointer stubs:
# a checkout without `git lfs pull` leaves ~130-byte text files named *.h5, and
# the app would silently run with no CNN branch.
RUN for f in models/daisee_mobilenet_stress.h5 models/daisee_efficientnet_stress.h5; do \
      [ "$(stat -c %s "$f")" -gt 1000000 ] || { echo "ERROR: $f looks like a Git LFS pointer, not the model (run git lfs pull)"; exit 1; }; \
    done

USER appuser
WORKDIR /app/code
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
