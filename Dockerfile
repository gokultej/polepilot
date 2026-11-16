FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/data/ultralytics OMP_NUM_THREADS=2 MPLCONFIGDIR=/data/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home polepilot \
    && mkdir -p /data /app/training_data && chown -R polepilot:polepilot /data /app
COPY requirements-runtime.txt .
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-runtime.txt
COPY --chown=polepilot:polepilot . .
USER polepilot
EXPOSE 8000
# Exactly one process: video task polling and evaluation queue are process-local.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "app:app"]
