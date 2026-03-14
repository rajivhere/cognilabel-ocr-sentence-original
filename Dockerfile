# syntax=docker/dockerfile:1
ARG PY_BASE=python:3.12.7-slim
FROM ${PY_BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=1 \
    PYTHONPATH=/app

# System deps for numpy/opencv/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- deps first (use layer cache) ----
# If you actually use requirements.s3.txt, COPY it too; otherwise remove the conditional below.
COPY requirements.train.txt ./
# COPY requirements.s3.txt ./
# COPY pyproject.toml ./

# Install TF first (to match Python 3.12.7), then the rest
RUN python -m pip install --upgrade pip && \
    pip install --upgrade "tensorflow==2.18.0"

# RUN pip install --upgrade-strategy only-if-needed -r requirements.train.txt


# Install your other deps, excluding tensorflow (and keras, see note)
RUN grep -viE '^(tensorflow|keras)(==|>=|=|$)' requirements.train.txt > req_nontf.txt && \
    pip install --upgrade-strategy only-if-needed -r req_nontf.txt && \
    if [ -f requirements.s3.txt ] && [ -s requirements.s3.txt ]; then \
      pip install --upgrade-strategy only-if-needed -r requirements.s3.txt ; \
    fi 

# ---- training entry files (only the ones we need) ----
RUN mkdir -p /app/Ocr-sentence-train
COPY Ocr-sentence-train/model.py                 /app/Ocr-sentence-train/model.py
COPY Ocr-sentence-train/configs.py               /app/Ocr-sentence-train/configs.py
COPY Ocr-sentence-train/train_ocr_cglml_exec.py  /app/Ocr-sentence-train/train_ocr_cglml_exec.py
COPY Ocr-sentence-train/last_state_writer.py     /app/Ocr-sentence-train/last_state_writer.py

WORKDIR /app/Ocr-sentence-train
ENTRYPOINT ["python", "-u", "/app/Ocr-sentence-train/train_ocr_cglml_exec.py"]
