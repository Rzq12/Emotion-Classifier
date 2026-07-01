# Dockerfile untuk API FastAPI (target: Hugging Face Spaces, Docker SDK).
# HF Spaces mengharuskan listen di port 7860 dan berjalan sebagai non-root user.

FROM python:3.11-slim

# CPU-only torch + lib dasar.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PORT=7860

# Dependency sistem minimal (build wheels tertentu).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (wajib di HF Spaces).
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install dependency (torch CPU dari index resmi agar image lebih kecil).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Kode aplikasi & konfigurasi.
COPY src ./src
COPY configs ./configs

# Artifact model & vector store di-mount/di-pull saat runtime (bukan di-bake ke
# image, sesuai kebijakan: model besar lewat HF Hub / volume).
#   - MODEL_DIR  -> direktori model IndoBERT (default artifacts/indobert)
#   - chroma_db/ -> hasil `python -m src.rag.build_index`

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
