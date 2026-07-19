# Dockerfile untuk API FastAPI (target: Hugging Face Spaces, Docker SDK).
# HF Spaces mengharuskan listen di port 7860 dan berjalan sebagai non-root user.

FROM python:3.11-slim

# CPU-only torch + lib dasar. HF_HOME di dalam /app supaya cache model yang
# terunduh saat build (root) tetap terbaca-tulis oleh appuser saat runtime —
# di luar itu, download model dari HF Hub saat runtime gagal permission.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
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
COPY Dataset ./Dataset

# Bake dataset processed + index vektor (ChromaDB) ke image saat build agar
# cold start cepat. Embedding model ikut ter-cache di layer ini.
RUN python -m src.data.prepare_dataset --config configs/data.yaml \
    && python -m src.rag.build_index --config configs/rag.yaml

# Model classifier: di-set lewat env MODEL_DIR.
#   - HF Hub repo id (mis. "username/indo-emotion-indobert") -> di-download saat
#     request pertama dan di-cache; ATAU
#   - path lokal jika model di-mount/di-commit ke repo Space.
ENV MODEL_DIR=artifacts/indobert

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
