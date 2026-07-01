# Indo Review Intelligence

Sistem hybrid yang menggabungkan **emotion classifier** (fine-tuned IndoBERT)
dengan **RAG + LLM** untuk menghasilkan insight otomatis dari review aplikasi
Indonesia (domain: Halodoc, Gojek, dsb).

Klasifikasi emotion 3-kelas — `anger`, `happiness`, `sadness` (tanpa kelas netral) —
dipadukan dengan retrieval-augmented generation untuk merangkum keluhan pengguna
dan menjawab pertanyaan tim produk berbasis data review nyata.

## Fitur

- **Classifier:** klasifikasi emosi review (IndoBERT fine-tuned) dengan baseline
  TF-IDF + Logistic Regression sebagai pembanding.
- **Insight generator:** ringkasan terstruktur (tema keluhan, kutipan, rekomendasi)
  dari review beremosi negatif, di-grounding ke data dan di-cache.
- **Chatbot internal:** menjawab pertanyaan bebas dengan grounding ke review
  ter-retrieve (anti-halusinasi) dan menyertakan sumber.
- **Experiment tracking:** MLflow (metrik utama F1-macro, confusion matrix, registry).
- **LLM provider-agnostic:** Groq / Gemini / Ollama lewat satu interface.

## Tech Stack

| Layer | Tools |
|---|---|
| Preprocessing | pandas, scikit-learn |
| Model | HuggingFace Transformers (IndoBERT), PyTorch |
| Tracking | MLflow (SQLite backend) |
| Vector store | ChromaDB |
| Embedding | sentence-transformers (multilingual MiniLM) |
| LLM | Groq / Gemini / Ollama |
| API | FastAPI |
| Frontend | React + Vite (Fase 5) |

## Struktur Repo

```
.
├── configs/            # konfigurasi pipeline (data.yaml, training.yaml, rag.yaml)
├── data/processed/     # output pipeline data (train/val/test.csv)
├── Dataset/            # sumber data emotion (CSV)
├── src/
│   ├── data/           # cleaning, normalisasi label, build dataset
│   ├── training/       # baseline & fine-tuning IndoBERT, metrik
│   ├── tracking/       # helper MLflow + model registry
│   ├── rag/            # embedding, vector store, insight & chat
│   ├── llm/            # LLM client (Groq/Gemini/Ollama) + prompt
│   ├── api/            # FastAPI app (Fase 4)
│   └── monitoring/     # logging & drift (Fase 6)
├── tests/              # unit test
└── web/                # frontend React (Fase 5)
```

## Setup

Proyek menggunakan Python 3.11 (contoh: conda env `trading`).

```bash
pip install -r requirements.txt
cp .env.example .env          # isi API key LLM (Groq/Gemini)
cp web/.env.example web/.env
```

## Penggunaan

### 1. Pipeline data

```bash
python -m src.data.prepare_dataset --config configs/data.yaml
```

Membersihkan teks (normalisasi slang Indo, emoji), menyatukan label, dedup, dan
split train/val. Output: `data/processed/{train,val,test}.csv` (kolom `text,label`).

### 2. Training & tracking

```bash
python -m src.training.baseline --config configs/training.yaml        # TF-IDF + LogReg
python -m src.training.train_indobert --config configs/training.yaml  # IndoBERT
python -m src.tracking.registry --register                            # daftarkan model terbaik
mlflow ui --backend-store-uri sqlite:///mlflow.db                     # dashboard eksperimen
```

Metrik utama F1-macro (dataset imbalanced). Tambahkan `--max-train-samples`/`--epochs`
untuk smoke test cepat.

### 3. RAG (insight & chat)

```bash
python -m src.rag.build_index --config configs/rag.yaml   # embed review ke ChromaDB
```

Insight & chatbot diakses lewat `InsightGenerator` / `ChatResponder`
(`src/rag/`). Pilih provider LLM via `LLM_PROVIDER` di `.env`. Retrieval berjalan
tanpa API key; generasi insight/chat memerlukan key LLM.

### 4. API (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 7860
```

Endpoint:

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | Status model, vector DB, LLM provider |
| POST | `/classify` | `{text}` → `{label, confidence}` |
| POST | `/insight` | `{query}` → ringkasan terstruktur (rate-limited) |
| POST | `/chat` | `{question}` → `{answer, sources}` (rate-limited) |

Docs interaktif tersedia di `/docs`. CORS dan rate limit dikonfigurasi via env
(`CORS_ALLOW_ORIGINS`, `RATE_LIMIT_PER_MINUTE`). Build image: `docker build -t indo-review-api .`

## Testing & Linting

```bash
pytest -q
ruff check src tests
```

<<<<<<< Updated upstream
=======
## Deployment

Arsitektur: **API** di Hugging Face Spaces (Docker), **web** di Vercel. Model
classifier di-pull dari HF Hub; index ChromaDB di-bake ke image saat build dari
`Dataset/` (cold start cepat).

### 1. Publikasikan model ke HF Hub

```bash
huggingface-cli login
python -m scripts.push_model_to_hub --repo <username>/indo-emotion-indobert
```

### 2. API → Hugging Face Spaces (Docker SDK)

1. Buat Space baru → SDK **Docker**.
2. Push isi repo ini ke Space (atau hubungkan dari GitHub). `Dockerfile` di root
   otomatis dipakai; index ChromaDB dibuild saat image build.
3. Tambahkan frontmatter di README Space (atau set di UI): `app_port: 7860`.
4. Isi **Settings → Variables & secrets**:
   - `MODEL_DIR` = `<username>/indo-emotion-indobert`
   - `LLM_PROVIDER` = `groq` (atau `gemini`)
   - `GROQ_API_KEY` / `GEMINI_API_KEY`
   - `CORS_ALLOW_ORIGINS` = URL Vercel (mis. `https://namaapp.vercel.app`)
5. Tunggu build selesai, verifikasi `https://<space>.hf.space/health`.

### 3. Web → Vercel

1. Import repo, set **Root Directory** = `web/` (`web/vercel.json` sudah ada).
2. Env var: `VITE_API_BASE_URL` = URL Space (mis. `https://<space>.hf.space`).
3. Deploy, lalu smoke test: Dashboard → Classify → Insight → Chat.

### 4. CI

`.github/workflows/ci.yml` menjalankan lint + test backend dan build frontend
di setiap push/PR. Untuk auto-deploy ke Space, tambahkan remote git Space +
`HF_TOKEN` secret di GitHub (opsional).

> Uji lokal image: `docker build -t indo-review-api . && docker run -p 7860:7860 -e MODEL_DIR=<username>/indo-emotion-indobert -e GROQ_API_KEY=... indo-review-api`

>>>>>>> Stashed changes
## Dataset

Emotion classification 3-kelas (`anger`, `happiness`, `sadness`) dari review
aplikasi Indonesia. Dataset imbalanced (kelas `anger` minoritas); penanganannya
memakai class weighting dan evaluasi berbasis F1-macro.
