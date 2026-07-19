---
title: Indo Review Intelligence
emoji: 😊
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

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

| Layer         | Tools                                        |
| ------------- | -------------------------------------------- |
| Preprocessing | pandas, scikit-learn                         |
| Model         | HuggingFace Transformers (IndoBERT), PyTorch |
| Tracking      | MLflow (SQLite backend)                      |
| Vector store  | ChromaDB                                     |
| Embedding     | sentence-transformers (multilingual MiniLM)  |
| LLM           | Groq / Gemini / Ollama                       |
| API           | FastAPI                                      |
| Frontend      | React + Vite (Fase 5)                        |

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
│   └── monitoring/     # logging prediksi + drift check (PSI)
├── reports/            # contoh drift report
├── tests/              # unit test
└── web/                # frontend React (Vite + Tailwind)
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
python -m src.rag.build_index --config configs/rag.yaml --reset  # embed review ke ChromaDB
python -m src.rag.evaluate_retrieval                             # eval kualitas retrieval
```

Retrieval bersifat **hybrid**: vector search (ChromaDB + multilingual MiniLM)
digabung BM25 lexical via Reciprocal Rank Fusion — kata kunci eksak ("dana",
"voucher") tetap tertangkap meski embedding meleset. Query dinormalisasi dengan
preprocessing yang sama dengan korpus (slang Indo, emoji). Gerbang relevansi
chat (`min_score` 0.55) dikalibrasi dari golden set
(`configs/retrieval_eval.yaml`): hit@1 1.0, dan query di luar domain ditolak.

Insight & chatbot diakses lewat `InsightGenerator` / `ChatResponder`
(`src/rag/`) — keduanya di-cache (TTL) dan jawaban chat menyertakan hanya
review ID yang benar-benar dikutip. Chat mendukung riwayat percakapan
(multi-turn) lewat field `history`. Pilih provider LLM via `LLM_PROVIDER` di
`.env`. Retrieval berjalan tanpa API key; generasi insight/chat memerlukan key
LLM.

### 4. API (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 7860
```

Endpoint:

| Method | Path        | Fungsi                                                          |
| ------ | ----------- | --------------------------------------------------------------- |
| GET    | `/health`   | Status model, vector DB, LLM provider                           |
| GET    | `/stats`    | Statistik dataset (distribusi emosi, per-split) untuk dashboard |
| POST   | `/classify` | `{text}` → `{label, confidence}`                                |
| POST   | `/insight`  | `{query}` → ringkasan terstruktur (rate-limited)                |
| POST   | `/chat`     | `{question}` → `{answer, sources}` (rate-limited)               |

Docs interaktif tersedia di `/docs`. CORS dan rate limit dikonfigurasi via env
(`CORS_ALLOW_ORIGINS`, `RATE_LIMIT_PER_MINUTE`). Build image: `docker build -t indo-review-api .`

## Frontend (web/)

Aplikasi React + Vite dengan 4 tab: Dashboard (distribusi emosi), Coba Klasifikasi,
Insight, dan Tanya Data (chat).

```bash
cd web
npm install
npm run dev      # http://localhost:5173
npm run build    # output ke web/dist
```

Set `VITE_API_BASE_URL` (lihat `web/.env.example`) ke URL backend.

## Monitoring

Setiap prediksi `/classify` otomatis di-log (timestamp, teks, label, confidence)
ke JSONL — path via env `PREDICTION_LOG_PATH` (default
`data/monitoring/predictions.jsonl`), ditulis non-blocking setelah response.

Drift check dijalankan manual:

```bash
python -m src.monitoring.check_drift              # log prediksi vs train.csv
python -m src.monitoring.check_drift --simulate   # demo: sample sengaja drifted
```

Metrik: **PSI** distribusi label & panjang teks (ambang 0.1 moderat / 0.2 drift)
plus porsi prediksi low-confidence. Report markdown ditulis ke
`reports/drift_report.md` (contoh hasil simulasi ter-commit di repo).

## Testing & Linting

```bash
pytest -q
ruff check src tests
cd web && npm run build
```

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

### 4. CI & otomasi HF Spaces

- `.github/workflows/ci.yml` — lint + test backend dan build frontend di setiap
  push/PR ke `main`.
- `.github/workflows/sync-to-hf.yml` — auto force-push `main` ke repo Space
  setiap push ke `main` (bisa juga manual via Actions tab).
- `.github/workflows/keep-alive.yml` — ping Space tiap 6 jam agar tidak sleep
  (free tier HF Spaces tidur setelah ~48 jam tanpa traffic).

Secret yang harus di-set di GitHub (**Settings → Secrets and variables →
Actions**):

| Secret          | Isi                                                                                |
| --------------- | ---------------------------------------------------------------------------------- |
| `HF_TOKEN`      | Token HF dengan akses write (https://huggingface.co/settings/tokens)               |
| `HF_SPACE_REPO` | Repo Space, mis. `riezqidr/indo-review-intelligence`                               |
| `HF_SPACE_URL`  | URL health Space, mis. `https://riezqidr-indo-review-intelligence.hf.space/health` |

> Catatan: workflow `schedule` (keep-alive) hanya berjalan dari branch default
> (`main`) — pastikan workflow sudah ter-merge ke `main` di GitHub.

Frontmatter HF Spaces (`sdk: docker`, `app_port: 7860`) sudah ada di bagian atas
README ini, jadi Space langsung terkonfigurasi saat sync.

> Uji lokal image: `docker build -t indo-review-api . && docker run -p 7860:7860 -e MODEL_DIR=<username>/indo-emotion-indobert -e GROQ_API_KEY=... indo-review-api`

## Dataset

Emotion classification 3-kelas (`anger`, `happiness`, `sadness`) dari review
aplikasi Indonesia. Dataset imbalanced (kelas `anger` minoritas); penanganannya
memakai class weighting dan evaluasi berbasis F1-macro.
