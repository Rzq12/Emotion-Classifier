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
| API | FastAPI (Fase 4) |
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

## Testing & Linting

```bash
pytest -q
ruff check src tests
```

## Dataset

Emotion classification 3-kelas (`anger`, `happiness`, `sadness`) dari review
aplikasi Indonesia. Dataset imbalanced (kelas `anger` minoritas); penanganannya
memakai class weighting dan evaluasi berbasis F1-macro.
