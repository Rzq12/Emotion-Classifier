# Indo Review Intelligence

Sistem hybrid yang menggabungkan **emotion classifier** (fine-tuned IndoBERT) dengan
**RAG + LLM** untuk menghasilkan insight otomatis dari review aplikasi Indonesia
(domain: Halodoc, Gojek, dsb). Proyek portofolio AI/ML Engineer yang menunjukkan
full ML lifecycle: data → training → experiment tracking → LLM layer → deployment → monitoring.

> **Catatan skema:** tugas klasifikasi utama adalah **emotion 3-kelas**
> (`anger`, `happiness`, `sadness`) — **tanpa kelas netral** — mengikuti dataset
> nyata yang tersedia. Lihat `EXPERIMENTS.md` untuk alasan keputusan ini.

## Status

| Fase | Status |
|---|---|
| Fase 0 — Setup project | ✅ Selesai |
| Fase 1 — Data & preprocessing | ✅ Selesai |
| Fase 2 — Training & tracking | ⏳ Belum |
| Fase 3 — RAG & LLM | ⏳ Belum |
| Fase 4 — API & serving | ⏳ Belum |
| Fase 5 — Deployment | ⏳ Belum |
| Fase 6 — Monitoring & polish | ⏳ Belum |

Roadmap detail: `PLAN.md`. Breakdown task: `TASK_BREAKDOWN.md`.

## Struktur Repo

```
.
├── configs/            # konfigurasi pipeline (data.yaml, dst.)
├── data/
│   ├── raw/            # data mentah (DVC-tracked, tidak di git)
│   └── processed/      # output pipeline (train/val/test.csv)
├── Dataset/            # sumber data emotion asli (CSV)
├── src/
│   ├── data/           # cleaning, normalisasi label, build dataset
│   ├── training/       # (Fase 2) baseline & fine-tuning
│   ├── tracking/       # (Fase 2) helper MLflow
│   ├── rag/            # (Fase 3) embedding & vector store
│   ├── llm/            # (Fase 3) LLM client & prompt
│   ├── api/            # (Fase 4) FastAPI app
│   └── monitoring/     # (Fase 6) logging & drift
├── tests/              # unit test
└── web/                # (Fase 5) frontend React
```

## Setup

Proyek menggunakan conda environment `trading` (Python 3.11).

```bash
conda activate trading
pip install -r requirements.txt
```

Salin environment example:

```bash
cp .env.example .env          # backend (LLM, vector store, API)
cp web/.env.example web/.env  # frontend
```

## Menjalankan Pipeline Data (Fase 1)

Membersihkan teks, menyatukan label, dedup, dan split train/val (test sudah terpisah):

```bash
python -m src.data.prepare_dataset --config configs/data.yaml
```

Output ditulis ke `data/processed/{train,val,test}.csv` dengan kolom `text,label`.

## Testing & Linting

```bash
pytest -q
ruff check src tests
```

## Dataset

| Split | Jumlah | Distribusi |
|---|---|---|
| train | 1.798 | happiness 1132 / sadness 485 / anger 181 |
| val   | 318   | happiness 200 / sadness 86 / anger 32 |
| test  | 548   | happiness 321 / sadness 126 / anger 101 |

Dataset sangat imbalanced (kelas `anger` minoritas) — strategi penanganannya
dibahas di Fase 2 (`PLAN.md`) dan dicatat di `EXPERIMENTS.md`.
