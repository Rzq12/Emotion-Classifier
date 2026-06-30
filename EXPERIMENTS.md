# EXPERIMENTS.md — Catatan Keputusan & Eksperimen

Dokumen ini merekam keputusan data/modeling dan trade-off untuk keperluan
storytelling portofolio. Diisi bertahap tiap fase.

---

## Fase 1 — Keputusan Data

### Reframe: Sentiment → Emotion (keputusan kunci)

Dokumen perencanaan awal (PRD/SRS) memframe tugas utama sebagai **sentiment 3-kelas**
(`positive/negative/neutral`), dengan emotion sebagai stretch goal. Namun dataset
nyata yang tersedia adalah **emotion classification tanpa kelas netral**:

- Label: `anger`, `happiness`, `sadness` (3 kelas).
- Nama file test secara eksplisit menyebut "tanpa netral".

**Keputusan:** mengikuti data nyata — tugas utama menjadi **emotion 3-kelas**.
Disetujui pemilik proyek. Lebih jujur terhadap data dan lebih menarik secara
teknis (emosi lebih nuanced daripada sentiment biner/terner berbasis rating).

### Sumber & bentuk data mentah

| File | Baris | Kolom | Casing label |
|---|---|---|---|
| `Dataset/train paling fix.csv` | 2.189 | `Unnamed: 0`, `review_text`, `majority_vote` | UPPERCASE |
| `Dataset/23-24test_emotion_data tanpa netral kurang 500.csv` | 550 | `review_text`, `majority_vote` | lowercase |

Temuan EDA awal:
- Tidak ada null / duplikat di dalam masing-masing file.
- Casing label **berbeda** antar file → dinormalisasi ke lowercase kanonik.
- Kolom indeks `Unnamed: 0` hanya ada di train → di-drop.
- Panjang teks: rata-rata ~90 karakter, maksimum 500.
- Domain dominan Halodoc (kesehatan), sebagian Gojek.

### Imbalance kelas

Distribusi train mentah: HAPPINESS 1404 / SADNESS 571 / ANGER 214. Kelas `anger`
adalah minoritas tajam (~10%). Implikasi untuk Fase 2: wajib pakai **F1-macro**
(bukan accuracy) sebagai metrik utama, dan eksperimen **class weighting** /
augmentasi untuk kelas minoritas.

### Pipeline preprocessing

Diimplementasikan murni dengan stdlib + regex (tanpa dependency `emoji`) agar
ringan & reproducible. Langkah (`src/data/preprocessing.py`):

1. Lowercase.
2. Hapus URL & mention (`@user`).
3. Hapus emoji (regex rentang Unicode).
4. Reduksi karakter berulang (`baguuusss` → `baguuss`, maks 2).
5. Buang simbol non-teks (sisakan huruf, angka, `.,!?`).
6. Normalisasi slang/singkatan Indonesia per-token (`src/data/slang_dict.py`).
7. Normalisasi whitespace.

Label dinormalisasi & divalidasi di `src/data/labels.py` — label tak dikenal
(mis. `netral`) sengaja melempar error agar data buruk gagal keras, bukan diam-diam.

### Hasil split (stratified, val_size=0.15, seed=42)

| Split | Jumlah | happiness | sadness | anger |
|---|---|---|---|---|
| train | 1.798 | 1132 | 485 | 181 |
| val   | 318   | 200  | 86  | 32  |
| test  | 548   | 321  | 126 | 101 |

Penyusutan dari 2.189 → 2.116 (train+val) akibat dedup teks identik setelah
cleaning dan pembuangan teks terlalu pendek (`min_chars=3`). Test 550 → 548.

### Catatan reproducibility

- Semua parameter di `configs/data.yaml` (bukan hardcode).
- Seed split tetap (42). Output deterministik.
- Langkah berikutnya: `dvc add` data/processed + push DagsHub (tertunda; `dvc`
  belum terpasang di environment saat ini).

---

## Fase 2 — Training & Tracking

### Infrastruktur

- **Experiment tracking:** MLflow dengan backend **SQLite** (`sqlite:///mlflow.db`).
  File store dipensiunkan di MLflow 3.x dan model registry butuh backend DB,
  jadi SQLite dipilih (lokal, ringan, mendukung registry untuk "production candidate").
- **Metrik utama:** **F1-macro** (bukan accuracy) karena imbalance tajam. Dicatat
  juga per-kelas F1 + confusion matrix (CSV + heatmap PNG) sebagai artifact.
- Modul: `src/training/{dataset,metrics,baseline,train_indobert}.py`,
  `src/tracking/{mlflow_utils,registry}.py`. Hyperparameter di `configs/training.yaml`.

### Baseline — TF-IDF + Logistic Regression (wajib)

Konfigurasi: ngram (1,2), min_df=2, max_features=20k, sublinear_tf, `class_weight="balanced"`.

| Metrik (test) | Nilai |
|---|---|
| Accuracy | 0.836 |
| **F1-macro** | **0.752** |
| F1 anger | 0.621 |
| F1 happiness | 0.946 |
| F1 sadness | 0.689 |

Observasi: kelas mayoritas `happiness` sangat mudah (F1 0.95); `anger` (minoritas)
adalah yang tersulit (F1 0.62) meski sudah `class_weight="balanced"`. Ini menetapkan
**ambang batas** yang harus dilampaui IndoBERT pada F1-macro.

### IndoBERT fine-tuning

- Model: `indobenchmark/indobert-base-p1`, weighted cross-entropy (inverse-frequency)
  untuk imbalance, max_length=128, 3 epoch, lr=2e-5.
- Pipeline tervalidasi end-to-end via smoke test (64 sampel, 1 epoch): model
  ter-download, training/eval/predict/log/eksport berjalan benar.
- **Catatan lingkungan:** environment hanya CPU (`torch 2.2.0+cpu`, CUDA tidak
  tersedia). Full training ~6.4 s/step → ~40 menit untuk 3 epoch. Dijalankan
  terpisah/background; hasil akhir & perbandingan vs baseline diisi setelah selesai.
- Eksperimen variasi (lr, max_length, class weighting on/off, augmentasi) dapat
  dijalankan dengan mengubah `configs/training.yaml` — semua otomatis ter-log ke MLflow.
- Pemilihan & registrasi model terbaik: `python -m src.tracking.registry --register`
  (memilih run dengan `test_f1_macro` tertinggi, tag `production-candidate`).

---

## Fase 3 — RAG & LLM Layer

### Komponen

- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (multilingual, CPU-friendly, mendukung Bahasa Indonesia), embedding ternormalisasi.
- **Vector store:** ChromaDB persistent (`chroma_db/`), cosine space, 1 vektor per
  review + metadata `{emotion, split}`. Total **2.664 review** ter-index.
- **LLM provider:** wrapper abstrak `LLMClient` dengan implementasi **Groq**,
  **Gemini** (google-genai), dan Ollama. Provider dipilih via `LLM_PROVIDER`;
  error provider dinormalisasi ke `LLMError` agar handler API seragam.
- **Insight generator:** retrieve review negatif (anger/sadness) → prompt JSON
  terstruktur → parse defensif → cache TTL (kontrol biaya).
- **Chatbot:** retrieve top-k by similarity → prompt anti-halusinasi → jawaban +
  `sources` (review_id). Hits di bawah `min_score` dianggap tidak relevan →
  balas eksplisit "tidak menemukan review relevan".

### Validasi retrieval (tanpa LLM)

Query "aplikasi error saat pembayaran" (filter anger/sadness) → top hit:
- 0.806 sadness "error saat transaksi pembayaran"
- 0.710 sadness "aplikasi sy mengalami error ... pembayaran"

Query terbuka "dokternya ramah dan membantu" → top hit 0.953 happiness
"dokter nya cekatan, sangat membantu". Relevansi retrieval baik.

### Catatan

- Dataset processed hanya punya `text,label`; metadata `date/app_source` (ada di
  skema SYSTEM_DESIGN) tidak tersedia, jadi metadata vektor dibatasi ke
  `emotion` + `split`. `review_id` di-generate `{split}_{index}`.
- Generasi insight/chat penuh butuh API key (Groq/Gemini) di `.env` — belum
  dijalankan live; logika diverifikasi via unit test (fake LLM/store).
- Prompt disimpan sebagai file di `src/llm/prompts/` (bukan inline), sesuai CLAUDE.md.
