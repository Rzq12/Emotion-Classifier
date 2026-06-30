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

*(belum dikerjakan)*
