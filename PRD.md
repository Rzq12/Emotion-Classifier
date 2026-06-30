# PRD — Indo Review Intelligence
**Hybrid Classifier + RAG Insight Generator**

Versi: 1.1
Status: Draft
Pemilik: (isi nama kamu)

> **Update v1.1 (reframe):** Tugas klasifikasi utama diubah dari *sentiment*
> (positive/negative/neutral) menjadi **emotion 3-kelas** (`anger`/`happiness`/`sadness`),
> **tanpa kelas netral**, mengikuti dataset nyata yang tersedia. Lihat `EXPERIMENTS.md`.
> Referensi "sentiment" pada dokumen lama dibaca sebagai "emotion" kecuali disebut lain.

---

## 1. Latar Belakang & Masalah

Tim produk aplikasi (Halodoc/Gojek/Tokopedia-style) menerima ribuan review per minggu di Play Store/App Store. Saat ini tidak ada cara cepat untuk:

- Mengetahui distribusi sentimen/emosi review secara otomatis dan terukur.
- Memahami *alasan* di balik review negatif tanpa membaca satu per satu.
- Menjawab pertanyaan ad-hoc seperti "apa keluhan utama minggu ini?" tanpa menunggu analis manual.

Proyek ini membangun sistem end-to-end yang menggabungkan **klasifikasi ML klasik** (cepat, murah, terukur) dengan **LLM/RAG** (kaya konteks, bisa menjawab pertanyaan bebas) untuk menutup gap tersebut, sekaligus menjadi portofolio yang menunjukkan pemahaman *full ML lifecycle*.

## 2. Tujuan

1. Membangun classifier sentiment/emotion berbasis IndoBERT yang fine-tuned dan ter-track eksperimennya.
2. Membangun sistem RAG yang bisa men-generate insight ringkasan dari kumpulan review negatif.
3. Membangun chatbot internal sederhana untuk tim produk menjawab pertanyaan berbasis data review.
4. Men-deploy seluruh sistem secara publicly accessible (API di Hugging Face Spaces, web di React) sehingga bisa didemokan langsung ke recruiter.
5. Menunjukkan praktik MLOps dasar: experiment tracking, data versioning, CI/CD, monitoring/drift check.

## 3. Non-Tujuan (Out of Scope)

- Tidak membangun sistem scraping real-time yang jalan terus-menerus (cukup snapshot dataset).
- Tidak menargetkan akurasi state-of-the-art riset; cukup solid dan well-documented.
- Tidak membangun autentikasi multi-user/role-based access — cukup single internal-tool experience.
- Tidak menangani bahasa selain Indonesia (dan campuran Indo-Inggris ringan) pada fase awal.

## 4. Target Pengguna

- **Primary persona (demo):** Tim produk/CS yang ingin tahu insight review tanpa baca manual.
- **Actual audience:** Recruiter/hiring manager AI/ML Engineer — sistem harus *terlihat* production-grade dan storytelling-nya jelas di README/demo.

## 5. User Stories

| # | Sebagai | Saya ingin | Supaya |
|---|---------|-----------|--------|
| 1 | Tim produk | Upload/lihat distribusi sentimen review per periode | Tahu kondisi kepuasan user secara cepat |
| 2 | Tim produk | Lihat ringkasan otomatis "kenapa user marah/sedih minggu ini" | Tidak perlu baca ratusan review satu-satu |
| 3 | Tim produk | Bertanya bebas ("apa keluhan utama soal pembayaran?") ke chatbot | Mendapat jawaban kontekstual berbasis data nyata |
| 4 | Engineer (saya) | Melihat seluruh eksperimen training ter-log | Bisa membandingkan model & reproduce hasil |
| 5 | Engineer (saya) | Tahu kalau distribusi prediksi model berubah signifikan dari waktu ke waktu | Bisa mendeteksi drift sebelum jadi masalah |

## 6. Fitur & Requirement

### 6.1 Data & Training (Must Have)
- Dataset review (Halodoc existing + opsional scrape Gojek/Tokopedia untuk variasi) dengan label sentiment dan/atau emotion.
- Preprocessing pipeline (cleaning, normalisasi teks Indo, handling emoji/slang).
- Fine-tuning IndoBERT untuk klasifikasi **emotion 3-kelas** (`anger`/`happiness`/`sadness`, tanpa netral) sesuai dataset nyata.
- Dataset versioning dengan DVC, remote storage via DagsHub.

### 6.2 Experiment Tracking (Must Have)
- Semua eksperimen (variasi preprocessing, hyperparameter, augmentasi data) ter-log ke MLflow.
- Minimal metrik: accuracy, F1 (macro), confusion matrix, training time.
- Model registry: model terbaik ditandai sebagai "production candidate".

### 6.3 LLM/RAG Layer (Must Have)
- Vector DB (ChromaDB atau FAISS) menyimpan embedding review (terutama negatif) untuk retrieval.
- Insight generator: ambil kumpulan review negatif relevan → LLM susun ringkasan terstruktur (tema keluhan, contoh kutipan anonim, saran tindak lanjut).
- Chatbot internal: terima pertanyaan bebas → retrieve review relevan → LLM jawab dengan grounding ke data.
- LLM provider: support minimal satu dari Anthropic API / OpenAI API; opsional fallback ke model open-source kecil via Ollama untuk demo offline/hemat biaya.

### 6.4 Serving & API (Must Have)
- FastAPI dengan endpoint:
  - `POST /classify` — input teks review → output label sentiment/emotion + confidence.
  - `POST /insight` — input filter (periode/kategori) → output ringkasan RAG.
  - `POST /chat` — input pertanyaan bebas → output jawaban grounded (untuk fitur chatbot).
  - `GET /health` — health check.
- Response time wajar untuk demo (`/classify` < 1s, `/insight` & `/chat` bisa beberapa detik karena LLM call).

### 6.5 Web Frontend (Must Have)
- Dibangun dengan React.
- Halaman/komponen minimal:
  - Dashboard ringkasan sentimen (chart distribusi).
  - Panel insight otomatis (hasil `/insight`).
  - Chat interface sederhana untuk `/chat`.
  - Form/input untuk uji `/classify` langsung (demo interaktif).
- Terhubung ke API yang di-deploy di Hugging Face Spaces.

### 6.6 CI/CD (Should Have)
- GitHub Actions: jalankan test otomatis (unit test classifier wrapper, API contract test) di setiap PR.
- Build Docker image otomatis dan push ke registry (atau langsung ke HF Spaces via git push, sesuai mekanisme HF).
- Deploy API ke Hugging Face Spaces (Docker SDK).
- Deploy web React ke static hosting (Vercel — pilih salah satu yang gratis dan stabil).

### 6.7 Monitoring (Should Have)
- Logging setiap prediksi (`/classify`) dengan timestamp, input, output, confidence.
- Drift check sederhana (Evidently AI atau perhitungan manual distribusi label dari waktu ke waktu) yang bisa dipicu manual atau via scheduled job.
- Dashboard atau laporan ringkas drift (bisa berupa halaman statis HTML/markdown report yang di-generate berkala).

### 6.8 Dokumentasi & Storytelling (Must Have)
- README utama dengan: problem statement, arsitektur diagram, cara menjalankan, demo link/GIF.
- Catatan eksperimen (apa yang dicoba, apa yang gagal, trade-off) — bisa di README atau `EXPERIMENTS.md`.

## 7. Arsitektur Tingkat Tinggi

```
[Dataset Review] --DVC/DagsHub--> [Preprocessing] --> [Fine-tune IndoBERT]
                                                            |
                                                       MLflow tracking
                                                            |
                                                     [Model Registry]
                                                            |
        +----------------------------+--------------------+
        |                                                  |
   [FastAPI /classify]                          [Embed review -> VectorDB]
        |                                                  |
        |                                       [FastAPI /insight, /chat]
        |                                          (retrieve + LLM call)
        +----------------------------+--------------------+
                                      |
                            [Docker -> HF Spaces]
                                      |
                             [React Web (Vercel/Netlify)]
                                      |
                          [Logging -> Drift/Monitoring report]
```

## 8. Stack Teknis (Ringkas)

| Layer | Tools |
|---|---|
| Data versioning | DVC + DagsHub |
| Model fine-tuning | HuggingFace Transformers, IndoBERT |
| Experiment tracking | MLflow |
| Vector DB | ChromaDB (default, lebih simpel) atau FAISS |
| LLM | Anthropic API / OpenAI API (+ opsional Ollama lokal) |
| Backend | FastAPI, Python |
| Deployment API | Docker, Hugging Face Spaces |
| Frontend | React (Vite), deploy ke Vercel/Netlify |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI atau custom drift script |

## 9. Metrik Keberhasilan

- **Teknis:** F1-macro classifier ≥ baseline TF-IDF + Logistic Regression (sebagai pembanding wajib).
- **Sistem:** API live dan dapat diakses publik tanpa downtime saat demo.
- **Portofolio:** README & demo bisa dipahami recruiter non-ML dalam < 3 menit baca/tonton.
- **MLOps story:** minimal 5 eksperimen tercatat di MLflow dengan perbandingan jelas.

## 10. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Biaya API LLM membengkak saat demo publik | Rate limit endpoint, cache hasil insight, sediakan mode "demo data" statis sebagai fallback |
| HF Spaces free tier resource terbatas (CPU only, cold start lambat) | Gunakan model classifier kecil/quantized, tampilkan loading state di frontend |
| Dataset review terbatas/tidak representatif | Augmentasi data, kombinasikan beberapa sumber (Halodoc + Gojek/Tokopedia) |
| Scope creep (terlalu banyak fitur "should have") | Prioritaskan Must Have dulu, Should Have jadi stretch goal di akhir timeline |

## 11. Milestone Tingkat Tinggi

Lihat `PLAN.md` untuk breakdown detail per fase dan estimasi waktu.
