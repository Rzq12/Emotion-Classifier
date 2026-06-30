# PLAN.md — Roadmap Eksekusi "Indo Review Intelligence"

Dokumen ini adalah sumber kebenaran progres proyek. Update checklist setiap selesai sesi kerja. Estimasi waktu asumsi dikerjakan part-time (~10-15 jam/minggu).

---

## Fase 0 — Setup Project (≈ 2-3 hari)

- [x] Init struktur folder sesuai `CLAUDE.md` (git init lokal; repo GitHub menyusul).
- [x] Setup environment (conda env `trading`, Python 3.11, `requirements.txt` awal) + `pyproject.toml` (ruff/black/pytest).
- [ ] Setup DVC + remote DagsHub — **tertunda**: `dvc` belum terpasang di environment.
- [ ] Setup MLflow tracking — digeser ke Fase 2 (saat training mulai).
- [~] Skeleton `web/` — **ditunda ke Fase 5** secara sengaja (placeholder + `.env.example` dibuat) agar tidak menarik dependency npm sebelum layer data siap.
- [x] Tulis README awal + `.env.example` backend & frontend.

**Output Fase 0:** Repo siap dikembangkan, tracking infra sudah connect.

---

## Fase 1 — Data Acquisition & Preprocessing (≈ 4-6 hari)

- [x] Dataset review (emotion, domain Halodoc/Gojek) tersedia di `Dataset/` — tidak perlu scraping tambahan untuk fase ini.
- [ ] (Opsional) Scrape tambahan review Gojek/Tokopedia — di-skip (data existing cukup).
- [x] EDA ringkas: distribusi label, panjang teks, duplikasi, casing label (dicatat di `EXPERIMENTS.md`).
- [x] Skema label final: **emotion 3-kelas** (`anger`/`happiness`/`sadness`), tanpa netral — mengikuti dataset nyata (reframe dari sentiment, disetujui).
- [x] Preprocessing pipeline: cleaning, normalisasi slang Indo, handling emoji (`src/data/preprocessing.py`, `src/data/slang_dict.py`).
- [x] Split data train/val (test sudah terpisah), simpan ke `data/processed/`.
- [ ] `dvc add` dataset, push DagsHub — **tertunda** (dvc belum terpasang).

**Output Fase 1:** Dataset bersih, ter-version, siap training. Checkpoint: tulis ringkat di `EXPERIMENTS.md` soal keputusan labeling/cleaning.

---

## Fase 2 — Model Training & Experiment Tracking (≈ 5-7 hari)

- [ ] Bangun baseline: TF-IDF + Logistic Regression (atau SVM), log ke MLflow sebagai eksperimen pertama.
- [ ] Setup fine-tuning IndoBERT (HuggingFace Transformers) untuk klasifikasi sentiment.
- [ ] Jalankan minimal 5 variasi eksperimen (learning rate, max_length, augmentasi data, class weighting untuk imbalance, dll), semua ter-log MLflow.
- [ ] Evaluasi tiap eksperimen: accuracy, F1-macro, confusion matrix — bandingkan dengan baseline.
- [ ] Pilih model terbaik, tandai sebagai "production candidate" di MLflow registry.
- [ ] (Opsional) Fine-tune juga untuk emotion classification jika waktu memungkinkan.
- [ ] Export model final ke format siap serving (HF format / TorchScript / ONNX jika perlu optimasi).
- [ ] Push model artifact ke HF Hub model repo (untuk dipull saat deploy, bukan disimpan di git).

**Output Fase 2:** Model classifier final + 5+ eksperimen tercatat rapi di MLflow.

---

## Fase 3 — RAG & LLM Insight Layer (≈ 5-7 hari)

- [ ] Setup ChromaDB lokal, bangun pipeline embedding review (terutama review negatif) ke vector store.
- [ ] Pilih embedding model (bisa pakai model multilingual kecil, atau embedding API).
- [ ] Bangun LLM client wrapper (abstraksi provider: Anthropic/OpenAI/Ollama).
- [ ] Buat prompt template untuk **insight generator**: input = kumpulan review ter-retrieve, output = ringkasan terstruktur (tema keluhan, contoh kutipan, saran tindak lanjut).
- [ ] Buat prompt template untuk **chatbot internal**: input = pertanyaan bebas, output = jawaban grounded ke review ter-retrieve, termasuk fallback kalau tidak ada data relevan.
- [ ] Uji manual kualitas output (cek hallucination, relevansi) dengan beberapa skenario pertanyaan nyata.
- [ ] Tambahkan caching sederhana untuk hasil insight per filter/periode.

**Output Fase 3:** RAG pipeline + LLM insight generator + chatbot logic berfungsi end-to-end (masih lokal, belum diserving via API).

---

## Fase 4 — API & Serving (≈ 3-5 hari)

- [ ] Bangun FastAPI app dengan endpoint `/classify`, `/insight`, `/chat`, `/health` (kontrak lihat PRD section 6.4).
- [ ] Pydantic schema untuk semua request/response.
- [ ] Integrasi model classifier (load dari HF Hub) ke `/classify`.
- [ ] Integrasi RAG+LLM pipeline ke `/insight` dan `/chat`.
- [ ] Tambahkan rate limiting sederhana untuk endpoint yang panggil LLM.
- [ ] Tulis unit test + API contract test (mock LLM call).
- [ ] Tulis `Dockerfile` untuk API, test build & run lokal.

**Output Fase 4:** API berjalan lokal via Docker, semua endpoint teruji.

---

## Fase 5 — Deployment (≈ 3-4 hari)

- [ ] Buat Hugging Face Space baru (Docker SDK), set secrets (API key LLM, dll) di Space settings.
- [ ] Deploy API ke HF Spaces, verifikasi semua endpoint bisa diakses publik via HTTPS.
- [ ] Konfigurasi CORS di FastAPI supaya bisa diakses dari domain web React.
- [ ] Setup GitHub Actions: run test → build Docker image → (push ke HF Spaces via git remote atau registry, sesuai metode yang dipilih).
- [ ] Bangun frontend React final: Dashboard, InsightPanel, ChatBox, ClassifyDemo (lihat PRD section 6.5).
- [ ] Set `VITE_API_BASE_URL` ke URL HF Spaces, test integrasi end-to-end.
- [ ] Deploy web React ke Vercel/Netlify.
- [ ] Smoke test full flow dari web publik: classify → insight → chat.

**Output Fase 5:** Sistem live, bisa diakses publik (API di HF Spaces, web di Vercel/Netlify).

---

## Fase 6 — Monitoring & Polish (≈ 3-4 hari)

- [ ] Implementasi logging prediksi `/classify` (timestamp, input, output, confidence) ke storage sederhana.
- [ ] Bangun script drift check (`src/monitoring/check_drift.py`) — bisa pakai Evidently AI atau custom.
- [ ] Generate contoh drift report (bisa simulasi dengan data baru yang sengaja berbeda distribusi, untuk demo capability).
- [ ] Review & lengkapi README utama: arsitektur diagram, cara run, link demo live, GIF/screenshot.
- [ ] Tulis `EXPERIMENTS.md` final: ringkas apa yang dicoba, apa yang gagal, trade-off (untuk storytelling portofolio).
- [ ] Cek ulang security hygiene: tidak ada secret ter-commit, `.env.example` tersedia, rate limit aktif.
- [ ] Final review keseluruhan demo flow seakan-akan kamu recruiter yang baru lihat pertama kali.

**Output Fase 6:** Proyek selesai, terdokumentasi rapi, siap dicantumkan di portofolio/LinkedIn/CV.

---

## Stretch Goals (kerjakan kalau waktu/energi masih ada)

- [ ] Emotion classification (bukan cuma sentiment) sebagai layer tambahan.
- [ ] Multi-app comparison (bandingkan insight Halodoc vs Gojek vs Tokopedia).
- [ ] Scheduled job (GitHub Actions cron) untuk auto-refresh drift report mingguan.
- [ ] Quantization model classifier untuk inference lebih cepat di HF Spaces free tier.
- [ ] Autentikasi sederhana untuk chatbot internal (biar tidak benar-benar publik dipakai sembarangan orang dan boros API cost).

---

## Catatan Progres

*(isi log singkat tiap sesi kerja di sini — tanggal, apa yang dikerjakan, blocker jika ada)*

- 2026-06-30: Dokumen PRD/CLAUDE/PLAN dibuat. Belum mulai eksekusi.
- 2026-06-30: **Fase 0 & Fase 1 dieksekusi.** Reframe sentiment → emotion (3 kelas, tanpa netral) disetujui. Scaffold repo, `requirements.txt`, `pyproject.toml`, `.env.example` (BE/FE), README, `.gitignore`. Pipeline data (`src/data/`) + 21 unit test (semua hijau) + lint ruff bersih. Output `data/processed/{train,val,test}.csv` (1798/318/548). Blocker minor: `dvc` & GitHub remote belum di-setup (digeser). Detail di `EXPERIMENTS.md`.
