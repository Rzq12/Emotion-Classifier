# SRS — Software Requirements Specification
**Indo Review Intelligence: Hybrid Classifier + RAG Insight Generator**

Versi: 1.0
Referensi: `PRD.md` (requirement bisnis/produk), dokumen ini menerjemahkan ke spesifikasi teknis yang lebih presisi dan testable.

---

## 1. Pendahuluan

### 1.1 Tujuan Dokumen
Mendefinisikan kebutuhan fungsional dan non-fungsional sistem secara presisi sehingga bisa dipakai sebagai acuan implementasi dan acceptance testing.

### 1.2 Ruang Lingkup
Sistem terdiri dari: pipeline data & training (offline), API backend (FastAPI di HF Spaces), web frontend (React), dan komponen monitoring. Lihat `PRD.md` section 3 untuk non-tujuan.

### 1.3 Definisi & Istilah

| Istilah | Definisi |
|---|---|
| Review | Teks ulasan pengguna aplikasi dari Play Store/App Store |
| Sentiment | Label positive/negative/neutral hasil klasifikasi |
| Insight | Ringkasan terstruktur hasil RAG+LLM dari kumpulan review |
| Grounded | Jawaban LLM yang didasarkan pada data ter-retrieve, bukan general knowledge |
| Drift | Perubahan signifikan distribusi data/prediksi dari waktu ke waktu |

---

## 2. Functional Requirements (FR)

### 2.1 Modul Klasifikasi

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-1.1 | Sistem dapat menerima satu teks review dan mengembalikan label emotion | Output berupa label (`anger`/`happiness`/`sadness`) + confidence score (0-1) dalam < 1 detik |
| FR-1.2 | Sistem dapat memproses input batch (opsional, stretch) | Menerima list teks, mengembalikan list hasil dengan urutan sama |
| FR-1.3 | Model classifier harus dapat di-reproduce dari eksperimen tertentu | Setiap model production candidate punya run_id MLflow yang dapat ditelusuri |

### 2.2 Modul RAG & Insight

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-2.1 | Sistem dapat mengambil review negatif relevan berdasarkan filter periode/kategori | Retrieval mengembalikan minimal top-k (default k=10) review paling relevan |
| FR-2.2 | Sistem dapat men-generate ringkasan insight dari review ter-retrieve | Output mencakup: tema keluhan utama, contoh kutipan (anonim), saran tindak lanjut |
| FR-2.3 | Insight generator harus grounded pada data, bukan halusinasi | Setiap klaim tema dalam insight dapat ditelusuri ke minimal satu review sumber |
| FR-2.4 | Hasil insight untuk kombinasi filter yang sama di-cache | Permintaan kedua dengan filter identik dalam window waktu tertentu tidak memanggil LLM ulang |

### 2.3 Modul Chatbot Internal

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-3.1 | Sistem menerima pertanyaan bebas bahasa natural dan mengembalikan jawaban berbasis data review | Jawaban menyertakan referensi ke review yang mendasari (minimal disebutkan jumlah review terkait) |
| FR-3.2 | Sistem menangani kasus tidak ada data relevan | Mengembalikan pesan eksplisit "tidak ditemukan data relevan", bukan jawaban mengarang |
| FR-3.3 | Riwayat percakapan dalam satu sesi chat dipertahankan (opsional, stretch) | Pertanyaan lanjutan dapat merujuk konteks pertanyaan sebelumnya dalam sesi yang sama |

### 2.4 Modul API

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-4.1 | Endpoint `POST /classify` menerima `{text: string}`, mengembalikan `{label, confidence}` | Sesuai schema Pydantic, validasi input kosong/terlalu panjang dengan error 422 |
| FR-4.2 | Endpoint `POST /insight` menerima filter periode/kategori, mengembalikan ringkasan terstruktur | Response mencakup `summary`, `themes[]`, `sample_quotes[]`, `recommendations[]` |
| FR-4.3 | Endpoint `POST /chat` menerima `{question: string, session_id?: string}`, mengembalikan jawaban + sumber | Response mencakup `answer`, `sources[]` (id review yang dirujuk) |
| FR-4.4 | Endpoint `GET /health` mengembalikan status sistem | Mengembalikan status model loaded, vector DB connected, LLM provider reachable |
| FR-4.5 | Semua endpoint LLM-call dibatasi rate limit | Default: maksimal N request/menit per IP (konfigurasi via env var) |

### 2.5 Modul Frontend

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-5.1 | Dashboard menampilkan distribusi sentimen dalam bentuk chart | Chart update sesuai filter periode yang dipilih user |
| FR-5.2 | Panel insight menampilkan hasil `/insight` dengan format terbaca (tema, kutipan, rekomendasi) | Loading state ditampilkan selama request berlangsung |
| FR-5.3 | Chat interface mengirim pertanyaan ke `/chat` dan menampilkan jawaban + sumber | History percakapan tampil dalam urutan kronologis di UI |
| FR-5.4 | Form demo classify menerima input teks bebas dan menampilkan hasil klasifikasi langsung | Hasil tampil < 2 detik setelah submit (di luar cold start) |

### 2.6 Modul Monitoring

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| FR-6.1 | Setiap prediksi `/classify` dicatat (timestamp, input, output, confidence) | Log dapat diquery untuk membuat drift report |
| FR-6.2 | Sistem dapat menghasilkan laporan drift dari periode tertentu | Laporan mencakup perbandingan distribusi label periode A vs B |

---

## 3. Non-Functional Requirements (NFR)

| ID | Kategori | Requirement |
|---|---|---|
| NFR-1 | Performance | `/classify` merespons < 1 detik (p95) pada CPU HF Spaces free tier (di luar cold start) |
| NFR-2 | Performance | `/insight` dan `/chat` merespons < 10 detik (p95), didominasi latency LLM API eksternal |
| NFR-3 | Availability | API target uptime selama jam demo aktif (tidak ada SLA formal, tapi cold start harus di-handle UI dengan baik) |
| NFR-4 | Security | Tidak ada secret/API key tersimpan di kode atau git history |
| NFR-5 | Security | CORS dikonfigurasi hanya mengizinkan origin domain web yang sah |
| NFR-6 | Cost Control | Pemanggilan LLM dibatasi rate limit dan caching untuk mencegah biaya API tidak terkontrol |
| NFR-7 | Reproducibility | Semua eksperimen training dapat direproduksi dari config + dataset version + MLflow run |
| NFR-8 | Maintainability | Kode mengikuti struktur modular sesuai `CLAUDE.md`, dengan type hints dan test coverage minimal untuk fungsi kritikal |
| NFR-9 | Usability | UI dapat digunakan tanpa training, label dan flow dalam Bahasa Indonesia yang jelas |
| NFR-10 | Portability | Seluruh sistem dapat dijalankan lokal via Docker Compose tanpa bergantung HF Spaces (untuk development) |

---

## 4. Data Requirements

- Dataset review minimal mencakup kolom: `review_id`, `text`, `rating` (jika ada), `app_source`, `date`, `label` (setelah anotasi/training).
- Data sensitif (jika ada nama pengguna eksplisit dalam teks review) wajib dianonimkan sebelum ditampilkan di insight/kutipan UI.
- Dataset versioning wajib via DVC; setiap perubahan signifikan dataset dicatat di `EXPERIMENTS.md`.

## 5. Constraints

- Backend harus dapat berjalan di lingkungan HF Spaces (CPU only, resource terbatas pada free tier).
- LLM provider eksternal (Anthropic/OpenAI) memerlukan API key — sistem harus tetap dapat didemokan meski API key tidak tersedia (fallback ke data/contoh statis untuk demo).
- Bahasa utama sistem dan dataset adalah Bahasa Indonesia, termasuk variasi informal/slang.

## 6. Asumsi

- Volume review untuk demo dalam skala ribuan, bukan jutaan — tidak perlu arsitektur big data.
- Pengguna sistem (tim produk) adalah single-tenant internal tool, bukan multi-organisasi.
- Tidak ada requirement kepatuhan data privasi formal (GDPR dsb) karena ini proyek portofolio, namun praktik anonimisasi tetap diterapkan sebagai good practice.

## 7. Traceability ke PRD

Setiap FR di atas memetakan ke section requirement terkait di `PRD.md` section 6 (6.1–6.8). Perubahan pada PRD wajib disinkronkan ke SRS ini agar tidak terjadi drift dokumentasi.
