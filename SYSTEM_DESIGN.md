# System Design — Indo Review Intelligence

Versi: 1.0
Referensi: `PRD.md`, `SRS.md`

---

## 1. Tujuan Desain

Mendesain sistem yang (a) ringan dijalankan di infra gratis/murah (HF Spaces CPU free tier), (b) modular sehingga tiap komponen (classifier, RAG, LLM) dapat diuji dan diganti independen, dan (c) menunjukkan praktik MLOps yang jelas untuk keperluan portofolio.

---

## 2. Arsitektur Komponen

```
                         ┌─────────────────────────┐
                         │      React Web App      │
                         │   (Vercel)      │
                         │  Dashboard | Insight |   │
                         │  Chat | Classify Demo    │
                         └────────────┬─────────────┘
                                      │ HTTPS (REST/JSON)
                                      ▼
                         ┌─────────────────────────┐
                         │     FastAPI Backend     │
                         │   (HF Spaces, Docker)   │
                         │                          │
                         │  /classify  /insight     │
                         │  /chat      /health      │
                         └──┬──────────┬─────────┬──┘
                            │          │         │
              ┌─────────────┘    ┌─────┘    ┌────┘
              ▼                  ▼          ▼
     ┌────────────────┐ ┌────────────────┐ ┌──────────────┐
     │  Classifier     │ │  Retriever      │ │  Monitoring  │
     │  (IndoBERT,     │ │  (ChromaDB)     │ │  (logging +  │
     │  loaded from    │ │                 │ │  drift check)│
     │  HF Hub)        │ │                 │ │              │
     └────────────────┘ └───────┬────────┘ └──────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │  LLM Client     │
                         │  Wrapper        │
                         │ (Anthropic/     │
                         │  OpenAI/Ollama) │
                         └────────────────┘

   ── Offline / Training Pipeline (terpisah dari serving path) ──

   [Raw Dataset] → DVC → [Preprocessing] → [Fine-tune IndoBERT]
                                                │
                                          MLflow Tracking
                                                │
                                       [Model Registry] → push ke HF Hub
                                                │
                                    [Embed reviews] → ChromaDB snapshot
```

**Prinsip kunci:** training/embedding pipeline berjalan **offline** (lokal/notebook/script terpisah), hasilnya (model + vector store snapshot) di-push ke storage (HF Hub / volume) lalu di-load oleh API saat startup. API tidak melakukan training maupun re-embedding penuh secara live — ini menjaga API tetap ringan dan cepat start di HF Spaces.

---

## 3. Komponen Detail

### 3.1 Classifier Service
- **Input:** teks review (string).
- **Proses:** tokenisasi IndoBERT tokenizer → forward pass model fine-tuned → softmax → label emotion (`anger`/`happiness`/`sadness`) + confidence.
- **Model loading:** model di-pull dari HF Hub model repo saat container start (cache di image layer atau volume agar tidak re-download tiap request).
- **Optimasi:** pertimbangkan quantization (dynamic quantization PyTorch atau ONNX) jika cold start/latency jadi masalah di CPU-only HF Spaces.

### 3.2 Retriever (Vector Store)
- **Engine:** ChromaDB, mode persistent (disimpan sebagai file di dalam container/volume, di-build ulang dari snapshot saat deploy).
- **Embedding model:** model embedding multilingual ringan (mis. `sentence-transformers` multilingual) — dipilih karena CPU-friendly, bukan API embedding berbayar (untuk kontrol biaya), kecuali kualitas retrieval terbukti tidak cukup.
- **Index granularity:** satu vector per review (chunking tidak terlalu diperlukan karena review umumnya pendek; jika ada review sangat panjang, baru di-chunk).
- **Metadata tersimpan per vector:** `review_id`, `date`, `app_source`, `sentiment_label`, `rating`.

### 3.3 LLM Client Wrapper
- Interface abstrak `LLMClient` dengan method seperti `generate(prompt, ...) -> str`.
- Implementasi konkret: `AnthropicClient`, `OpenAIClient`, `OllamaClient` — dipilih via environment variable `LLM_PROVIDER`.
- Wrapper menangani retry sederhana dan error handling seragam (timeout, rate limit dari provider) sehingga route handler API tidak perlu tahu detail provider.

### 3.4 Insight Generator (RAG Pipeline)
1. Terima filter (periode, kategori/app_source) dari request `/insight`.
2. Query ChromaDB dengan filter metadata + similarity search (atau pure metadata filter jika filter cukup spesifik) untuk ambil top-k review negatif relevan.
3. Susun prompt terstruktur (template di `src/llm/prompts/insight.txt`) berisi daftar review + instruksi format output (tema, kutipan, rekomendasi).
4. Panggil LLM via wrapper, parse output (idealnya LLM diminta output JSON terstruktur untuk memudahkan parsing di API).
5. Cache hasil (key: hash dari filter) dengan TTL tertentu (mis. 1 jam) untuk efisiensi biaya.

### 3.5 Chatbot (RAG Pipeline)
1. Terima pertanyaan bebas dari `/chat`.
2. Embed pertanyaan, query ChromaDB untuk review paling relevan (top-k, mis. 8).
3. Susun prompt dengan instruksi eksplisit: jawab hanya berdasarkan review yang diberikan, jika tidak ada yang relevan katakan tidak tahu.
4. Panggil LLM, kembalikan jawaban + daftar `review_id` sumber.
5. (Stretch) simpan riwayat sesi in-memory (dict per `session_id`) untuk follow-up question — tidak perlu database session permanen untuk skala demo.

### 3.6 Monitoring
- Setiap request `/classify` menulis baris log (JSON lines file atau SQLite ringan) berisi timestamp, input (hash/truncated untuk privasi), label, confidence.
- Script terpisah (`src/monitoring/check_drift.py`) membaca log, membandingkan distribusi label antar periode, output report (markdown/HTML) — dijalankan manual atau via scheduled GitHub Action (opsional).

---

## 4. Desain API (Kontrak)

### `POST /classify`
```json
// Request
{ "text": "Aplikasinya sering error pas checkout" }

// Response
{ "label": "negative", "confidence": 0.93 }
```

### `POST /insight`
```json
// Request
{ "period_start": "2026-06-01", "period_end": "2026-06-30", "app_source": "halodoc" }

// Response
{
  "summary": "Keluhan dominan terkait proses checkout dan respon CS.",
  "themes": [
    { "theme": "Checkout error", "count": 42, "example_review_ids": ["r123", "r456"] },
    { "theme": "Respon CS lambat", "count": 18, "example_review_ids": ["r789"] }
  ],
  "sample_quotes": ["Checkout selalu gagal di langkah pembayaran", "..."],
  "recommendations": ["Audit ulang flow pembayaran", "Tambah kapasitas tim CS jam sibuk"]
}
```

### `POST /chat`
```json
// Request
{ "question": "Apa keluhan utama soal pembayaran minggu ini?", "session_id": "abc123" }

// Response
{
  "answer": "Keluhan utama soal pembayaran adalah gagal checkout berulang...",
  "sources": ["r123", "r456", "r789"]
}
```

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "vector_db_connected": true,
  "llm_provider_reachable": true
}
```

---

## 5. Skema Data

### Tabel/Collection `reviews` (processed dataset, source of truth untuk embedding & training)
| Field | Tipe | Keterangan |
|---|---|---|
| review_id | string | unique identifier |
| text | string | isi review |
| rating | int (1-5) | opsional, jika tersedia dari sumber |
| app_source | string | `halodoc` / `gojek` / `tokopedia` dll |
| date | date | tanggal review |
| sentiment_label | string | hasil klasifikasi/anotasi |
| emotion_label | string (opsional) | hasil klasifikasi emosi jika dikerjakan |

### Log `prediction_logs`
| Field | Tipe |
|---|---|
| timestamp | datetime |
| input_hash | string |
| label | string |
| confidence | float |

---

## 6. Deployment Topology

- **API:** 1 Hugging Face Space (Docker SDK), berisi FastAPI app + model classifier + ChromaDB persistent file + dependency LLM client.
- **Web:** 1 static deployment di Vercel/Netlify, build dari `web/` React app, environment variable `VITE_API_BASE_URL` menunjuk ke URL HF Spaces.
- **Secrets:** disimpan di HF Spaces "Repository secrets" (API key LLM) — tidak pernah masuk git.
- **CORS:** FastAPI mengizinkan origin domain Vercel/Netlify spesifik (bukan wildcard `*` di production).

---

## 7. Skalabilitas & Batasan yang Disadari

Sistem ini didesain untuk skala demo/portofolio (ratusan-ribuan review, traffic rendah), bukan production scale jutaan user. Beberapa simplifikasi yang disengaja:
- ChromaDB persistent file (bukan cluster terdistribusi) cukup untuk skala data ini.
- Tidak ada job queue/async worker terpisah — pemanggilan LLM dilakukan sinkron dalam request (dengan timeout wajar) karena trafik rendah.
- Cold start HF Spaces free tier diterima sebagai trade-off biaya, di-mitigasi dengan UI loading state yang jelas, bukan dihilangkan sepenuhnya.

Jika nanti perlu scale up, jalur natural-nya: ChromaDB → managed vector DB, sync call LLM → job queue (Celery/RQ) + websocket/polling di frontend, HF Spaces → container hosting dengan resource lebih besar (Railway/Render/AWS).

---

## 8. Keputusan Desain & Alasan (ADR Ringkas)

| Keputusan | Alasan |
|---|---|
| ChromaDB dibanding FAISS | Lebih mudah setup persistent + metadata filtering, cukup untuk skala data demo |
| LLM client abstrak multi-provider | Development bisa pakai Ollama (gratis/lokal), demo publik pakai Anthropic/OpenAI (kualitas) tanpa ubah banyak kode |
| Training pipeline terpisah dari serving | Menjaga API ringan & cepat start; mengikuti praktik MLOps standar (offline training, online serving) |
| Sinkron call untuk LLM (bukan async job queue) | Trafik demo rendah, kompleksitas job queue tidak sepadan manfaatnya di skala ini |
| Caching insight per filter | Kontrol biaya API LLM saat demo publik diakses berulang dengan filter sama |
