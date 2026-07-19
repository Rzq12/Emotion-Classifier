# RAG Evaluation Report — Indo Review Intelligence

Generated: 2026-07-19T06:37:37+00:00 | LLM: groq:llama-3.3-70b-versatile | korpus: 2664 review

Evaluasi end-to-end dengan index ChromaDB asli dan panggilan LLM sungguhan
(bukan mock). Groundedness dinilai LLM-as-judge (skala 1-5).

## 1. Retrieval (golden set)

| Mode | hit@1 | hit@5 | MRR |
|---|---|---|---|
| vector-only | 1.0 | 1.0 | 1.0 |
| hybrid | 1.0 | 1.0 | 1.0 |

## 2. Chat end-to-end

- Terjawab: **10/10**
- Sources valid (⊆ retrieved): **10/10**
- Groundedness rata-rata: **4.80/5** (min 4, max 5)
- Latency rata-rata: **2.30s**

| Query | Jawab | Sumber | Grounded | Latency |
|---|---|---|---|---|
| knp ga bisa bayar pake dana | ya | 4 | 5/5 | 1.07s |
| aplikasi error pas transaksi | ya | 5 | 5/5 | 0.81s |
| dokternya gimana, ramah ga? | ya | 8 | 5/5 | 0.52s |
| pengiriman obat lama banget | ya | 8 | 5/5 | 0.62s |
| cs nya lambat merespon keluhan | ya | 7 | 4/5 | 0.64s |
| aplikasi lemot pas dibuka | ya | 8 | 5/5 | 0.52s |
| voucher ga bisa dipake | ya | 8 | 5/5 | 3.73s |
| saldo ga balik padahal transaksi gagal | ya | 4 | 5/5 | 4.69s |
| konsultasi sama dokter memuaskan ga? | ya | 8 | 5/5 | 4.63s |
| resep obat ga bisa ditebus | ya | 3 | 4/5 | 5.73s |

## 3. Penolakan query di luar domain

Ditolak dengan benar: **3/3**

| Probe | Ditolak | Sources |
|---|---|---|
| resep masakan rendang enak | ya | 0 |
| jadwal pertandingan bola nanti malam | ya | 0 |
| harga saham hari ini naik atau turun | ya | 0 |

## 4. Insight end-to-end

- Schema valid: **3/3**
- Semua example_review_ids valid: **3/3**
- Cache bekerja: **3/3**
- Groundedness summary rata-rata: **4.67/5**

| Fokus | Tema | Contoh ID (valid) | Grounded | Latency |
|---|---|---|---|---|
| keluhan utama pengguna | 5 | 20 (ya) | 4/5 | 8.68s |
| masalah pembayaran dan transaksi | 5 | 14 (ya) | 5/5 | 9.55s |
| keluhan seputar layanan dokter dan konsultasi | 5 | 16 (ya) | 5/5 | 12.94s |
