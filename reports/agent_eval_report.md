# Agent Routing Evaluation — Indo Review Intelligence

Generated: 2026-11-09T04:26:29+00:00 | data: `data/processed/test.csv` (n=548) | threshold eskalasi: **0.75**

Classifier IndoBERT asli + router dari `configs/agent.yaml`, dibandingkan dengan cabang yang dipilih policy yang sama untuk label manusia (gold). Tanpa panggilan LLM — kualitas draft dinilai terpisah lewat approval rate di `/agent/stats`.

## 1. Ringkasan

| Metrik | Nilai |
|---|---|
| Akurasi cabang | 0.827 |
| Escalation precision | 0.837 |
| Escalation recall | 0.406 |
| False escalation rate | 0.163 |
| Missed-negative rate (negatif → archive) | 0.128 (29 review) |
| Latency classify+route p50 / p95 | 176.7 / 232.4 ms |

## 2. Confusion matrix cabang (baris = gold, kolom = prediksi)

| gold \ pred | escalate | draft | archive |
|---|---|---|---|
| escalate | 41 | 54 | 6 |
| draft | 8 | 95 | 23 |
| archive | 0 | 4 | 317 |

## 3. Sweep threshold `escalate_min_confidence`

| Threshold | Precision | Recall | Jumlah eskalasi | Akurasi cabang |
|---|---|---|---|---|
| 0.50 | 0.796 | 0.426 | 54 | 0.825 |
| 0.60 | 0.824 | 0.416 | 51 | 0.827 |
| 0.70 | 0.840 | 0.416 | 50 | 0.829 |
| 0.75 ← | 0.837 | 0.406 | 49 | 0.827 |
| 0.80 | 0.833 | 0.396 | 48 | 0.825 |
| 0.85 | 0.848 | 0.386 | 46 | 0.825 |
| 0.90 | 0.841 | 0.366 | 44 | 0.821 |
| 0.95 | 0.875 | 0.346 | 40 | 0.821 |

## 4. Cara baca

- **Missed-negative** adalah error paling mahal: review negatif yang diarsipkan tidak pernah dilihat manusia. Router sengaja mengirim label tak dikenal ke `draft`.
- Threshold lebih rendah → recall eskalasi naik, tapi notifikasi palsu (precision) ikut naik. Pilih titik yang sesuai kapasitas tim yang menerima notifikasi.
- Latency diukur di mesin lokal (CPU), tanpa LLM; `/agent/run` end-to-end menambah retrieval + generate draft.
