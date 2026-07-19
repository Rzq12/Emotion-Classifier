# Drift Report — Indo Review Intelligence

Generated: 2026-07-19T05:39:54.756690+00:00
Reference: 1798 rows | Current: 300 rows

## Ringkasan

| Metrik | PSI | Verdict |
|---|---|---|
| Distribusi label | 1.0936 | DRIFT |
| Distribusi panjang teks | 0.3055 | DRIFT |

Ambang PSI: < 0.1 stabil, 0.1-0.2 moderat, >= 0.2 drift signifikan.

## Distribusi Label

| Label | Reference | Current |
|---|---|---|
| anger | 10.07% | 48.00% |
| happiness | 62.96% | 20.00% |
| sadness | 26.97% | 32.00% |

## Distribusi Panjang Teks (karakter)

| Bucket | Reference | Current |
|---|---|---|
| 0-25 | 21.64% | 9.33% |
| 26-50 | 22.36% | 11.00% |
| 51-100 | 24.25% | 27.00% |
| 101-200 | 20.02% | 29.33% |
| 201-500 | 10.85% | 20.67% |
| >500 | 0.89% | 2.67% |

## Confidence Prediksi

- Rata-rata confidence: **0.6933**
- Porsi prediksi low-confidence (< 0.7): **51.67%**

## Kesimpulan

Terindikasi **drift signifikan** — pertimbangkan review data terbaru dan retraining model.
