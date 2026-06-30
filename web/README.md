# web/ — Frontend React (Vite)

Frontend aplikasi (Dashboard, Insight Panel, Chat, Classify Demo) akan di-scaffold
penuh pada **Fase 5** sesuai `UIUX_FLOW.md` dan `TASK_BREAKDOWN.md` (T5.6–T5.14).

Scaffolding ditunda dari Fase 0 secara sengaja agar `npm install` tidak menambah
beban dependency sebelum layer data/model siap. Konfigurasi environment sudah
disiapkan di `web/.env.example` (`VITE_API_BASE_URL`).

Rencana stack: React + Vite + Tailwind, terhubung ke API FastAPI via `VITE_API_BASE_URL`.
