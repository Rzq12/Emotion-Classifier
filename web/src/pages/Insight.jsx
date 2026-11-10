import { useState } from "react";
import { api } from "../lib/api";
import { Spinner, ErrorState, EmptyState } from "../components/States";
import { ReviewSources } from "../components/ReviewSources";

// Category filter (PRD 6.4): which negative emotion the insight draws from.
const EMOTION_FILTERS = [
  { id: "", label: "Semua negatif" },
  { id: "anger", label: "Marah" },
  { id: "sadness", label: "Sedih" },
];

function ThemeCard({ theme }) {
  return (
    <div className="border-l-2 border-line pl-4">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-medium text-ink">{theme.theme}</h3>
        {theme.count != null && (
          <span className="shrink-0 font-mono text-xs tnum text-faint">{theme.count} review</span>
        )}
      </div>
      <ReviewSources
        ids={theme.example_review_ids}
        label={`Lihat ${theme.example_review_ids?.length} review contoh`}
      />
    </div>
  );
}

export function Insight() {
  const [query, setQuery] = useState("keluhan utama pengguna");
  const [emotion, setEmotion] = useState("");
  const [state, setState] = useState({ status: "idle", data: null, error: null });

  const generate = () => {
    if (!query.trim()) return;
    setState({ status: "loading", data: null, error: null });
    api
      .insight(query.trim(), emotion ? [emotion] : [])
      .then((data) => setState({ status: "ok", data, error: null }))
      .catch((e) => setState({ status: "error", data: null, error: e.message }));
  };

  const data = state.data;
  const hasThemes = data?.themes?.length > 0;

  return (
    <div className="animate-fade-up space-y-8">
      <div>
        <h1 className="font-display text-2xl text-ink">Insight Otomatis</h1>
        <p className="mt-1 text-sm text-muted">
          Ringkasan keluhan dari review beremosi negatif, dirangkum oleh LLM dengan grounding ke data.
        </p>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && generate()}
          placeholder="Fokus insight, mis. 'masalah pembayaran'"
          className="flex-1 rounded-md border border-line bg-surface px-4 py-2.5 text-sm text-ink placeholder:text-faint focus:border-ink"
        />
        <button
          onClick={generate}
          disabled={state.status === "loading" || !query.trim()}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
        >
          {state.status === "loading" && <Spinner />}
          {state.status === "loading" ? "Menganalisis..." : "Buat Insight"}
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Filter emosi">
        <span className="mr-2 text-xs text-faint">Emosi:</span>
        {EMOTION_FILTERS.map((f) => (
          <button
            key={f.id}
            onClick={() => setEmotion(f.id)}
            aria-pressed={emotion === f.id}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
              emotion === f.id ? "bg-ink text-paper" : "text-faint hover:text-muted"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {state.status === "idle" && (
        <EmptyState
          title="Belum ada insight"
          hint="Tentukan fokus lalu klik Buat Insight untuk merangkum keluhan pengguna."
        />
      )}
      {state.status === "error" && <ErrorState message={state.error} onRetry={generate} />}

      {state.status === "ok" && data && (
        <div className="animate-fade-up space-y-8">
          <div className="flex items-start gap-3 rounded-lg border border-line bg-surface p-6">
            <p className="text-[15px] leading-relaxed text-ink">{data.summary}</p>
            {data.cached && (
              <span className="shrink-0 rounded-full border border-line px-2 py-0.5 text-xs text-faint">
                dari cache
              </span>
            )}
            {data.degraded && (
              <span className="shrink-0 rounded-full border border-anger/40 px-2 py-0.5 text-xs text-anger">
                tanpa LLM
              </span>
            )}
          </div>

          {data.degraded ? (
            data.note && <p className="text-xs italic text-faint">{data.note}</p>
          ) : hasThemes ? (
            <section>
              <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-faint">
                Tema Keluhan
              </h2>
              <div className="space-y-5">
                {data.themes.map((t, i) => (
                  <ThemeCard key={i} theme={t} />
                ))}
              </div>
              {data.note && <p className="mt-4 text-xs italic text-faint">{data.note}</p>}
            </section>
          ) : (
            <EmptyState title="Tidak ada tema spesifik" hint="Data negatif belum cukup untuk dirangkum." />
          )}

          {data.sample_quotes?.length > 0 && (
            <section>
              <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-faint">
                Kutipan
              </h2>
              <div className="space-y-3">
                {data.sample_quotes.map((q, i) => (
                  <blockquote key={i} className="border-l-2 border-sadness/40 pl-4 text-sm italic text-muted">
                    “{q}”
                  </blockquote>
                ))}
              </div>
            </section>
          )}

          {data.recommendations?.length > 0 && (
            <section>
              <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-faint">
                Rekomendasi
              </h2>
              <ul className="space-y-2">
                {data.recommendations.map((r, i) => (
                  <li key={i} className="flex gap-2.5 text-sm text-ink">
                    <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-happiness" aria-hidden />
                    {r}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
