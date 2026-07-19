import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { emotionMeta, EMOTION_ORDER } from "../lib/constants";
import { DonutChart } from "../components/DonutChart";
import { LoadingBlock, ErrorState } from "../components/States";

function Stat({ label, value, sub }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-faint">{label}</p>
      <p className="mt-1 font-display text-3xl tnum text-ink">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </div>
  );
}

function SplitBar({ name, counts }) {
  const total = EMOTION_ORDER.reduce((s, k) => s + (counts[k] || 0), 0);
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-sm capitalize text-ink">{name}</span>
        <span className="font-mono text-xs tnum text-faint">{total.toLocaleString("id-ID")}</span>
      </div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-line">
        {EMOTION_ORDER.map((k) => {
          const w = total > 0 ? ((counts[k] || 0) / total) * 100 : 0;
          if (w === 0) return null;
          return <div key={k} style={{ width: `${w}%`, backgroundColor: emotionMeta(k).color }} />;
        })}
      </div>
    </div>
  );
}

export function Dashboard() {
  const [state, setState] = useState({ status: "loading", data: null, error: null });

  const load = () => {
    setState({ status: "loading", data: null, error: null });
    api
      .stats()
      .then((data) => setState({ status: "ok", data, error: null }))
      .catch((e) => setState({ status: "error", data: null, error: e.message }));
  };

  useEffect(load, []);

  if (state.status === "loading") return <LoadingBlock label="Memuat statistik review..." />;
  if (state.status === "error") return <ErrorState message={state.error} onRetry={load} />;

  const { total, by_emotion, by_split, negative_ratio } = state.data;

  return (
    <div className="animate-fade-up space-y-10">
      <div>
        <h1 className="font-display text-2xl text-ink">Gambaran Emosi Review</h1>
        <p className="mt-1 text-sm text-muted">
          Distribusi emosi dari seluruh review yang dianalisis sistem.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6 border-y border-line py-6 sm:grid-cols-4">
        <Stat label="Total Review" value={total.toLocaleString("id-ID")} />
        <Stat
          label="Emosi Negatif"
          value={`${Math.round(negative_ratio * 100)}%`}
          sub="marah + sedih"
        />
        <Stat label="Senang" value={by_emotion.happiness.toLocaleString("id-ID")} />
        <Stat label="Marah" value={by_emotion.anger.toLocaleString("id-ID")} />
      </div>

      <section className="rounded-lg border border-line bg-surface p-6 sm:p-8">
        <h2 className="mb-6 text-sm font-medium uppercase tracking-wide text-faint">
          Distribusi Emosi
        </h2>
        <DonutChart data={by_emotion} total={total} />
      </section>

      <section>
        <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-faint">
          Komposisi per Bagian Data
        </h2>
        <div className="space-y-4">
          {Object.entries(by_split).map(([name, counts]) => (
            <SplitBar key={name} name={name} counts={counts} />
          ))}
        </div>
      </section>
    </div>
  );
}
