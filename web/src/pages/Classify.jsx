import { useState } from "react";
import { api } from "../lib/api";
import { CLASSIFY_EXAMPLES, EMOTION_ORDER, emotionMeta } from "../lib/constants";
import { EmotionBadge } from "../components/EmotionBadge";
import { ConfidenceBar } from "../components/ConfidenceBar";
import { BatchClassify } from "../components/BatchClassify";
import { Spinner, ErrorState } from "../components/States";

const MAX_LEN = 2000;

const MODES = [
  { id: "single", label: "Satu review" },
  { id: "batch", label: "Banyak review (CSV)" },
];

// All class probabilities, so an ambiguous prediction is visible as ambiguous.
function Probabilities({ probabilities }) {
  if (!probabilities || Object.keys(probabilities).length === 0) return null;
  return (
    <div className="mt-6 border-t border-line pt-5">
      <p className="mb-3 text-xs uppercase tracking-wide text-faint">Probabilitas tiap emosi</p>
      <div className="space-y-2.5">
        {EMOTION_ORDER.map((k) => {
          const pct = Math.round((probabilities[k] || 0) * 100);
          return (
            <div key={k} className="flex items-center gap-3">
              <span className="w-16 shrink-0 text-xs text-muted">{emotionMeta(k).label}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${pct}%`, backgroundColor: emotionMeta(k).color }}
                />
              </div>
              <span className="w-10 shrink-0 text-right font-mono text-xs tnum text-ink">{pct}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SingleClassify() {
  const [text, setText] = useState("");
  const [state, setState] = useState({ status: "idle", result: null, error: null });

  const submit = (value) => {
    const input = (value ?? text).trim();
    if (!input) return;
    setState({ status: "loading", result: null, error: null });
    api
      .classify(input)
      .then((result) => setState({ status: "ok", result, error: null }))
      .catch((e) => setState({ status: "error", result: null, error: e.message }));
  };

  const tooLong = text.length > MAX_LEN;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap gap-2">
        {CLASSIFY_EXAMPLES.map((ex) => (
          <button
            key={ex}
            onClick={() => {
              setText(ex);
              submit(ex);
            }}
            className="rounded-full border border-line bg-surface px-3 py-1.5 text-xs text-muted transition-colors hover:border-ink hover:text-ink"
          >
            {ex.length > 42 ? `${ex.slice(0, 42)}…` : ex}
          </button>
        ))}
      </div>

      <div>
        <label htmlFor="review" className="sr-only">
          Teks review
        </label>
        <textarea
          id="review"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          placeholder="Contoh: Aplikasinya bagus tapi sering lambat saat dibuka..."
          className="w-full resize-none rounded-md border border-line bg-surface p-4 text-sm text-ink placeholder:text-faint focus:border-ink"
        />
        <div className="mt-2 flex items-center justify-between">
          <span className={`font-mono text-xs tnum ${tooLong ? "text-anger" : "text-faint"}`}>
            {text.length}/{MAX_LEN}
          </span>
          <button
            onClick={() => submit()}
            disabled={!text.trim() || tooLong || state.status === "loading"}
            className="inline-flex items-center gap-2 rounded-md bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
          >
            {state.status === "loading" && <Spinner />}
            Klasifikasikan
          </button>
        </div>
      </div>

      {state.status === "error" && <ErrorState message={state.error} onRetry={() => submit()} />}

      {state.status === "ok" && state.result && (
        <div className="animate-fade-up rounded-lg border border-line bg-surface p-6">
          <div className="mb-5 flex items-center justify-between">
            <span className="text-xs uppercase tracking-wide text-faint">Emosi terdeteksi</span>
            <EmotionBadge emotion={state.result.label} size="lg" />
          </div>
          <ConfidenceBar value={state.result.confidence} emotion={state.result.label} />
          <Probabilities probabilities={state.result.probabilities} />
        </div>
      )}
    </div>
  );
}

export function Classify() {
  const [mode, setMode] = useState("single");

  return (
    <div className="animate-fade-up space-y-8">
      <div>
        <h1 className="font-display text-2xl text-ink">Coba Klasifikasi Emosi</h1>
        <p className="mt-1 text-sm text-muted">
          Tulis satu review, atau unggah banyak review sekaligus untuk melihat sebaran emosinya.
        </p>
      </div>

      <div className="flex gap-1" role="tablist" aria-label="Mode klasifikasi">
        {MODES.map((m) => (
          <button
            key={m.id}
            role="tab"
            aria-selected={mode === m.id}
            onClick={() => setMode(m.id)}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === m.id ? "bg-ink text-paper" : "text-faint hover:text-muted"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {mode === "single" ? <SingleClassify /> : <BatchClassify />}
    </div>
  );
}
