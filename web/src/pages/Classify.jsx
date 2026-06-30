import { useState } from "react";
import { api } from "../lib/api";
import { CLASSIFY_EXAMPLES } from "../lib/constants";
import { EmotionBadge } from "../components/EmotionBadge";
import { ConfidenceBar } from "../components/ConfidenceBar";
import { Spinner, ErrorState } from "../components/States";

const MAX_LEN = 2000;

export function Classify() {
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
    <div className="animate-fade-up space-y-8">
      <div>
        <h1 className="font-display text-2xl text-ink">Coba Klasifikasi Emosi</h1>
        <p className="mt-1 text-sm text-muted">
          Tulis atau tempel sebuah review, lihat emosi yang terdeteksi secara langsung.
        </p>
      </div>

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
        </div>
      )}
    </div>
  );
}
