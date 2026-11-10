import { useState } from "react";
import { api } from "../lib/api";
import { EmotionBadge } from "./EmotionBadge";
import { Spinner } from "./States";

const MAX_IDS = 50; // matches the /reviews/lookup request limit

// Expandable list of the actual reviews behind cited IDs, fetched on first open
// so a reader can verify grounding instead of trusting an opaque identifier.
export function ReviewSources({ ids, label }) {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState({ status: "idle", reviews: [], missing: [], error: null });

  if (!ids?.length) return null;

  const load = () => {
    setState({ status: "loading", reviews: [], missing: [], error: null });
    api
      .reviewsLookup(ids.slice(0, MAX_IDS))
      .then((res) =>
        setState({ status: "ok", reviews: res.reviews, missing: res.missing, error: null }),
      )
      .catch((e) => setState({ status: "error", reviews: [], missing: [], error: e.message }));
  };

  const toggle = () => {
    if (!open && state.status === "idle") load();
    setOpen((o) => !o);
  };

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="text-xs text-faint underline underline-offset-2 hover:text-muted"
      >
        {label ?? `Sumber: ${ids.length} review`}
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {state.status === "loading" && (
            <p className="flex items-center gap-2 text-xs text-faint">
              <Spinner className="h-3 w-3" /> Memuat review...
            </p>
          )}

          {state.status === "error" && (
            <p className="text-xs text-anger">
              {state.error}{" "}
              <button type="button" onClick={load} className="underline underline-offset-2">
                Coba lagi
              </button>
            </p>
          )}

          {state.status === "ok" &&
            state.reviews.map((r) => (
              <div key={r.review_id} className="rounded-md border border-line bg-paper/60 p-3">
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <EmotionBadge emotion={r.emotion} />
                  <span className="font-mono text-xs text-faint">{r.review_id}</span>
                </div>
                <p className="text-sm not-italic text-muted">{r.text}</p>
              </div>
            ))}

          {state.status === "ok" && state.missing.length > 0 && (
            <p className="text-xs text-faint">
              {state.missing.length} sumber tidak ada di korpus: {state.missing.join(", ")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
