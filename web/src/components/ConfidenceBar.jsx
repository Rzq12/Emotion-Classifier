import { useEffect, useState } from "react";
import { emotionMeta } from "../lib/constants";

export function ConfidenceBar({ value, emotion }) {
  const meta = emotionMeta(emotion);
  const pct = Math.round(value * 100);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const t = requestAnimationFrame(() => setWidth(pct));
    return () => cancelAnimationFrame(t);
  }, [pct]);

  return (
    <div className="w-full">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wide text-faint">Keyakinan</span>
        <span className="font-mono text-sm tnum text-ink">{pct}%</span>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-line"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="h-full rounded-full transition-[width] duration-700 ease-out"
          style={{ width: `${width}%`, backgroundColor: meta.color }}
        />
      </div>
    </div>
  );
}
