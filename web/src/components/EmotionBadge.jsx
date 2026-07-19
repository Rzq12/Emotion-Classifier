import { emotionMeta } from "../lib/constants";

// Color + text label + dot — never color alone (accessibility).
export function EmotionBadge({ emotion, size = "md" }) {
  const meta = emotionMeta(emotion);
  const pad = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2.5 py-1 text-xs";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${pad}`}
      style={{ borderColor: `${meta.color}40`, color: meta.color, backgroundColor: `${meta.color}10` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: meta.color }} aria-hidden />
      {meta.label}
    </span>
  );
}
