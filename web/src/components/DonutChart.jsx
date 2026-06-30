import { emotionMeta, EMOTION_ORDER } from "../lib/constants";

// Hand-built SVG donut — distribution of emotions. Signature visual element.
export function DonutChart({ data, total }) {
  const size = 168;
  const stroke = 22;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;

  const segments = EMOTION_ORDER.map((key) => ({ key, value: data[key] || 0 })).filter(
    (s) => s.value > 0,
  );

  let offset = 0;
  const arcs = segments.map((s) => {
    const fraction = total > 0 ? s.value / total : 0;
    const dash = fraction * circumference;
    const arc = {
      key: s.key,
      color: emotionMeta(s.key).color,
      dasharray: `${dash} ${circumference - dash}`,
      dashoffset: -offset,
    };
    offset += dash;
    return arc;
  });

  const summary = segments
    .map((s) => `${emotionMeta(s.key).label} ${Math.round((s.value / total) * 100)}%`)
    .join(", ");

  return (
    <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-center sm:gap-10">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={`Distribusi emosi: ${summary}`}
        className="shrink-0"
      >
        <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
          <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#e4e1d8" strokeWidth={stroke} />
          {arcs.map((a) => (
            <circle
              key={a.key}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={a.color}
              strokeWidth={stroke}
              strokeDasharray={a.dasharray}
              strokeDashoffset={a.dashoffset}
              strokeLinecap="butt"
            />
          ))}
        </g>
        <text x="50%" y="46%" textAnchor="middle" className="fill-ink font-display tnum" fontSize="30">
          {total.toLocaleString("id-ID")}
        </text>
        <text x="50%" y="60%" textAnchor="middle" className="fill-faint" fontSize="11" letterSpacing="1">
          REVIEW
        </text>
      </svg>

      <ul className="w-full space-y-2.5">
        {EMOTION_ORDER.map((key) => {
          const meta = emotionMeta(key);
          const value = data[key] || 0;
          const pct = total > 0 ? Math.round((value / total) * 100) : 0;
          return (
            <li key={key} className="flex items-center gap-3">
              <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: meta.color }} aria-hidden />
              <span className="w-16 text-sm text-ink">{meta.label}</span>
              <span className="font-mono text-sm tnum text-muted">{value.toLocaleString("id-ID")}</span>
              <span className="ml-auto font-mono text-sm tnum text-ink">{pct}%</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
