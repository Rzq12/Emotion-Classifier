import { TABS } from "../lib/constants";

function HealthDot({ health }) {
  const online = health?.status === "ok";
  const label = health === undefined ? "Memeriksa..." : online ? "API aktif" : "API tidak terjangkau";
  const color = health === undefined ? "#9b988f" : online ? "#2f8a6a" : "#bf3b30";
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted" title={label}>
      <span className="relative flex h-2 w-2">
        {online && (
          <span
            className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
            style={{ backgroundColor: color }}
          />
        )}
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      </span>
      <span className="hidden sm:inline">{label}</span>
    </span>
  );
}

export function Header({ active, onChange, health }) {
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-paper/85 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-5 pt-4">
        <div className="flex items-baseline gap-2.5">
          <span className="font-display text-lg font-semibold tracking-tight text-ink">
            Indo&nbsp;Review
          </span>
          <span className="text-xs uppercase tracking-[0.2em] text-faint">Intelligence</span>
        </div>
        <HealthDot health={health} />
      </div>

      <nav className="mx-auto flex max-w-5xl gap-1 px-3" aria-label="Navigasi utama">
        {TABS.map((tab) => {
          const isActive = active === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onChange(tab.id)}
              aria-current={isActive ? "page" : undefined}
              className={`relative px-3 py-3 text-sm font-medium transition-colors ${
                isActive ? "text-ink" : "text-faint hover:text-muted"
              }`}
            >
              {tab.label}
              {isActive && <span className="absolute inset-x-3 -bottom-px h-0.5 bg-ink" />}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
