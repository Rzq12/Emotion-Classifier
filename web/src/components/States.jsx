export function Spinner({ className = "" }) {
  return (
    <span
      className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent ${className}`}
      role="status"
      aria-label="Memuat"
    />
  );
}

export function LoadingBlock({ label = "Memuat data..." }) {
  return (
    <div className="flex items-center gap-3 py-12 text-muted">
      <Spinner />
      <span className="text-sm">{label}</span>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="rounded-md border border-anger/30 bg-anger/5 px-5 py-6">
      <p className="text-sm text-ink">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 text-sm font-medium text-anger underline underline-offset-4 hover:no-underline"
        >
          Coba lagi
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, hint }) {
  return (
    <div className="rounded-md border border-dashed border-line px-5 py-10 text-center">
      <p className="font-medium text-ink">{title}</p>
      {hint && <p className="mt-1 text-sm text-muted">{hint}</p>}
    </div>
  );
}
