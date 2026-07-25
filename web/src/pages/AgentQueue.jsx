import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { AGENT_EXAMPLES, branchMeta, STATUS_LABELS } from "../lib/constants";
import { EmotionBadge } from "../components/EmotionBadge";
import { Spinner, ErrorState, EmptyState, LoadingBlock } from "../components/States";

const FILTERS = [
  { id: "", label: "Semua" },
  { id: "pending", label: "Menunggu" },
  { id: "approved", label: "Disetujui" },
  { id: "rejected", label: "Ditolak" },
];

function BranchBadge({ branch, priority }) {
  const meta = branchMeta(branch);
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium"
      style={{ borderColor: `${meta.color}40`, color: meta.color, backgroundColor: `${meta.color}10` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: meta.color }} aria-hidden />
      {meta.label}
      {priority === "high" && <span className="font-mono">• prioritas tinggi</span>}
    </span>
  );
}

// Literal class names (Tailwind purges dynamically-built class strings).
const STATUS_TONE = {
  approved: "text-happiness",
  rejected: "text-anger",
  pending: "text-faint",
};

function StatusPill({ status }) {
  return (
    <span className={`font-mono text-xs uppercase tracking-wide ${STATUS_TONE[status] || "text-faint"}`}>
      {STATUS_LABELS[status] || status}
    </span>
  );
}

function StatStrip({ stats }) {
  if (!stats) return null;
  const cells = [
    { label: "Total tiket", value: stats.total },
    { label: "Menunggu", value: stats.pending },
    { label: "Eskalasi", value: stats.escalations },
    { label: "Approval rate", value: `${Math.round(stats.approval_rate * 100)}%` },
  ];
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-4">
      {cells.map((c) => (
        <div key={c.label} className="bg-surface px-4 py-3">
          <div className="font-mono text-xl tnum text-ink">{c.value}</div>
          <div className="text-xs text-faint">{c.label}</div>
        </div>
      ))}
    </div>
  );
}

function TicketCard({ ticket, onApprove, onReject, busy }) {
  const isPending = ticket.status === "pending";
  return (
    <div className="animate-fade-up rounded-lg border border-line bg-surface p-5">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <EmotionBadge emotion={ticket.label} />
        <BranchBadge branch={ticket.branch} priority={ticket.priority} />
        <span className="font-mono text-xs tnum text-faint">
          {Math.round(ticket.confidence * 100)}%
        </span>
        <span className="ml-auto flex items-center gap-3">
          <StatusPill status={ticket.status} />
          <span className="font-mono text-xs text-faint">#{ticket.id}</span>
        </span>
      </div>

      <p className="text-sm text-ink">{ticket.review_text}</p>

      {ticket.draft && (
        <div className="mt-3 rounded-md border border-line bg-paper/60 p-3">
          <div className="mb-1 text-xs uppercase tracking-wide text-faint">Draft balasan</div>
          <p className="text-sm text-muted">{ticket.draft}</p>
          {ticket.grounding_ids?.length > 0 && (
            <p className="mt-2 font-mono text-xs text-faint">
              grounding: {ticket.grounding_ids.join(", ")}
            </p>
          )}
        </div>
      )}

      {ticket.reason && (
        <p className="mt-2 text-xs italic text-anger">Alasan ditolak: {ticket.reason}</p>
      )}

      {isPending && (
        <div className="mt-4 flex gap-2">
          <button
            onClick={() => onApprove(ticket.id)}
            disabled={busy}
            className="inline-flex items-center gap-2 rounded-md bg-ink px-4 py-2 text-sm font-medium text-paper transition-opacity disabled:opacity-40"
          >
            {busy && <Spinner />}
            Setujui
          </button>
          <button
            onClick={() => onReject(ticket.id)}
            disabled={busy}
            className="rounded-md border border-line px-4 py-2 text-sm font-medium text-muted transition-colors hover:border-anger hover:text-anger disabled:opacity-40"
          >
            Tolak
          </button>
        </div>
      )}
    </div>
  );
}

export function AgentQueue() {
  const [filter, setFilter] = useState("");
  const [tickets, setTickets] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);

  const [review, setReview] = useState("");
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);

  const load = useCallback(() => {
    setError(null);
    Promise.all([api.agentQueue(filter || undefined), api.agentStats()])
      .then(([q, s]) => {
        setTickets(q.tickets);
        setStats(s);
      })
      .catch((e) => setError(e.message));
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  const runAgent = (value) => {
    const text = (value ?? review).trim();
    if (!text) return;
    setRunning(true);
    setRunError(null);
    api
      .agentRun(text)
      .then(() => {
        setReview("");
        load();
      })
      .catch((e) => setRunError(e.message))
      .finally(() => setRunning(false));
  };

  const approve = (id) => {
    setBusyId(id);
    api
      .agentApprove(id)
      .then(load)
      .catch((e) => setError(e.message))
      .finally(() => setBusyId(null));
  };

  const reject = (id) => {
    const reason = window.prompt("Alasan menolak draft ini? (opsional)") ?? "";
    setBusyId(id);
    api
      .agentReject(id, reason)
      .then(load)
      .catch((e) => setError(e.message))
      .finally(() => setBusyId(null));
  };

  return (
    <div className="animate-fade-up space-y-8">
      <div>
        <h1 className="font-display text-2xl text-ink">Agent Queue</h1>
        <p className="mt-1 text-sm text-muted">
          Agent mengklasifikasi review, memutuskan tindakan (eskalasi / draft / arsip), lalu
          menyimpan draft untuk ditinjau manusia sebelum dikirim.
        </p>
      </div>

      {/* Demo: run the agent on a review */}
      <div className="rounded-lg border border-line bg-surface p-5">
        <div className="mb-3 flex flex-wrap gap-2">
          {AGENT_EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => {
                setReview(ex);
                runAgent(ex);
              }}
              className="rounded-full border border-line bg-paper/60 px-3 py-1.5 text-xs text-muted transition-colors hover:border-ink hover:text-ink"
            >
              {ex.length > 46 ? `${ex.slice(0, 46)}…` : ex}
            </button>
          ))}
        </div>
        <textarea
          value={review}
          onChange={(e) => setReview(e.target.value)}
          rows={3}
          placeholder="Tempel review pengguna, lalu jalankan agent..."
          className="w-full resize-none rounded-md border border-line bg-paper/60 p-3 text-sm text-ink placeholder:text-faint focus:border-ink"
        />
        <div className="mt-2 flex items-center justify-between">
          {runError ? (
            <span className="text-xs text-anger">{runError}</span>
          ) : (
            <span className="text-xs text-faint">Memanggil classifier + RAG + LLM.</span>
          )}
          <button
            onClick={() => runAgent()}
            disabled={!review.trim() || running}
            className="inline-flex items-center gap-2 rounded-md bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running && <Spinner />}
            {running ? "Memproses..." : "Jalankan Agent"}
          </button>
        </div>
      </div>

      <StatStrip stats={stats} />

      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                filter === f.id ? "bg-ink text-paper" : "text-faint hover:text-muted"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <button onClick={load} className="text-xs text-faint underline-offset-4 hover:underline">
          Muat ulang
        </button>
      </div>

      {error && <ErrorState message={error} onRetry={load} />}

      {!error && tickets === null && <LoadingBlock label="Memuat antrean..." />}

      {!error && tickets?.length === 0 && (
        <EmptyState
          title="Antrean kosong"
          hint="Jalankan agent pada sebuah review untuk mengisi antrean."
        />
      )}

      {!error && tickets?.length > 0 && (
        <div className="space-y-4">
          {tickets.map((t) => (
            <TicketCard
              key={t.id}
              ticket={t}
              onApprove={approve}
              onReject={reject}
              busy={busyId === t.id}
            />
          ))}
        </div>
      )}
    </div>
  );
}
