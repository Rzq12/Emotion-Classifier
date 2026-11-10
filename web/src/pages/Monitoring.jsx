import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { EMOTION_ORDER, emotionMeta } from "../lib/constants";
import { LoadingBlock, ErrorState, EmptyState } from "../components/States";

// Verdict text is always shown; color only reinforces it. Literal class names
// because Tailwind purges dynamically-built class strings.
const VERDICTS = {
  STABLE: { label: "Stabil", tone: "text-happiness" },
  MODERATE: { label: "Bergeser moderat", tone: "text-ink" },
  DRIFT: { label: "Drift signifikan", tone: "text-anger" },
};

const pct = (value) => `${Math.round((value || 0) * 100)}%`;

function Stat({ label, value, sub }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-faint">{label}</p>
      <p className="mt-1 font-display text-3xl tnum text-ink">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </div>
  );
}

function PsiRow({ name, psi, verdict }) {
  const meta = VERDICTS[verdict] || { label: verdict, tone: "text-muted" };
  return (
    <div className="flex items-baseline justify-between gap-3 border-t border-line py-3 first:border-t-0">
      <span className="text-sm text-ink">{name}</span>
      <span className="flex items-baseline gap-3">
        <span className="font-mono text-xs tnum text-faint">PSI {psi.toFixed(3)}</span>
        <span className={`text-sm font-medium ${meta.tone}`}>{meta.label}</span>
      </span>
    </div>
  );
}

function DriftCard({ drift, n }) {
  let body;
  if (drift.status === "ok") {
    body = (
      <>
        <PsiRow name="Distribusi label" psi={drift.label_psi} verdict={drift.label_verdict} />
        <PsiRow name="Panjang teks" psi={drift.length_psi} verdict={drift.length_verdict} />
        <p className="pt-2 text-xs text-faint">
          PSI di bawah 0,1 stabil; 0,1–0,2 bergeser moderat; 0,2 ke atas drift signifikan.
        </p>
      </>
    );
  } else if (drift.status === "insufficient_data") {
    body = (
      <p className="text-sm text-muted">
        Verdict drift belum dihitung: baru {n} dari minimal {drift.min_samples} prediksi. PSI atas
        sampel sekecil ini tidak bermakna.
      </p>
    );
  } else {
    body = (
      <p className="text-sm text-muted">
        Data referensi (training) tidak tersedia di server, jadi drift tidak bisa dihitung.
      </p>
    );
  }
  return (
    <section className="rounded-lg border border-line bg-surface p-6">
      <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-faint">
        Status drift vs data training
      </h2>
      {body}
    </section>
  );
}

function CompareBar({ label, value, color, muted }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-20 shrink-0 text-xs text-faint">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-line">
        <div
          className="h-full rounded-full"
          style={{ width: pct(value), backgroundColor: color, opacity: muted ? 0.4 : 1 }}
        />
      </div>
      <span className="w-10 shrink-0 text-right font-mono text-xs tnum text-ink">{pct(value)}</span>
    </div>
  );
}

function DailyVolume({ daily }) {
  const max = Math.max(...daily.map((d) => d.count), 1);
  return (
    <div className="space-y-2">
      {daily.map((d) => (
        <div key={d.date} className="flex items-center gap-3">
          <span className="w-24 shrink-0 font-mono text-xs text-faint">{d.date}</span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-line">
            <div className="h-full rounded-full bg-ink" style={{ width: `${(d.count / max) * 100}%` }} />
          </div>
          <span className="w-10 shrink-0 text-right font-mono text-xs tnum text-ink">{d.count}</span>
        </div>
      ))}
    </div>
  );
}

export function Monitoring() {
  const [state, setState] = useState({ status: "loading", data: null, error: null });

  const load = () => {
    setState({ status: "loading", data: null, error: null });
    api
      .monitoringSummary()
      .then((data) => setState({ status: "ok", data, error: null }))
      .catch((e) => setState({ status: "error", data: null, error: e.message }));
  };

  useEffect(load, []);

  if (state.status === "loading") return <LoadingBlock label="Memuat ringkasan monitoring..." />;
  if (state.status === "error") return <ErrorState message={state.error} onRetry={load} />;

  const d = state.data;
  const hasReference = Object.keys(d.reference_distribution).length > 0;

  return (
    <div className="animate-fade-up space-y-10">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl text-ink">Monitoring Prediksi</h1>
          <p className="mt-1 text-sm text-muted">
            Ringkasan prediksi yang dilayani <span className="font-mono">/classify</span>, dibandingkan
            dengan data training untuk mendeteksi drift.
          </p>
        </div>
        <button onClick={load} className="shrink-0 text-xs text-faint underline-offset-4 hover:underline">
          Muat ulang
        </button>
      </div>

      {d.n_predictions === 0 ? (
        <EmptyState
          title="Belum ada prediksi tercatat"
          hint="Jalankan klasifikasi di tab Coba Klasifikasi; hasilnya muncul di sini. Log di server demo terhapus saat Space restart."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-6 border-y border-line py-6 sm:grid-cols-3">
            <Stat label="Prediksi tercatat" value={d.n_predictions.toLocaleString("id-ID")} />
            <Stat
              label="Rata-rata keyakinan"
              value={d.mean_confidence == null ? "–" : pct(d.mean_confidence)}
            />
            <Stat
              label="Keyakinan rendah"
              value={d.low_confidence_share == null ? "–" : pct(d.low_confidence_share)}
              sub={`di bawah ${pct(d.low_confidence_threshold)}`}
            />
          </div>

          <DriftCard drift={d.drift} n={d.n_predictions} />

          <section>
            <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-faint">
              Distribusi emosi: prediksi live vs training
            </h2>
            <div className="space-y-5">
              {EMOTION_ORDER.map((k) => (
                <div key={k}>
                  <p className="mb-1.5 text-sm text-ink">{emotionMeta(k).label}</p>
                  <div className="space-y-1.5">
                    <CompareBar label="Live" value={d.label_distribution[k]} color={emotionMeta(k).color} />
                    {hasReference && (
                      <CompareBar
                        label="Training"
                        value={d.reference_distribution[k]}
                        color={emotionMeta(k).color}
                        muted
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {d.daily.length > 0 && (
            <section>
              <h2 className="mb-4 text-sm font-medium uppercase tracking-wide text-faint">
                Volume prediksi per hari (UTC)
              </h2>
              <DailyVolume daily={d.daily} />
            </section>
          )}
        </>
      )}
    </div>
  );
}
