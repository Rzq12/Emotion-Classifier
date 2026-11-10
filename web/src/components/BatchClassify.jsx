import { useRef, useState } from "react";
import { api } from "../lib/api";
import { EMOTION_ORDER, emotionMeta } from "../lib/constants";
import { extractReviews, toCsv, MAX_BATCH, MAX_TEXT_LEN } from "../lib/csv";
import { EmotionBadge } from "./EmotionBadge";
import { Spinner, ErrorState } from "./States";

function download(results) {
  const blob = new Blob([toCsv(results)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "hasil-klasifikasi.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function SummaryBar({ counts, total }) {
  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-line">
        {EMOTION_ORDER.map((k) => {
          const w = total > 0 ? ((counts[k] || 0) / total) * 100 : 0;
          if (w === 0) return null;
          return <div key={k} style={{ width: `${w}%`, backgroundColor: emotionMeta(k).color }} />;
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
        {EMOTION_ORDER.map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: emotionMeta(k).color }}
              aria-hidden
            />
            {emotionMeta(k).label}
            <span className="font-mono tnum text-ink">{counts[k] || 0}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// Classify many reviews at once: upload a .csv/.txt or paste one review per line.
export function BatchClassify() {
  const fileRef = useRef(null);
  const [file, setFile] = useState(null); // { name, texts, column, tooLong, overLimit }
  const [pasted, setPasted] = useState("");
  const [state, setState] = useState({ status: "idle", data: null, error: null });

  const parsed = file ?? extractReviews(pasted);
  const { texts } = parsed;

  const onFile = (e) => {
    const picked = e.target.files?.[0];
    if (!picked) return;
    picked
      .text()
      .then((content) => {
        setFile({ name: picked.name, ...extractReviews(content, picked.name) });
        setState({ status: "idle", data: null, error: null });
      })
      .catch(() =>
        setState({ status: "error", data: null, error: "File tidak bisa dibaca sebagai teks." }),
      );
  };

  const clearFile = () => {
    setFile(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const submit = () => {
    if (texts.length === 0) return;
    setState({ status: "loading", data: null, error: null });
    api
      .classifyBatch(texts)
      .then((data) => setState({ status: "ok", data, error: null }))
      .catch((e) => setState({ status: "error", data: null, error: e.message }));
  };

  const loading = state.status === "loading";

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-line bg-surface p-5">
        <label htmlFor="batch-file" className="block text-sm font-medium text-ink">
          Unggah file review
        </label>
        <p className="mt-1 text-xs text-faint">
          .csv (kolom <span className="font-mono">text</span>, <span className="font-mono">review</span>,
          atau kolom pertama) atau .txt satu review per baris. Maksimal {MAX_BATCH} review, masing-masing
          hingga {MAX_TEXT_LEN.toLocaleString("id-ID")} karakter. File dibaca di browser Anda.
        </p>
        <input
          id="batch-file"
          ref={fileRef}
          type="file"
          accept=".csv,.txt,text/csv,text/plain"
          onChange={onFile}
          className="mt-3 block w-full text-sm text-muted file:mr-3 file:rounded-md file:border file:border-line file:bg-paper file:px-3 file:py-1.5 file:text-sm file:text-ink"
        />

        {file ? (
          <p className="mt-3 text-xs text-muted">
            <span className="font-mono text-ink">{file.name}</span>
            {file.column && <> — kolom “{file.column}”</>}{" "}
            <button type="button" onClick={clearFile} className="underline underline-offset-2">
              Hapus file
            </button>
          </p>
        ) : (
          <>
            <label htmlFor="batch-paste" className="mt-5 block text-sm font-medium text-ink">
              Atau tempel review
            </label>
            <textarea
              id="batch-paste"
              value={pasted}
              onChange={(e) => setPasted(e.target.value)}
              rows={5}
              placeholder={"Satu review per baris, contoh:\nAplikasi error terus saat bayar\nDokternya ramah sekali"}
              className="mt-1.5 w-full resize-y rounded-md border border-line bg-paper/60 p-3 text-sm text-ink placeholder:text-faint focus:border-ink"
            />
          </>
        )}

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs text-faint">
            {texts.length > 0 ? `${texts.length} review siap diklasifikasi.` : "Belum ada review."}
            {parsed.overLimit > 0 && ` ${parsed.overLimit} review setelah batas ${MAX_BATCH} dilewati.`}
            {parsed.tooLong > 0 && ` ${parsed.tooLong} review terlalu panjang dilewati.`}
          </span>
          <button
            type="button"
            onClick={submit}
            disabled={texts.length === 0 || loading}
            className="inline-flex items-center gap-2 rounded-md bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading && <Spinner />}
            {loading ? "Mengklasifikasi..." : `Klasifikasikan ${texts.length || ""} review`}
          </button>
        </div>
      </div>

      {state.status === "error" && <ErrorState message={state.error} onRetry={submit} />}

      {state.status === "ok" && state.data && (
        <div className="animate-fade-up space-y-5">
          <div className="rounded-lg border border-line bg-surface p-5">
            <div className="mb-3 flex items-center justify-between gap-3">
              <span className="text-xs uppercase tracking-wide text-faint">
                Ringkasan {state.data.count} review
              </span>
              <button
                type="button"
                onClick={() => download(state.data.results)}
                className="text-xs font-medium text-ink underline underline-offset-4 hover:no-underline"
              >
                Unduh CSV
              </button>
            </div>
            <SummaryBar counts={state.data.by_emotion} total={state.data.count} />
          </div>

          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full min-w-[480px] text-left text-sm">
              <thead className="bg-surface text-xs uppercase tracking-wide text-faint">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Review</th>
                  <th className="px-4 py-2.5 font-medium">Emosi</th>
                  <th className="px-4 py-2.5 text-right font-medium">Keyakinan</th>
                </tr>
              </thead>
              <tbody>
                {state.data.results.map((r, i) => (
                  <tr key={i} className="border-t border-line align-top">
                    <td className="px-4 py-2.5 text-ink">{r.text}</td>
                    <td className="whitespace-nowrap px-4 py-2.5">
                      <EmotionBadge emotion={r.label} />
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono tnum text-muted">
                      {Math.round(r.confidence * 100)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
