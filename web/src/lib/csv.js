// CSV helpers for batch classification. Files are parsed in the browser so the
// API only ever receives a plain list of review texts.

export const MAX_BATCH = 100;
export const MAX_TEXT_LEN = 2000;

// Header names recognised as the review-text column (case-insensitive).
const TEXT_COLUMNS = ["text", "review", "content", "ulasan", "komentar", "teks"];

// Minimal RFC 4180 parser: quoted fields, escaped quotes (""), CRLF or LF rows.
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch !== '"') field += ch;
      else if (text[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = false;
    } else if (ch === '"' && field === "") {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

// Pull review texts out of an uploaded file or pasted text.
// .csv: uses a recognised header column if present, otherwise the first column.
// Anything else: one review per line.
// Returns the usable texts plus counts of what was left out, so the UI can say so.
export function extractReviews(content, fileName = "") {
  // Drop a leading UTF-8 byte-order mark (common in Excel-exported CSVs).
  const clean = content.charCodeAt(0) === 0xfeff ? content.slice(1) : content;
  let candidates;
  let column = null;

  if (fileName.toLowerCase().endsWith(".csv")) {
    const rows = parseCsv(clean).filter((r) => r.some((cell) => cell.trim()));
    const header = (rows[0] || []).map((cell) => cell.trim().toLowerCase());
    const idx = header.findIndex((cell) => TEXT_COLUMNS.includes(cell));
    if (idx >= 0) {
      column = rows[0][idx].trim();
      candidates = rows.slice(1).map((r) => r[idx] ?? "");
    } else {
      candidates = rows.map((r) => r[0] ?? "");
    }
  } else {
    candidates = clean.split(/\r?\n/);
  }

  const nonEmpty = candidates.map((t) => t.trim()).filter(Boolean);
  const valid = nonEmpty.filter((t) => t.length <= MAX_TEXT_LEN);
  return {
    texts: valid.slice(0, MAX_BATCH),
    column,
    tooLong: nonEmpty.length - valid.length,
    overLimit: Math.max(0, valid.length - MAX_BATCH),
  };
}

function escapeCell(value) {
  const s = String(value);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

// Serialise batch results as a CSV the user can download.
export function toCsv(results) {
  const lines = [["text", "label", "confidence"]];
  for (const r of results) lines.push([r.text, r.label, r.confidence]);
  return lines.map((cells) => cells.map(escapeCell).join(",")).join("\r\n");
}
