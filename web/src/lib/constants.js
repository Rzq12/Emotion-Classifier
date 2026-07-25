// Emotion semantics shared across the UI. Color is always paired with a label.
export const EMOTIONS = {
  anger: { label: "Marah", color: "#bf3b30", tw: "anger" },
  sadness: { label: "Sedih", color: "#3f6694", tw: "sadness" },
  happiness: { label: "Senang", color: "#2f8a6a", tw: "happiness" },
};

export const EMOTION_ORDER = ["happiness", "sadness", "anger"];

export function emotionMeta(key) {
  return EMOTIONS[key] || { label: key, color: "#6c6a63", tw: "muted" };
}

export const TABS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "classify", label: "Coba Klasifikasi" },
  { id: "insight", label: "Insight" },
  { id: "chat", label: "Tanya Data" },
  { id: "agent", label: "Agent Queue" },
];

// Router branch semantics for the agent queue (color always paired with label).
export const BRANCHES = {
  escalate: { label: "Eskalasi", color: "#bf3b30" },
  draft: { label: "Draft", color: "#3f6694" },
  archive: { label: "Arsip", color: "#6c6a63" },
};

export function branchMeta(key) {
  return BRANCHES[key] || { label: key, color: "#6c6a63" };
}

export const STATUS_LABELS = {
  pending: "Menunggu",
  approved: "Disetujui",
  rejected: "Ditolak",
};

export const AGENT_EXAMPLES = [
  "Aplikasi selalu error saat mau bayar, uang saya sudah kepotong tapi pesanan gagal!",
  "Terima kasih, dokternya ramah dan konsultasinya sangat membantu.",
  "Sudah seminggu tiket saya belum direspons, kecewa dengan layanannya.",
];

export const CLASSIFY_EXAMPLES = [
  "Aplikasinya sering error pas mau bayar, kecewa banget",
  "Dokternya ramah dan sangat membantu, terima kasih Halodoc",
  "Sudah bayar tapi pesanan tidak diproses, kesal sekali",
  "Pengiriman obat cepat, pelayanan memuaskan",
];

export const CHAT_SUGGESTIONS = [
  "Apa keluhan utama soal pembayaran?",
  "Kenapa pengguna merasa kecewa?",
  "Hal apa yang paling disukai pengguna?",
];
