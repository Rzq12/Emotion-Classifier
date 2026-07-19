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
