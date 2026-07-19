const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:7860").replace(/\/$/, "");

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError("Tidak dapat terhubung ke server. Periksa koneksi atau coba lagi.", 0);
  }

  if (res.status === 429) {
    throw new ApiError("Terlalu banyak permintaan. Coba lagi dalam beberapa saat.", 429);
  }
  if (!res.ok) {
    let detail = `Permintaan gagal (${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail && typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  stats: () => request("/stats"),
  classify: (text) => request("/classify", { method: "POST", body: JSON.stringify({ text }) }),
  insight: (query) => request("/insight", { method: "POST", body: JSON.stringify({ query }) }),
  chat: (question, history = []) =>
    request("/chat", { method: "POST", body: JSON.stringify({ question, history }) }),
};

export { ApiError, BASE_URL };
