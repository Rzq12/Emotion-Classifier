import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { CHAT_SUGGESTIONS } from "../lib/constants";
import { Spinner } from "../components/States";

function Sources({ ids }) {
  const [open, setOpen] = useState(false);
  if (!ids?.length) return null;
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-xs text-faint underline underline-offset-2 hover:text-muted"
        aria-expanded={open}
      >
        Sumber: {ids.length} review terkait
      </button>
      {open && (
        <p className="mt-1 font-mono text-xs text-faint">{ids.join(", ")}</p>
      )}
    </div>
  );
}

function Bubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "rounded-br-sm bg-ink text-paper"
            : "rounded-bl-sm border border-line bg-surface text-ink"
        }`}
      >
        {msg.error ? <span className="text-anger">{msg.content}</span> : msg.content}
        {!isUser && !msg.error && <Sources ids={msg.sources} />}
      </div>
    </div>
  );
}

export function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = (value) => {
    const question = (value ?? input).trim();
    if (!question || loading) return;
    // Last 6 turns as context so follow-up questions stay coherent.
    const history = messages
      .filter((m) => !m.error)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));
    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    api
      .chat(question, history)
      .then((res) =>
        setMessages((m) => [...m, { role: "bot", content: res.answer, sources: res.sources }]),
      )
      .catch((e) => setMessages((m) => [...m, { role: "bot", content: e.message, error: true }]))
      .finally(() => setLoading(false));
  };

  return (
    <div className="animate-fade-up flex h-[calc(100vh-220px)] min-h-[420px] flex-col">
      <div>
        <h1 className="font-display text-2xl text-ink">Tanya Data Review</h1>
        <p className="mt-1 text-sm text-muted">
          Pertanyaan dijawab berdasarkan review nyata — bukan pengetahuan umum.
        </p>
      </div>

      <div className="mt-6 flex-1 space-y-4 overflow-y-auto pr-1">
        {messages.length === 0 && (
          <div className="space-y-4">
            <p className="text-sm text-muted">Halo! Tanyakan apa saja soal review pengguna.</p>
            <div className="flex flex-wrap gap-2">
              {CHAT_SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-full border border-line bg-surface px-3 py-1.5 text-xs text-muted transition-colors hover:border-ink hover:text-ink"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <Bubble key={i} msg={msg} />
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-sm text-faint">
            <Spinner className="text-muted" /> sedang mengetik...
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        className="mt-4 flex items-center gap-2 border-t border-line pt-4"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ketik pertanyaan..."
          className="flex-1 rounded-md border border-line bg-surface px-4 py-2.5 text-sm text-ink placeholder:text-faint focus:border-ink"
        />
        <button
          type="submit"
          disabled={!input.trim() || loading}
          className="rounded-md bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
        >
          Kirim
        </button>
      </form>
    </div>
  );
}
