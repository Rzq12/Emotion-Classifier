import { useEffect, useState } from "react";
import { api } from "./lib/api";
import { Header } from "./components/Header";
import { Dashboard } from "./pages/Dashboard";
import { Classify } from "./pages/Classify";
import { Insight } from "./pages/Insight";
import { Chat } from "./pages/Chat";

const PAGES = {
  dashboard: Dashboard,
  classify: Classify,
  insight: Insight,
  chat: Chat,
};

export default function App() {
  const [tab, setTab] = useState("dashboard");
  const [health, setHealth] = useState(undefined);

  useEffect(() => {
    let active = true;
    const check = () =>
      api
        .health()
        .then((h) => active && setHealth(h))
        .catch(() => active && setHealth(null));
    check();
    const id = setInterval(check, 30000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const Page = PAGES[tab];

  return (
    <div className="min-h-dvh">
      <Header active={tab} onChange={setTab} health={health} />
      <main className="mx-auto max-w-5xl px-5 py-10">
        <Page key={tab} />
      </main>
      <footer className="mx-auto max-w-5xl px-5 pb-10 pt-6 text-xs text-faint">
        Indo Review Intelligence — emotion classifier + RAG insight. Bahasa Indonesia.
      </footer>
    </div>
  );
}
