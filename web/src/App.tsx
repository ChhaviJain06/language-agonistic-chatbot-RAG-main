import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./index.css";
import ChatPane from "./components/ChatPane";
import CitationDrawer from "./components/CitationDrawer";
import type { ComposerHandle } from "./components/Composer";
import ErrorBoundary from "./components/ErrorBoundary";
import Header from "./components/Header";
import Toasts from "./components/Toasts";
import VisualStage from "./components/VisualStage";
import VoicePanel from "./components/VoicePanel";
import { useChat } from "./hooks/useChat";
import { useHealth } from "./hooks/useHealth";
import { useTheme } from "./hooks/useTheme";
import { useToasts } from "./hooks/useToasts";
import { useVoice } from "./hooks/useVoice";
import { IconClose } from "./lib/icons";
import { LANGS, type DrawerItem, type VisualStageData } from "./lib/types";

const WIDE = "(min-width: 1024px)";

export default function App() {
  const { theme, toggle } = useTheme();
  const { health, down, checking, retry } = useHealth();
  const { toasts, push, dismiss } = useToasts();
  const [lang, setLang] = useState("auto");
  const [visual, setVisual] = useState<VisualStageData | null>(null);
  const [stageOpen, setStageOpen] = useState(() => window.matchMedia(WIDE).matches);
  const [drawer, setDrawer] = useState<DrawerItem | null>(null);
  const composer = useRef<ComposerHandle>(null);

  const langCode = LANGS.find((l) => l.value === lang)?.code ?? null;
  const onError = useCallback((m: string) => push(m, "error"), [push]);

  const chat = useChat({ lang: langCode, onError, onVisual: setVisual });
  const { appendVoice } = chat;
  const voice = useVoice({ onError, onVisual: setVisual, onMessage: appendVoice });

  const latestSources = useMemo(() => {
    for (let i = chat.messages.length - 1; i >= 0; i--) {
      const m = chat.messages[i];
      if (m.role === "assistant" && m.sources?.length) return m.sources;
    }
    return [];
  }, [chat.messages]);

  const { busy, stop } = chat;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing = !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);
      if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        composer.current?.focus();
      } else if (e.key === "Escape") {
        if (drawer) setDrawer(null);
        else if (busy) stop();
        else if (!window.matchMedia(WIDE).matches) setStageOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawer, busy, stop]);

  return (
    <div className="app">
      <Header
        health={health}
        down={down}
        checking={checking}
        lang={lang}
        onLang={setLang}
        theme={theme}
        onTheme={toggle}
        stageOpen={stageOpen}
        onStage={() => setStageOpen((o) => !o)}
        onClear={chat.clear}
        hasMessages={chat.messages.length > 0}
      />

      {down && (
        <div className="banner" role="alert">
          <span>
            Cannot reach the BHASHA backend. Make sure the server is running on port 8000 (it is checked every 15 s).
          </span>
          <button type="button" className="btn ghost sm" onClick={retry} disabled={checking}>
            {checking ? "Checking…" : "Retry now"}
          </button>
        </div>
      )}

      <main className={`main${stageOpen ? " with-stage" : ""}`}>
        <ChatPane
          messages={chat.messages}
          busy={chat.busy}
          composerRef={composer}
          onSend={(q) => void chat.send(q)}
          onStop={chat.stop}
          onRegenerate={chat.regenerate}
          onOpenSource={setDrawer}
          voice={
            <VoicePanel
              state={voice.state}
              level={voice.level}
              userText={voice.userText}
              agentText={voice.agentText}
              interrupted={voice.interrupted}
              onConnect={() => void voice.connect()}
              onDisconnect={voice.disconnect}
            />
          }
        />

        <aside
          className={`stage-pane${stageOpen ? " open" : ""}`}
          aria-label="Visual Stage"
          aria-hidden={!stageOpen}
          inert={!stageOpen}
        >
          <button type="button" className="sheet-close icon-btn" onClick={() => setStageOpen(false)} aria-label="Close Visual Stage">
            <IconClose />
          </button>
          <div className="sheet-grip" aria-hidden="true" />
          <ErrorBoundary label="Visual Stage" key={visual ? visual.entity ?? "x" : "none"}>
            <VisualStage data={visual} sources={latestSources} theme={theme} onOpen={setDrawer} />
          </ErrorBoundary>
        </aside>
      </main>

      {!stageOpen && (
        <button type="button" className="stage-fab" onClick={() => setStageOpen(true)}>
          Visual Stage
        </button>
      )}

      <CitationDrawer item={drawer} onClose={() => setDrawer(null)} />
      <Toasts toasts={toasts} onDismiss={dismiss} />
    </div>
  );
}
