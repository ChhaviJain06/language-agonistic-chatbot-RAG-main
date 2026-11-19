import { useCallback, useEffect, useRef, useState } from "react";
import { ServerError, postQuery, streamQuery } from "../lib/api";
import { buildHistory, stripTags } from "../lib/text";
import type { ChatMessage, DoneEvent, VisualStageData } from "../lib/types";

interface Opts {
  lang: string | null;
  onError: (msg: string) => void;
  onVisual: (v: VisualStageData | null) => void;
}

let counter = 0;
const uid = (p: string) => `${p}-${Date.now()}-${++counter}`;

export function useChat({ lang, onError, onVisual }: Opts) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const acRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  useEffect(() => () => acRef.current?.abort(), []);

  const patch = useCallback((id: string, fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }, []);

  const finalize = useCallback(
    (id: string, d: DoneEvent) => {
      patch(id, (m) => ({
        ...m,
        text: d.answer ?? m.text,
        streaming: false,
        parts: undefined,
        lang: d.lang ?? m.lang,
        script: d.script ?? m.script,
        sources: d.sources ?? [],
        fidelity: d.fidelity_passed,
        cached: d.cached,
        timing: d.timing ?? null,
        visual: d.visual_stage ?? null,
      }));
      if (d.visual_stage) onVisual(d.visual_stage);
    },
    [patch, onVisual],
  );

  const send = useCallback(
    async (question: string, historyOverride?: ChatMessage[]) => {
      const q = question.trim();
      if (!q || acRef.current) return;
      const history = buildHistory(historyOverride ?? messagesRef.current);
      const aid = uid("a");
      setMessages((ms) => [
        ...(historyOverride ?? ms),
        { id: uid("u"), role: "user", text: q, ts: Date.now() },
        { id: aid, role: "assistant", text: "", ts: Date.now(), streaming: true },
      ]);
      const ac = new AbortController();
      acRef.current = ac;
      setBusy(true);
      const body = { question: q, history, ...(lang ? { lang } : {}) };
      try {
        let done: DoneEvent;
        try {
          done = await streamQuery(
            body,
            {
              onStart: (e) => patch(aid, (m) => ({ ...m, lang: e.lang, script: e.script })),
              onSentence: (t) =>
                patch(aid, (m) => {
                  const part = stripTags(t);
                  const parts = [...(m.parts ?? []), part];
                  return { ...m, parts, text: parts.join(" ") };
                }),
            },
            ac.signal,
          );
        } catch (e) {
          // Only fall back on network failure, never on server-reported errors / aborts.
          if (e instanceof ServerError || ac.signal.aborted || !(e instanceof TypeError)) throw e;
          done = await postQuery(body, ac.signal);
        }
        finalize(aid, done);
      } catch (e) {
        if (ac.signal.aborted) {
          patch(aid, (m) => ({ ...m, streaming: false, text: m.text || "(stopped)" }));
        } else {
          const msg = e instanceof Error ? e.message : "Request failed";
          onError(msg);
          patch(aid, (m) => ({
            ...m,
            streaming: false,
            error: true,
            text: m.text || "Sorry, I could not get an answer. Please try again.",
          }));
        }
      } finally {
        acRef.current = null;
        setBusy(false);
      }
    },
    [lang, patch, finalize, onError],
  );

  const stop = useCallback(() => acRef.current?.abort(), []);

  const regenerate = useCallback(() => {
    if (acRef.current) return;
    const ms = messagesRef.current;
    let i = ms.length - 1;
    while (i >= 0 && ms[i].role !== "user") i--;
    if (i < 0) return;
    void send(ms[i].text, ms.slice(0, i));
  }, [send]);

  const clear = useCallback(() => {
    acRef.current?.abort();
    setMessages([]);
    onVisual(null);
  }, [onVisual]);

  const appendVoice = useCallback((m: ChatMessage) => setMessages((ms) => [...ms, m]), []);

  return { messages, busy, send, stop, regenerate, clear, appendVoice };
}
