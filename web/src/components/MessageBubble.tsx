import { useState } from "react";
import { IconCheck, IconCopy, IconRefresh, IconShield, IconWave } from "../lib/icons";
import { formatTime, stripMarkers } from "../lib/text";
import type { ChatMessage, Source } from "../lib/types";
import Markdown, { Inline } from "./Markdown";
import Sources from "./Sources";

interface Props {
  msg: ChatMessage;
  isLast: boolean;
  canRegenerate: boolean;
  onRegenerate: () => void;
  onOpenSource: (s: Source) => void;
}

const LANG_NAMES: Record<string, string> = {
  en: "English",
  hi: "Hindi",
  "hi-en": "Hinglish",
  mr: "Marathi",
  bn: "Bengali",
  ta: "Tamil",
};

export default function MessageBubble({ msg, isLast, canRegenerate, onRegenerate, onOpenSource }: Props) {
  const [copied, setCopied] = useState(false);
  const sources = msg.sources ?? [];
  const known = new Set(sources.map((s) => s.n));
  const onCite = (n: number) => {
    const s = sources.find((x) => x.n === n);
    if (s) onOpenSource(s);
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(stripMarkers(msg.text));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  if (msg.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble">
          <p>{msg.text}</p>
        </div>
        <div className="meta">
          {msg.via === "voice" && (
            <span className="chip">
              <IconWave size={12} /> voice
            </span>
          )}
          <time>{formatTime(msg.ts)}</time>
        </div>
      </div>
    );
  }

  const waiting = msg.streaming && !msg.text;
  return (
    <div className={`msg msg-bot${msg.error ? " is-error" : ""}`}>
      <div className="avatar" aria-hidden="true">भ</div>
      <div className="msg-body">
        <div className="bubble" aria-busy={msg.streaming ? "true" : "false"}>
          {waiting ? (
            <div className="typing" role="status" aria-label="BHASHA is thinking">
              <span /><span /><span />
              <div className="skeleton" aria-hidden="true">
                <i style={{ width: "92%" }} /><i style={{ width: "78%" }} /><i style={{ width: "54%" }} />
              </div>
            </div>
          ) : msg.streaming && msg.parts ? (
            <p>
              {msg.parts.map((p, i) => (
                <span key={i} className="fade-in">
                  <Inline text={p + " "} known={known} onCite={onCite} />
                </span>
              ))}
            </p>
          ) : (
            <Markdown text={msg.text} known={known} onCite={onCite} />
          )}
        </div>

        {!msg.streaming && !msg.error && (
          <div className="meta">
            <time>{formatTime(msg.ts)}</time>
            {msg.lang && (
              <span className="chip" title="Answer language / script">
                {LANG_NAMES[msg.lang] ?? msg.lang}
                {msg.script ? ` · ${msg.script}` : ""}
              </span>
            )}
            {msg.fidelity === true && (
              <span className="chip ok" title="Every sentence passed script validation">
                <IconShield size={12} /> script verified
              </span>
            )}
            {msg.cached && <span className="chip">cached</span>}
            {msg.timing?.first_sentence_ms != null && (
              <span className="chip" title="Time to first sentence">
                {Math.round(msg.timing.first_sentence_ms)} ms
              </span>
            )}
            {msg.via === "voice" && (
              <span className="chip">
                <IconWave size={12} /> voice
              </span>
            )}
            <span className="spacer" />
            <button type="button" className="icon-btn sm" onClick={copy} aria-label="Copy answer">
              {copied ? <IconCheck size={14} /> : <IconCopy size={14} />}
            </button>
            {isLast && canRegenerate && (
              <button type="button" className="icon-btn sm" onClick={onRegenerate} aria-label="Regenerate answer">
                <IconRefresh size={14} />
              </button>
            )}
          </div>
        )}
        {msg.error && isLast && canRegenerate && (
          <div className="meta">
            <button type="button" className="btn ghost sm" onClick={onRegenerate}>
              <IconRefresh size={14} /> Retry
            </button>
          </div>
        )}
        {!msg.streaming && sources.length > 0 && <Sources sources={sources} onOpen={onOpenSource} />}
      </div>
    </div>
  );
}
