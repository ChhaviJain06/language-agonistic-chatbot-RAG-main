import { useEffect, useRef } from "react";
import type { ChatMessage, Source } from "../lib/types";
import Composer, { type ComposerHandle } from "./Composer";
import EmptyState from "./EmptyState";
import MessageBubble from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  composerRef: React.Ref<ComposerHandle>;
  onSend: (q: string) => void;
  onStop: () => void;
  onRegenerate: () => void;
  onOpenSource: (s: Source) => void;
  voice?: React.ReactNode;
}

export default function ChatPane({ messages, busy, composerRef, onSend, onStop, onRegenerate, onOpenSource, voice }: Props) {
  const end = useRef<HTMLDivElement>(null);
  const log = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = log.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 240;
    if (near || messages[messages.length - 1]?.role === "user") {
      end.current?.scrollIntoView({ block: "end", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }
  }, [messages]);

  return (
    <section className="chat" aria-label="Conversation">
      <div className="chat-log" ref={log} role="log" aria-live="polite" aria-relevant="additions text">
        {messages.length === 0 ? (
          <EmptyState onPick={onSend} disabled={busy} />
        ) : (
          <div className="thread">
            {messages.map((m, i) => (
              <MessageBubble
                key={m.id}
                msg={m}
                isLast={i === messages.length - 1}
                canRegenerate={!busy}
                onRegenerate={onRegenerate}
                onOpenSource={onOpenSource}
              />
            ))}
            <div ref={end} />
          </div>
        )}
      </div>
      {voice}
      <Composer ref={composerRef} busy={busy} onSend={onSend} onStop={onStop} />
    </section>
  );
}
