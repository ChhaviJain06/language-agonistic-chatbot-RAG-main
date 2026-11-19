import type { DoneEvent, Health, HistoryItem, StreamEvent } from "./types";

const API_URL: string = import.meta.env.VITE_API_URL ?? "";
const ACCESS_KEY: string | undefined = import.meta.env.VITE_ACCESS_KEY || undefined;

function headers(json = true): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (ACCESS_KEY) h["X-Access-Key"] = ACCESS_KEY;
  return h;
}

export class ServerError extends Error {}

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const r = await fetch(`${API_URL}/health`, { signal, headers: headers(false) });
  if (!r.ok) throw new Error(`Health check failed (${r.status})`);
  return (await r.json()) as Health;
}

export async function fetchLivekitToken(): Promise<{ token: string; url: string; room: string }> {
  const r = await fetch(`${API_URL}/livekit/token`, { method: "POST", headers: headers() });
  if (!r.ok) throw new ServerError(`Token request failed (${r.status})`);
  return (await r.json()) as { token: string; url: string; room: string };
}

/** Answers the agent finished for a voice room — used to recover ones the data channel lost. */
export async function fetchVoiceTurns(room: string): Promise<Record<string, unknown>[]> {
  const r = await fetch(`${API_URL}/voice/turns?room=${encodeURIComponent(room)}`, { headers: headers(false) });
  if (!r.ok) throw new ServerError(`Could not fetch voice turns (${r.status})`);
  const body = (await r.json()) as { turns?: Record<string, unknown>[] };
  return body.turns ?? [];
}

export interface QueryBody {
  question: string;
  lang?: string;
  history: HistoryItem[];
}

export async function postQuery(body: QueryBody, signal: AbortSignal): Promise<DoneEvent> {
  const r = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) throw new ServerError(`Server error (${r.status})`);
  return (await r.json()) as DoneEvent;
}

export interface StreamHandlers {
  onStart: (e: { lang?: string; script?: string }) => void;
  onSentence: (text: string) => void;
}

/**
 * Streams /query/stream. Resolves with the final "done" event.
 * Throws ServerError for server-reported errors (never falls back),
 * and a plain Error / TypeError for network failures.
 */
export async function streamQuery(
  body: QueryBody,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<DoneEvent> {
  const r = await fetch(`${API_URL}/query/stream`, {
    method: "POST",
    headers: { ...headers(), Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) throw new ServerError(`Server error (${r.status})`);

  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const result: { done: DoneEvent | null } = { done: null };

  const handle = (frame: string) => {
    const data = frame
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).replace(/^ /, ""))
      .join("\n");
    if (!data) return;
    let ev: StreamEvent;
    try {
      ev = JSON.parse(data) as StreamEvent;
    } catch {
      return; // malformed frame; ignore
    }
    if (ev.event === "start") handlers.onStart(ev);
    else if (ev.event === "sentence") handlers.onSentence(ev.text);
    else if (ev.event === "done") result.done = ev;
    else if (ev.event === "error") throw new ServerError(ev.error || "Unknown server error");
  };

  for (;;) {
    const { value, done: eof } = await reader.read();
    if (value) buf += dec.decode(value, { stream: !eof });
    let idx: number;
    while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx).replace(/^\r?\n\r?\n/, "");
      handle(frame);
    }
    if (eof) break;
  }
  if (buf.trim()) handle(buf);
  if (!result.done) throw new Error("Stream ended before the answer was complete");
  return result.done;
}
