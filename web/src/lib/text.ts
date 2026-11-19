import type { ChatMessage, HistoryItem } from "./types";

const CHUNK_TAG = /\[[a-z0-9_-]+::\d+\]/gi;
const CJK_TAG = /【[^】]*】/g;

/** Remove raw chunk-id tags and 【…】 spans from text meant for display. */
export function stripTags(text: string): string {
  return text.replace(CHUNK_TAG, "").replace(CJK_TAG, "").replace(/[ \t]{2,}/g, " ");
}

/** Remove [n] citation markers as well (for history sent to the backend / copy). */
export function stripMarkers(text: string): string {
  return stripTags(text)
    .replace(/\s*\[\d+\]/g, "")
    .trim();
}

export function buildHistory(messages: ChatMessage[], limit = 6): HistoryItem[] {
  return messages
    .filter((m) => !m.error && !m.streaming && m.text.trim())
    .slice(-limit)
    .map((m) => ({
      role: m.role,
      content: m.role === "assistant" ? stripMarkers(m.text) : m.text,
    }));
}

export function formatYear(y: number): string {
  return y < 0 ? `${Math.abs(y)} BCE` : `${y} CE`;
}

export function formatEra(years: [number, number]): string {
  const [a, b] = years;
  return a === b ? formatYear(a) : `${formatYear(a)} – ${formatYear(b)}`;
}

export function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function wikiHref(url: string): string | null {
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : null;
  } catch {
    return null;
  }
}

/* ---- tiny safe markdown: paragraphs, **bold**, bullet lists ---- */
export type Block =
  | { kind: "p"; text: string }
  | { kind: "ul"; items: string[] };

export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: string[] | null = null;
  const flushPara = () => {
    if (para.length) blocks.push({ kind: "p", text: para.join(" ") });
    para = [];
  };
  const flushList = () => {
    if (list) blocks.push({ kind: "ul", items: list });
    list = null;
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    if (!line) {
      flushPara();
      flushList();
    } else if (bullet) {
      flushPara();
      (list ??= []).push(bullet[1]);
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return blocks;
}

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "cite"; n: number };

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  const re = /\*\*(.+?)\*\*|\[(\d{1,2})\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    if (m[1] !== undefined) out.push({ kind: "bold", text: m[1] });
    else out.push({ kind: "cite", n: Number(m[2]) });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}
