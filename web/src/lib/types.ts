export interface Source {
  n: number;
  chunk_id: string;
  article: string;
  section: string;
  url: string;
  text: string;
  score: number | null;
}

export interface GraphNode {
  id: string;
  name: string;
  group?: string | number | null;
}
export interface GraphLink {
  source: string;
  target: string;
  relation: string;
}
export interface Citation {
  chunk_id: string;
  article: string;
  section: string;
  text: string;
  url: string;
}
export interface VisualStageData {
  event?: "HISTORICAL_CONTEXT";
  matched: boolean;
  entity: string | null;
  dynasty: string | null;
  era_years: [number, number] | null;
  geo_coordinates: [number, number] | null;
  subgraph: { nodes: GraphNode[]; links: GraphLink[] } | null;
  citation: Citation | null;
}

export interface Timing {
  retrieval_ms?: number | null;
  first_sentence_ms?: number | null;
  total_ms?: number | null;
}

export interface HistoryItem {
  role: "user" | "assistant";
  content: string;
}

export interface Health {
  status: string;
  profile: string;
  providers: Record<string, string>;
  version: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  ts: number;
  lang?: string;
  script?: string;
  sources?: Source[];
  fidelity?: boolean;
  cached?: boolean;
  timing?: Timing | null;
  visual?: VisualStageData | null;
  streaming?: boolean;
  parts?: string[];
  via?: "text" | "voice";
  error?: boolean;
}

export interface DoneEvent {
  answer: string;
  sources?: Source[];
  lang?: string;
  script?: string;
  fidelity_passed?: boolean;
  repairs?: number;
  cached?: boolean;
  visual_stage?: VisualStageData | null;
  timing?: Timing | null;
}

export type StreamEvent =
  | { event: "start"; lang?: string; script?: string; msg_id?: string }
  | { event: "sentence"; text: string }
  | ({ event: "done" } & DoneEvent)
  | { event: "error"; error: string };

export type VoiceState = "disconnected" | "connecting" | "reconnecting" | "listening" | "thinking" | "speaking";

export interface Toast {
  id: number;
  kind: "error" | "info";
  text: string;
}

export const LANGS: { value: string; label: string; code: string | null }[] = [
  { value: "auto", label: "Auto-detect", code: null },
  { value: "en", label: "English", code: "en" },
  { value: "hi", label: "हिन्दी (Hindi)", code: "hi" },
  { value: "hi-en", label: "Hinglish", code: "hi-en" },
  { value: "mr", label: "मराठी (Marathi)", code: "mr" },
  { value: "bn", label: "বাংলা (Bengali)", code: "bn" },
  { value: "ta", label: "தமிழ் (Tamil)", code: "ta" },
];

export interface DrawerItem {
  n?: number;
  article: string;
  section: string;
  text: string;
  url: string;
}
