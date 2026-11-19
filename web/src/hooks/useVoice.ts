import { useCallback, useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track, type RemoteTrack } from "livekit-client";
import { fetchLivekitToken, fetchVoiceTurns } from "../lib/api";
import { stripTags } from "../lib/text";
import type { ChatMessage, Source, Timing, VisualStageData, VoiceState } from "../lib/types";

interface VoiceMsg {
  event?: string;
  msg_id?: string;
  status?: string;
  question?: string;
  transcript?: string;
  answer?: string;
  sources?: Source[];
  lang?: string;
  script?: string;
  fidelity_passed?: boolean;
  cached?: boolean;
  visual_stage?: VisualStageData | null;
  timing?: Timing | null;
  error?: string;
  spoken_text?: string;
}

interface Opts {
  onError: (msg: string) => void;
  onVisual: (v: VisualStageData | null) => void;
  onMessage: (m: ChatMessage) => void;
}

const SEEN_MAX = 200;
let n = 0;

export function useVoice({ onError, onVisual, onMessage }: Opts) {
  const [state, setState] = useState<VoiceState>("disconnected");
  const [level, setLevel] = useState(0);
  const [userText, setUserText] = useState("");
  const [agentText, setAgentText] = useState("");
  const [interrupted, setInterrupted] = useState(false);

  const roomRef = useRef<Room | null>(null);
  const roomName = useRef<string | null>(null);
  // Set while a question has been heard but its answer has not arrived yet.
  const pendingAnswer = useRef(false);
  const audioEls = useRef<Set<HTMLMediaElement>>(new Set());
  const seen = useRef<string[]>([]);
  const alive = useRef(true);
  const connecting = useRef(false);
  const cb = useRef({ onError, onVisual, onMessage });
  useEffect(() => {
    cb.current = { onError, onVisual, onMessage };
  });

  const cleanup = useCallback(() => {
    const room = roomRef.current;
    roomRef.current = null;
    audioEls.current.forEach((el) => {
      el.pause();
      el.srcObject = null;
      el.remove();
    });
    audioEls.current.clear();
    room?.removeAllListeners();
    void room?.disconnect();
  }, []);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      cleanup();
    };
  }, [cleanup]);

  // Keyed by event + id: one turn sends several events ("thinking" status, then the
  // answer) under the SAME msg_id, and keying on the id alone dropped every answer.
  const isDup = (key?: string) => {
    if (!key) return false;
    if (seen.current.includes(key)) return true;
    seen.current.push(key);
    if (seen.current.length > SEEN_MAX) seen.current.shift();
    return false;
  };

  const apply = useCallback((m: VoiceMsg) => {
    const { onError, onVisual, onMessage } = cb.current;
    switch (m.event) {
      case "status": {
        if (isDup(m.msg_id && `status:${m.msg_id}:${m.status}`)) return;
        if (m.question) {
          setUserText(m.question);
          setAgentText("");
        }
        if (m.status === "thinking") pendingAnswer.current = true;
        setInterrupted(false);
        if (m.status === "thinking") setState("thinking");
        else if (m.status === "speaking") setState("speaking");
        else if (m.status === "listening" || m.status === "idle") setState("listening");
        break;
      }
      case "answer": {
        if (isDup(m.msg_id && `answer:${m.msg_id}`)) return;
        pendingAnswer.current = false;
        const answer = m.answer ?? "";
        const q = m.transcript ?? "";
        setAgentText(stripTags(answer));
        if (q) setUserText(q);
        if (m.visual_stage) onVisual(m.visual_stage);
        const ts = Date.now();
        if (q) onMessage({ id: `vu-${ts}-${++n}`, role: "user", text: q, ts, via: "voice" });
        onMessage({
          id: `va-${ts}-${++n}`,
          role: "assistant",
          text: answer,
          ts,
          via: "voice",
          lang: m.lang,
          script: m.script,
          sources: m.sources ?? [],
          fidelity: m.fidelity_passed,
          cached: m.cached,
          timing: m.timing ?? null,
          visual: m.visual_stage ?? null,
        });
        break;
      }
      case "barge_in":
        setInterrupted(true);
        setState("listening");
        window.setTimeout(() => alive.current && setInterrupted(false), 3000);
        break;
      case "error":
        if (isDup(m.msg_id && `error:${m.msg_id}`)) return;
        pendingAnswer.current = false;
        onError(m.error ?? "Voice agent error");
        setState((s) => (s === "disconnected" ? s : "listening"));
        break;
    }
  }, []);

  const handleData = useCallback(
    (payload: Uint8Array) => {
      try {
        apply(JSON.parse(new TextDecoder().decode(payload)) as VoiceMsg);
      } catch {
        // not a BHASHA message (or malformed JSON) — nothing to show
      }
    },
    [apply],
  );

  // Fetches answers the data channel may have lost. When a question is still unanswered
  // it keeps polling for a while, because the agent finishes the turn after a drop.
  const recover = useCallback(
    async (room: string) => {
      const deadline = Date.now() + 30_000;
      do {
        try {
          const turns = await fetchVoiceTurns(room);
          turns.forEach((t) => apply(t as VoiceMsg));
        } catch {
          // server briefly unreachable — retried below while we are still waiting
        }
        if (!pendingAnswer.current || !alive.current) return;
        await new Promise((r) => setTimeout(r, 3000));
      } while (pendingAnswer.current && alive.current && Date.now() < deadline);
    },
    [apply],
  );

  const disconnect = useCallback(() => {
    pendingAnswer.current = false;
    cleanup();
    setState("disconnected");
    setLevel(0);
  }, [cleanup]);

  const connect = useCallback(async () => {
    if (connecting.current || roomRef.current) return;
    connecting.current = true;
    setState("connecting");
    setUserText("");
    setAgentText("");
    try {
      const { token, url, room: name } = await fetchLivekitToken();
      if (!alive.current) return;
      roomName.current = name;
      pendingAnswer.current = false;
      seen.current = [];   // new room; kept across a disconnect so recovered answers aren't shown twice
      const room = new Room({ adaptiveStream: true, dynacast: true });
      roomRef.current = room;

      room.on(RoomEvent.DataReceived, handleData);
      room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind !== Track.Kind.Audio) return;
        const el = track.attach();
        el.style.display = "none";
        el.setAttribute("data-livekit-audio", "1");
        document.body.appendChild(el);
        audioEls.current.add(el);
        el.play().catch(() => cb.current.onError("Browser blocked audio playback. Click the page and reconnect."));
      });
      room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        track.detach().forEach((el) => {
          audioEls.current.delete(el);
          el.remove();
        });
      });
      room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const remote = speakers.filter((s) => !s.isLocal);
        const local = speakers.find((s) => s.isLocal);
        // Level only drives the orb animation. The state label comes from the server
        // ("status" events): audio gaps between TTS chunks must not flip it to "listening".
        if (remote.length) setLevel(Math.max(...remote.map((s) => s.audioLevel)));
        else setLevel(local?.audioLevel ?? 0);
      });
      room.on(RoomEvent.Reconnecting, () => alive.current && setState("reconnecting"));
      room.on(RoomEvent.Reconnected, () => {
        if (!alive.current) return;
        setState("listening");
        if (roomName.current) void recover(roomName.current);   // anything sent while we were away
      });
      room.on(RoomEvent.Disconnected, () => {
        if (roomRef.current !== room) return;
        cleanup();
        if (alive.current) {
          setState("disconnected");
          setLevel(0);
          if (roomName.current) {
            if (pendingAnswer.current) cb.current.onError("Connection lost — fetching your answer…");
            void recover(roomName.current);
          }
        }
      });

      await room.connect(url, token);
      await room.localParticipant.setMicrophoneEnabled(true);
      if (!alive.current || roomRef.current !== room) {
        void room.disconnect();
        return;
      }
      await room.startAudio().catch(() => undefined);
      setState("listening");
    } catch (e) {
      cleanup();
      if (alive.current) {
        setState("disconnected");
        const denied = e instanceof Error && /permission|denied|NotAllowed/i.test(`${e.name} ${e.message}`);
        cb.current.onError(
          denied
            ? "Microphone access was denied. Allow the microphone to use Live Voice."
            : e instanceof Error
              ? e.message
              : "Could not start voice session",
        );
      }
    } finally {
      connecting.current = false;
    }
  }, [cleanup, handleData, recover]);

  return { state, level, userText, agentText, interrupted, connect, disconnect };
}
