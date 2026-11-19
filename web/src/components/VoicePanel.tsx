import type { VoiceState } from "../lib/types";
import { IconMic } from "../lib/icons";

interface Props {
  state: VoiceState;
  level: number;
  userText: string;
  agentText: string;
  interrupted: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
}

const LABEL: Record<VoiceState, string> = {
  disconnected: "Live Voice",
  connecting: "Connecting…",
  reconnecting: "Reconnecting… your answer will appear here",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

export default function VoicePanel({ state, level, userText, agentText, interrupted, onConnect, onDisconnect }: Props) {
  const on = state !== "disconnected";
  const scale = 1 + Math.min(1, level * 3) * 0.35;
  return (
    <div className={`voice voice-${state}`}>
      <button
        type="button"
        className="orb-btn"
        aria-pressed={on}
        aria-label={on ? "Disconnect live voice" : "Start live voice conversation"}
        disabled={state === "connecting"}
        onClick={on ? onDisconnect : onConnect}
      >
        <span className="orb-ring" style={{ transform: `scale(${scale})` }} aria-hidden="true" />
        <span className="orb" aria-hidden="true">
          <IconMic size={18} />
        </span>
      </button>
      <div className="voice-text" aria-live="polite">
        <div className="voice-state">
          {LABEL[state]}
          {interrupted && <span className="chip warn">interrupted</span>}
        </div>
        {on ? (
          <>
            {userText && <div className="voice-line"><b>You</b> {userText}</div>}
            {agentText && <div className="voice-line"><b>BHASHA</b> {agentText}</div>}
            {!userText && !agentText && state === "listening" && (
              <div className="voice-hint">Speak a question about Indian history.</div>
            )}
          </>
        ) : (
          <div className="voice-hint">Talk instead of typing. Audio goes to your own LiveKit server only.</div>
        )}
      </div>
    </div>
  );
}
