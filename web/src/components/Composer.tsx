import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import { IconSend, IconStop } from "../lib/icons";

export interface ComposerHandle {
  focus: () => void;
}

interface Props {
  busy: boolean;
  onSend: (q: string) => void;
  onStop: () => void;
}

const Composer = forwardRef<ComposerHandle, Props>(function Composer({ busy, onSend, onStop }, ref) {
  const [text, setText] = useState("");
  const ta = useRef<HTMLTextAreaElement>(null);
  useImperativeHandle(ref, () => ({ focus: () => ta.current?.focus() }));

  const submit = () => {
    const q = text.trim();
    if (!q || busy) return;
    onSend(q);
    setText("");
    if (ta.current) ta.current.style.height = "";
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={ta}
        value={text}
        rows={1}
        lang="und"
        placeholder="Ask about Indian history…  (Enter to send, Shift+Enter for a new line)"
        aria-label="Your question"
        onChange={(e) => {
          setText(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      {busy ? (
        <button type="button" className="send stop" onClick={onStop} aria-label="Stop generating">
          <IconStop />
        </button>
      ) : (
        <button type="submit" className="send" disabled={!text.trim()} aria-label="Send question">
          <IconSend />
        </button>
      )}
    </form>
  );
});

export default Composer;
