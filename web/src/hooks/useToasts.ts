import { useCallback, useRef, useState } from "react";
import type { Toast } from "../lib/types";

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seq = useRef(0);

  const dismiss = useCallback((id: number) => {
    setToasts((t) => t.filter((x) => x.id !== id));
  }, []);

  const push = useCallback(
    (text: string, kind: Toast["kind"] = "error") => {
      const id = ++seq.current;
      setToasts((t) => [...t.slice(-3), { id, kind, text }]);
      window.setTimeout(() => dismiss(id), kind === "error" ? 9000 : 4500);
    },
    [dismiss],
  );

  return { toasts, push, dismiss };
}
