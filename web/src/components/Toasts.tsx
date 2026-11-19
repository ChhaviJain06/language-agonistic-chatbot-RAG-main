import { IconAlert, IconClose } from "../lib/icons";
import type { Toast } from "../lib/types";

export default function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="toasts" role="region" aria-label="Notifications" aria-live="assertive">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} role="alert">
          <IconAlert size={16} />
          <span>{t.text}</span>
          <button type="button" className="icon-btn sm" aria-label="Dismiss notification" onClick={() => onDismiss(t.id)}>
            <IconClose size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
