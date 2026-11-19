import { useEffect, useRef } from "react";
import { IconClose, IconExternal } from "../lib/icons";
import { wikiHref } from "../lib/text";
import type { DrawerItem } from "../lib/types";

export default function CitationDrawer({ item, onClose }: { item: DrawerItem | null; onClose: () => void }) {
  const closeBtn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (item) closeBtn.current?.focus();
  }, [item]);
  const href = item ? wikiHref(item.url) : null;
  return (
    <>
      <div className={`scrim${item ? " show" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside
        className={`drawer${item ? " open" : ""}`}
        role="dialog"
        aria-modal="false"
        aria-label="Citation"
        aria-hidden={item ? "false" : "true"}
        inert={!item}
      >
        <header className="drawer-head">
          <h2>{item?.n != null ? `Source [${item.n}]` : "Source"}</h2>
          <button ref={closeBtn} type="button" className="icon-btn" onClick={onClose} aria-label="Close citation">
            <IconClose />
          </button>
        </header>
        {item && (
          <div className="drawer-body">
            <div className="drawer-article">{item.article}</div>
            {item.section && <div className="drawer-section">{item.section}</div>}
            <blockquote className="source-text lg">{item.text}</blockquote>
            {href ? (
              <a className="btn" href={href} target="_blank" rel="noopener noreferrer">
                Read on Wikipedia <IconExternal size={14} />
              </a>
            ) : (
              <p className="muted">No link available for this source.</p>
            )}
          </div>
        )}
      </aside>
    </>
  );
}
