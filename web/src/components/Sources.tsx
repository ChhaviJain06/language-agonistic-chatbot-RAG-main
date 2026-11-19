import { useState } from "react";
import { IconChevron, IconExternal } from "../lib/icons";
import { wikiHref } from "../lib/text";
import type { Source } from "../lib/types";

export default function Sources({
  sources,
  onOpen,
  highlight,
}: {
  sources: Source[];
  onOpen: (s: Source) => void;
  highlight?: number | null;
}) {
  const [open, setOpen] = useState<number | null>(null);
  if (!sources.length) return null;
  return (
    <section className="sources" aria-label="Sources">
      <h4 className="sources-title">Sources</h4>
      <ol className="source-list">
        {sources.map((s) => {
          const href = wikiHref(s.url);
          const expanded = open === s.n;
          return (
            <li key={s.n} className={`source-card${highlight === s.n ? " is-hl" : ""}`}>
              <div className="source-head">
                <span className="source-n">{s.n}</span>
                <button
                  type="button"
                  className="source-main"
                  onClick={() => onOpen(s)}
                  title="Open in the citation drawer"
                >
                  <span className="source-article">{s.article}</span>
                  {s.section && <span className="source-section">{s.section}</span>}
                </button>
                {href && (
                  <a className="icon-btn sm" href={href} target="_blank" rel="noopener noreferrer" aria-label={`Open ${s.article} on Wikipedia`}>
                    <IconExternal size={14} />
                  </a>
                )}
                <button
                  type="button"
                  className={`icon-btn sm chev${expanded ? " open" : ""}`}
                  aria-expanded={expanded}
                  aria-label={expanded ? "Hide passage" : "Show passage"}
                  onClick={() => setOpen(expanded ? null : s.n)}
                >
                  <IconChevron size={14} />
                </button>
              </div>
              {expanded && <blockquote className="source-text">{s.text}</blockquote>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
