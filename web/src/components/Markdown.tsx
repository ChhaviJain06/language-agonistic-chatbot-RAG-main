import { Fragment } from "react";
import { parseBlocks, parseInline, stripTags } from "../lib/text";

interface Props {
  text: string;
  /** citation numbers that have a matching source */
  known: Set<number>;
  onCite: (n: number) => void;
}

export function Inline({ text, known, onCite }: Props) {
  return (
    <>
      {parseInline(text).map((t, i) => {
        if (t.kind === "bold") return <strong key={i}>{t.text}</strong>;
        if (t.kind === "cite") {
          return known.has(t.n) ? (
            <sup key={i} className="cite-wrap">
              <button type="button" className="cite" onClick={() => onCite(t.n)} aria-label={`Open source ${t.n}`}>
                {t.n}
              </button>
            </sup>
          ) : (
            <sup key={i} className="cite cite-static">
              {t.n}
            </sup>
          );
        }
        return <Fragment key={i}>{t.text}</Fragment>;
      })}
    </>
  );
}

export default function Markdown({ text, known, onCite }: Props) {
  const blocks = parseBlocks(stripTags(text));
  return (
    <>
      {blocks.map((b, i) =>
        b.kind === "p" ? (
          <p key={i}>
            <Inline text={b.text} known={known} onCite={onCite} />
          </p>
        ) : (
          <ul key={i}>
            {b.items.map((it, j) => (
              <li key={j}>
                <Inline text={it} known={known} onCite={onCite} />
              </li>
            ))}
          </ul>
        ),
      )}
    </>
  );
}
