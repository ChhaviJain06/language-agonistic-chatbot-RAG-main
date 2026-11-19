import { useState } from "react";
import { IconMoon, IconPanel, IconSun, IconTrash } from "../lib/icons";
import { LANGS, type Health } from "../lib/types";
import type { Theme } from "../hooks/useTheme";

interface Props {
  health: Health | null;
  down: boolean;
  checking: boolean;
  lang: string;
  onLang: (v: string) => void;
  theme: Theme;
  onTheme: () => void;
  stageOpen: boolean;
  onStage: () => void;
  onClear: () => void;
  hasMessages: boolean;
}

export default function Header(p: Props) {
  const [pop, setPop] = useState(false);
  const status = p.down ? "down" : p.health ? "ok" : "wait";
  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">भ</span>
        <span className="brand-name">BHASHA</span>
        <span className="brand-tag">Indian history, multilingual</span>
      </div>

      <div className="header-actions">
        <div
          className="status-wrap"
          onMouseEnter={() => setPop(true)}
          onMouseLeave={() => setPop(false)}
          onFocus={() => setPop(true)}
          onBlur={() => setPop(false)}
        >
          <button type="button" className={`pill ${status}`} aria-describedby="health-pop" aria-label={`Backend status: ${status === "ok" ? "online" : status === "down" ? "unreachable" : "checking"}`}>
            <span className="dot" />
            {status === "ok" ? `${p.health?.profile ?? "online"}` : status === "down" ? "offline" : "checking"}
          </button>
          <div id="health-pop" role="tooltip" className={`popover${pop ? " show" : ""}`}>
            {p.health ? (
              <>
                <div className="pop-title">
                  Profile: <b>{p.health.profile}</b> · v{p.health.version}
                </div>
                <dl>
                  {Object.entries(p.health.providers ?? {}).map(([k, v]) => (
                    <div key={k}>
                      <dt>{k}</dt>
                      <dd>{String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : (
              <div>{p.down ? "Backend not reachable." : "Checking backend…"}</div>
            )}
          </div>
        </div>

        <label className="select-wrap">
          <span className="sr-only">Answer language</span>
          <select value={p.lang} onChange={(e) => p.onLang(e.target.value)} aria-label="Answer language">
            {LANGS.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </label>

        {p.hasMessages && (
          <button type="button" className="icon-btn" onClick={p.onClear} aria-label="Clear conversation" title="Clear conversation">
            <IconTrash />
          </button>
        )}
        <button
          type="button"
          className="icon-btn"
          onClick={p.onTheme}
          aria-label={`Switch to ${p.theme === "dark" ? "light" : "dark"} theme`}
          title="Toggle theme"
        >
          {p.theme === "dark" ? <IconSun /> : <IconMoon />}
        </button>
        <button
          type="button"
          className={`icon-btn${p.stageOpen ? " active" : ""}`}
          onClick={p.onStage}
          aria-pressed={p.stageOpen}
          aria-label="Toggle Visual Stage"
          title="Visual Stage"
        >
          <IconPanel />
        </button>
      </div>
    </header>
  );
}
