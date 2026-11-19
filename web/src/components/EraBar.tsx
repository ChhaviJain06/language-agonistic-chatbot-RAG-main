import { formatEra, formatYear } from "../lib/text";

const MIN = -3300;
const MAX = 2000;
const TICKS = [-3000, -2000, -1000, 0, 1000, 2000];
const pct = (y: number) => Math.max(0, Math.min(100, ((y - MIN) / (MAX - MIN)) * 100));

export default function EraBar({ years }: { years: [number, number] }) {
  const a = pct(Math.min(...years));
  const b = pct(Math.max(...years));
  return (
    <div className="era">
      <div className="era-label">{formatEra(years)}</div>
      <div
        className="era-bar"
        role="img"
        aria-label={`Era ${formatEra(years)} on a timeline from ${formatYear(MIN)} to ${formatYear(MAX)}`}
      >
        <div className="era-span" style={{ left: `${a}%`, width: `${Math.max(b - a, 1.2)}%` }} />
        {TICKS.map((t) => (
          <span key={t} className="era-tick" style={{ left: `${pct(t)}%` }}>
            <i />
            <em>{t === 0 ? "0" : formatYear(t).replace(" CE", "")}</em>
          </span>
        ))}
      </div>
    </div>
  );
}
