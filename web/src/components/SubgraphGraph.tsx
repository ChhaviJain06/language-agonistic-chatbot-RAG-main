import { useMemo, useRef, useState } from "react";
import type { GraphLink, GraphNode } from "../lib/types";

const W = 640;
const H = 420;
const PAD = 56;
const R = 15;

interface P {
  x: number;
  y: number;
}

function layout(nodes: GraphNode[], links: GraphLink[]): Map<string, P> {
  const n = nodes.length;
  const pos = new Map<string, P>();
  const vel = new Map<string, P>();
  nodes.forEach((nd, i) => {
    const a = (i / Math.max(n, 1)) * Math.PI * 2;
    pos.set(nd.id, { x: W / 2 + Math.cos(a) * 140, y: H / 2 + Math.sin(a) * 110 });
    vel.set(nd.id, { x: 0, y: 0 });
  });
  const edges = links.filter((l) => pos.has(l.source) && pos.has(l.target));
  for (let it = 0; it < 260; it++) {
    const cool = 1 - it / 260;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = pos.get(nodes[i].id)!;
        const b = pos.get(nodes[j].id)!;
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) {
          dx = Math.random() - 0.5;
          dy = Math.random() - 0.5;
          d2 = 1;
        }
        const d = Math.sqrt(d2);
        const f = 5200 / d2;
        const va = vel.get(nodes[i].id)!;
        const vb = vel.get(nodes[j].id)!;
        va.x += (dx / d) * f;
        va.y += (dy / d) * f;
        vb.x -= (dx / d) * f;
        vb.y -= (dy / d) * f;
      }
    }
    for (const e of edges) {
      const a = pos.get(e.source)!;
      const b = pos.get(e.target)!;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - 130) * 0.04;
      const va = vel.get(e.source)!;
      const vb = vel.get(e.target)!;
      va.x += (dx / d) * f;
      va.y += (dy / d) * f;
      vb.x -= (dx / d) * f;
      vb.y -= (dy / d) * f;
    }
    for (const nd of nodes) {
      const p = pos.get(nd.id)!;
      const v = vel.get(nd.id)!;
      v.x += (W / 2 - p.x) * 0.01;
      v.y += (H / 2 - p.y) * 0.01;
      p.x += Math.max(-14, Math.min(14, v.x)) * cool;
      p.y += Math.max(-14, Math.min(14, v.y)) * cool;
      v.x *= 0.6;
      v.y *= 0.6;
    }
  }
  // fit to viewBox
  const xs = [...pos.values()].map((p) => p.x);
  const ys = [...pos.values()].map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (W - PAD * 2) / Math.max(maxX - minX, 1);
  const sy = (H - PAD * 2) / Math.max(maxY - minY, 1);
  const s = Math.min(sx, sy, 2);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  pos.forEach((p) => {
    p.x = W / 2 + (p.x - cx) * s;
    p.y = H / 2 + (p.y - cy) * s;
  });
  return pos;
}

const short = (s: string, n = 20) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

interface GProps {
  nodes: GraphNode[];
  links: GraphLink[];
}

/** Re-mounts the inner graph (fresh layout) whenever the data changes. */
export default function SubgraphGraph(props: GProps) {
  const sig = JSON.stringify([props.nodes, props.links]);
  return <Graph key={sig} {...props} />;
}

function Graph({ nodes, links }: GProps) {
  const [pos, setPos] = useState<Map<string, P>>(() => layout(nodes, links));
  const [hover, setHover] = useState<string | null>(null);
  const [drag, setDrag] = useState<string | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  const groups = useMemo(() => [...new Set(nodes.map((n) => String(n.group ?? "")))], [nodes]);
  const colorOf = (n: GraphNode) => `var(--g${groups.indexOf(String(n.group ?? "")) % 6})`;

  const edges = links.filter((l) => pos.has(l.source) && pos.has(l.target));
  const names = new Map(nodes.map((n) => [n.id, n.name]));
  const linked = (id: string) => hover === id || edges.some((e) => (e.source === hover && e.target === id) || (e.target === hover && e.source === id));

  const toSvg = (ev: React.PointerEvent): P | null => {
    const el = svg.current;
    const m = el?.getScreenCTM();
    if (!el || !m) return null;
    const pt = el.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    const r = pt.matrixTransform(m.inverse());
    return { x: Math.max(R, Math.min(W - R, r.x)), y: Math.max(R, Math.min(H - R, r.y)) };
  };

  return (
    <div className="graph-wrap">
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className="graph"
        role="img"
        aria-label={`Knowledge graph: ${nodes.map((n) => n.name).join(", ")}. ${edges.map((e) => `${names.get(e.source)} ${e.relation} ${names.get(e.target)}`).join("; ")}`}
        onPointerMove={(e) => {
          if (!drag) return;
          const p = toSvg(e);
          if (p) setPos((m) => new Map(m).set(drag, p));
        }}
        onPointerUp={() => setDrag(null)}
        onPointerLeave={() => setDrag(null)}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 10 5 0 10z" className="arrow" />
          </marker>
          <marker id="arrow-hl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 10 5 0 10z" className="arrow hl" />
          </marker>
        </defs>

        {edges.map((e, i) => {
          const a = pos.get(e.source)!;
          const b = pos.get(e.target)!;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d = Math.sqrt(dx * dx + dy * dy) || 1;
          const ux = dx / d, uy = dy / d;
          const rev = edges.some((o) => o.source === e.target && o.target === e.source);
          const bend = rev ? 22 : 0;
          const mx = (a.x + b.x) / 2 - uy * bend;
          const my = (a.y + b.y) / 2 + ux * bend;
          const x1 = a.x + ux * R, y1 = a.y + uy * R;
          const x2 = b.x - ux * (R + 4), y2 = b.y - uy * (R + 4);
          const hl = hover !== null && (e.source === hover || e.target === hover);
          const dim = hover !== null && !hl;
          return (
            <g key={i} className={`edge${hl ? " hl" : ""}${dim ? " dim" : ""}`}>
              <path d={`M${x1} ${y1} Q${mx} ${my} ${x2} ${y2}`} markerEnd={hl ? "url(#arrow-hl)" : "url(#arrow)"} fill="none" />
              <text x={(x1 + x2) / 2 - uy * bend * 0.5} y={(y1 + y2) / 2 + ux * bend * 0.5 - 4} textAnchor="middle">
                {short(e.relation, 24)}
              </text>
            </g>
          );
        })}

        {nodes.map((n, i) => {
          const p = pos.get(n.id);
          if (!p) return null;
          const dim = hover !== null && hover !== n.id && !linked(n.id);
          return (
            <g key={n.id} transform={`translate(${p.x} ${p.y})`}>
            <g
              className={`gnode${dim ? " dim" : ""}${hover === n.id ? " hl" : ""}`}
              style={{ ["--i" as string]: i } as React.CSSProperties}
              tabIndex={0}
              onPointerEnter={() => setHover(n.id)}
              onPointerLeave={() => setHover(null)}
              onFocus={() => setHover(n.id)}
              onBlur={() => setHover(null)}
              onPointerDown={(e) => {
                (e.currentTarget as SVGGElement).setPointerCapture(e.pointerId);
                setDrag(n.id);
              }}
            >
              <title>{n.name}</title>
              <circle r={R} fill={colorOf(n)} />
              <text y={R + 14} textAnchor="middle">
                {short(n.name)}
              </text>
            </g>
            </g>
          );
        })}
      </svg>
      <p className="graph-hint">Arrows point from source to target. Drag nodes to rearrange.</p>
    </div>
  );
}
