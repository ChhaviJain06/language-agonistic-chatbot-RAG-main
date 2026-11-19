# BHASHA web UI

React 19 + TypeScript + Vite frontend for the multilingual Indian-history RAG assistant.
Text chat (SSE streaming), Live Voice (LiveKit) and a Visual Stage (map, knowledge graph, sources/timeline).
No CDN fonts/scripts; the only optional external request is map tiles (see below).

## Run

```bash
npm install
npm run dev        # http://localhost:5173, proxies /livekit /health /query to :8000
npm run build      # tsc -b && vite build
npx oxlint
```

No backend yet? `node mock/mock-server.mjs` starts a dev-only mock on :8000 with clearly
labelled DEMO data (`/health`, `/query`, `/query/stream`; type "error" to simulate a server error).
Live Voice needs the real backend.

## Environment (optional, `.env.local`)

| Variable | Purpose |
| --- | --- |
| `VITE_API_URL` | Backend origin if not using the dev proxy |
| `VITE_ACCESS_KEY` | Sent as `X-Access-Key` on API calls |
| `VITE_OFFLINE_MAP=1` | Skip online tiles; use the vector subcontinent outline |

The map uses CARTO/OpenStreetMap tiles and falls back to the offline vector map on tile errors.

## Layout

- `src/lib` API client (SSE), types, text/markdown helpers, icons
- `src/hooks` chat, voice (LiveKit), health polling, theme, toasts
- `src/components` header, chat, voice panel, Visual Stage (map/graph/era bar), citation drawer

Shortcuts: `/` focus input, `Enter` send, `Esc` stop generation / close drawer.
