// DEV-ONLY mock backend for trying the UI without the real server.
// All data below is clearly-labelled DEMO data. Run: node mock/mock-server.mjs
import http from "node:http";

const PORT = Number(process.env.PORT ?? 8000);
const DEMO = "[DEMO DATA] ";

const sources = [
  { n: 1, chunk_id: "demo::0001", article: "Shah Jahan (demo)", section: "Early life", url: "https://en.wikipedia.org/wiki/Shah_Jahan",
    text: DEMO + "Placeholder passage standing in for a retrieved Wikipedia chunk about Shah Jahan.", score: 0.91 },
  { n: 2, chunk_id: "demo::0002", article: "Akbar (demo)", section: "Legacy", url: "https://en.wikipedia.org/wiki/Akbar",
    text: DEMO + "Placeholder passage standing in for a retrieved Wikipedia chunk about Akbar.", score: 0.84 },
];
const visual_stage = {
  event: "HISTORICAL_CONTEXT", matched: true, entity: "Shah Jahan (demo)", dynasty: "Mughal Empire",
  era_years: [1628, 1658], geo_coordinates: [27.175, 78.042],
  subgraph: {
    nodes: [
      { id: "a", name: "Akbar", group: "ruler" }, { id: "j", name: "Jahangir", group: "ruler" },
      { id: "s", name: "Shah Jahan", group: "ruler" }, { id: "t", name: "Taj Mahal", group: "monument" },
      { id: "m", name: "Mughal Empire", group: "dynasty" },
    ],
    links: [
      { source: "a", target: "j", relation: "father of" }, { source: "j", target: "s", relation: "father of" },
      { source: "s", target: "t", relation: "built" }, { source: "s", target: "m", relation: "ruled" },
    ],
  },
  citation: { chunk_id: "demo::0001", article: "Shah Jahan (demo)", section: "Early life", text: sources[0].text, url: sources[0].url },
};
const answerText = DEMO + "The Taj Mahal was built by Shah Jahan [1]. His grandfather was Akbar [2]. **This is canned text**, not a real answer.";
const timing = { retrieval_ms: 42, first_sentence_ms: 380, total_ms: 900 };

const send = (res, code, obj) => {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
};
const body = (req) => new Promise((r) => { let b = ""; req.on("data", (c) => (b += c)); req.on("end", () => { try { r(JSON.parse(b || "{}")); } catch { r({}); } }); });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

http.createServer(async (req, res) => {
  const path = req.url?.split("?")[0];
  if (req.method === "GET" && path === "/health")
    return send(res, 200, { status: "ok", profile: "onprem", providers: { llm: "demo-mock", tts: "demo-mock", embed: "demo-mock" }, version: "mock-0" });
  if (req.method === "POST" && path === "/livekit/token")
    return send(res, 503, { error: "Mock server has no LiveKit" });
  if (req.method === "POST" && path === "/query") {
    const q = await body(req);
    return send(res, 200, { question: q.question, lang: "en", script: "Latn", answer: answerText, sources, fidelity_passed: true, repairs: 0, cached: false, visual_stage, timing });
  }
  if (req.method === "POST" && path === "/query/stream") {
    const q = await body(req);
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
    const ev = (o) => res.write(`data: ${JSON.stringify(o)}\n\n`);
    if (/error/i.test(q.question ?? "")) { ev({ event: "start", lang: "en", script: "Latn", msg_id: "m1" }); ev({ event: "error", error: "DEMO: simulated server error" }); return res.end(); }
    ev({ event: "start", lang: "en", script: "Latn", msg_id: "m1" });
    for (const s of [DEMO + "The Taj Mahal was built by Shah Jahan [demo::0001].", "His grandfather was Akbar.", "This is canned text."]) { await sleep(500); ev({ event: "sentence", text: s }); }
    await sleep(300);
    ev({ event: "done", answer: answerText, sources, lang: "en", script: "Latn", fidelity_passed: true, repairs: 0, cached: false, visual_stage, timing });
    return res.end();
  }
  send(res, 404, { error: "not found" });
}).listen(PORT, () => console.log(`DEMO mock backend on http://localhost:${PORT}`));
