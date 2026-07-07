// lithos mini-GPU playground — loads the real WASM engine, runs kernels, and animates
// the enriched trace across three panels: (A) warps × lanes, (B) a word-level memory-block
// map that makes coalescing visible, and (C) a register-level data-flow view. The register
// and memory state shown are REPLAYED on the client from the engine's per-issue trace
// (every writing op emits its per-lane result), so nothing here re-implements the engine.
"use strict";

const EXAMPLES = {
  "vector-add": `# C[i] = A[i] + B[i]  (one warp, well-coalesced)
mov  r1, 0
mov  r2, 32
mov  r3, 64
tid  r0
iadd r4, r1, r0
iadd r5, r2, r0
iadd r6, r3, r0
ld   r7, r4
ld   r8, r5
iadd r9, r7, r8
st   r6, r9
halt`,
  "divergence": `# if (tid < 16) then-path else else-path  -> a warp SPLITS (watch it serialize)
mov  r5, 16
tid  r0
slt  r1, r0, r5
bra  r1, else, join
mov  r2, 100
iadd r3, r2, r0
st   r0, r3
jmp  done
else:
mov  r2, 200
iadd r3, r2, r0
st   r0, r3
join:
done:
halt`,
  "coalesced": `# addr = tid  -> 32 lanes hit a few segments (few transactions)
tid r0
ld  r1, r0
halt`,
  "scattered": `# addr = tid*8 -> every lane in its own segment (many transactions!)
tid  r0
mov  r2, 8
imul r3, r0, r2
ld   r1, r3
halt`,
  "latency-hiding": `# two loads; run with threads=64 (2 warps) to see latency hidden
tid r0
ld  r1, r0
ld  r2, r0
halt`,
};

// ---- palette (mirrors style.css) --------------------------------------------
const C = {
  bg: "#0d1117", panel: "#161b22", panel2: "#0e131a", border: "#30363d",
  fg: "#c9d1d9", muted: "#8b949e", faint: "#6e7681", accent: "#58a6ff",
  alu: "#388bfd", mem: "#f0883e", branch: "#a371f7", idle: "#21262d",
  diverge: "#f85149", ok: "#3fb950", warn: "#f0883e",
};
const OP_KIND = { ld: "mem", st: "mem", bra: "branch", jmp: "branch" };
const kindOf = (op) => OP_KIND[op] || "alu";
const KIND_COLOR = { alu: C.alu, mem: C.mem, branch: C.branch };
// Which register slots each op reads/produces (data-flow arrows). Matches the ISA.
const OP_SEM = {
  mov: { src: ["imm"], dst: true, sym: "=" },
  tid: { src: ["tid"], dst: true, sym: "=" },
  iadd: { src: ["ra", "rb"], dst: true, sym: "+" },
  imul: { src: ["ra", "rb"], dst: true, sym: "×" },
  slt: { src: ["ra", "rb"], dst: true, sym: "<" },
  ld: { src: ["mem"], dst: true, sym: "=" },
  st: { src: ["rb"], dst: false, sym: "→" },
  bra: { src: ["ra"], dst: false, sym: "?" },
  jmp: { src: [], dst: false, sym: "" },
  halt: { src: [], dst: false, sym: "" },
};

let runKernel = null;
let anim = { trace: [], step: 0, playing: false, raf: null, acc: 0, stats: null };

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
let CW = 760, CH = 620;               // logical drawing size (CSS px)

// ---- boot -------------------------------------------------------------------
SimtModule().then((M) => {
  runKernel = M.cwrap("run_kernel", "string",
    ["string", "number", "number", "number", "number"]);
  buildExamples();
  $("editor").value = EXAMPLES["vector-add"];
  resizeCanvas();
  run();
});
window.addEventListener("resize", () => { resizeCanvas(); draw(); });

function buildExamples() {
  const box = $("examples");
  for (const name of Object.keys(EXAMPLES)) {
    const b = document.createElement("button");
    b.textContent = name;
    b.onclick = () => {
      $("editor").value = EXAMPLES[name];
      if (name === "latency-hiding") $("threads").value = 64;
      run();
    };
    box.appendChild(b);
  }
}

// crisp rendering on HiDPI: back the canvas with devicePixelRatio, draw in CSS px.
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  CW = Math.max(360, canvas.clientWidth || 760);
  const nWarps = anim.stats ? anim.stats.n_warps : 1;
  CH = 300 + Math.max(0, nWarps - 1) * 26;   // grow a little with more warps
  canvas.width = Math.round(CW * dpr);
  canvas.height = Math.round(CH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ---- run --------------------------------------------------------------------
function run() {
  if (!runKernel) return;
  const src = $("editor").value;
  const threads = parseInt($("threads").value, 10) || 32;
  const latency = parseInt($("latency").value, 10) || 200;
  const seg = parseInt($("segwords").value, 10) || 8;
  $("error").textContent = "";
  let data;
  try {
    data = JSON.parse(runKernel(src, threads, latency, seg, 8));
  } catch (e) {
    $("error").textContent = "failed to run: " + e;
    return;
  }
  if (data.error) { $("error").textContent = data.error; renderStats(null); return; }
  anim.trace = data.trace;
  anim.stats = data.stats;
  anim.step = 0;
  stopPlay();
  resizeCanvas();
  renderStats(data.stats);
  startPlay();
}

function renderStats(s) {
  const box = $("stats");
  if (!s) { box.innerHTML = ""; return; }
  const items = [
    ["cycles", s.cycles], ["warp-instrs", s.warp_instructions],
    ["mem ops", s.mem_ops], ["mem txns", s.mem_transactions],
    ["divergences", s.divergent_branches], ["warps", s.n_warps],
  ];
  box.innerHTML = items.map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");
}

// ---- replay: reconstruct per-lane registers up to the current issue ----------
// The engine emits, for every writing op, the value each active lane produced. Applying
// those in order rebuilds the exact register file — so we can show real operand values.
function regfileBefore(step) {
  const s = anim.stats;
  const NR = 16, WS = s.warp_size, nW = s.n_warps;
  // regs[warp][lane][reg]
  const regs = Array.from({ length: nW }, () =>
    Array.from({ length: WS }, () => new Int32Array(NR)));
  for (let i = 0; i < step - 1 && i < anim.trace.length; i++) {
    const e = anim.trace[i];
    if (e.writes_rd && e.lane_val && e.rd >= 0) {
      for (let l = 0; l < WS; l++)
        if (e.mask & (1 << l)) regs[e.warp][l][e.rd] = e.lane_val[l];
    }
  }
  return regs;
}
function firstActiveLane(mask, WS) {
  for (let l = 0; l < WS; l++) if (mask & (1 << l)) return l;
  return -1;
}

// ---- animation --------------------------------------------------------------
function startPlay() { anim.playing = true; $("play").textContent = "⏸ pause"; lastT = 0; loop(); }
function stopPlay() {
  anim.playing = false; $("play").textContent = "▶ play";
  if (anim.raf) cancelAnimationFrame(anim.raf);
}
$("play").onclick = () => { if (anim.playing) stopPlay(); else { if (anim.step >= anim.trace.length) anim.step = 0; startPlay(); } };
$("step").onclick = () => { stopPlay(); if (anim.step < anim.trace.length) anim.step++; draw(); };
$("reset").onclick = () => { stopPlay(); anim.step = 0; draw(); };
$("run").onclick = run;

let lastT = 0;
function loop(t) {
  if (!anim.playing) return;
  const eps = parseInt($("speed").value, 10);
  if (!lastT) lastT = t || 0;
  const dt = ((t || 0) - lastT) / 1000;
  lastT = t || 0;
  anim.acc += dt * eps;
  while (anim.acc >= 1 && anim.step < anim.trace.length) { anim.step++; anim.acc -= 1; }
  draw();
  if (anim.step >= anim.trace.length) { stopPlay(); anim.acc = 0; lastT = 0; return; }
  anim.raf = requestAnimationFrame(loop);
}

// ---- drawing helpers --------------------------------------------------------
function rr(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function panelTitle(text, x, y) {
  ctx.fillStyle = C.muted; ctx.font = "600 11px ui-monospace, Consolas, monospace";
  ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
  ctx.fillText(text.toUpperCase(), x, y);
}

// geometry of the lane grid, shared so memory connectors can point at real lanes.
let laneGeom = null;

function draw() {
  ctx.clearRect(0, 0, CW, CH);
  const s = anim.stats;
  if (!s) return;
  const cur = anim.step > 0 ? anim.trace[anim.step - 1] : null;
  $("cyclereadout").textContent = cur
    ? `cycle ${cur.cycle}  ·  issue ${anim.step}/${anim.trace.length}`
    : "cycle —";

  const padX = 14;
  let y = 22;
  y = drawWarps(padX, y, s, cur);
  y += 14;
  y = drawMemory(padX, y, s, cur);
  y += 14;
  drawDataflow(padX, y, s, cur);
  drawProgress();
}

// Panel A — warps × lanes ------------------------------------------------------
function drawWarps(x0, y0, s, cur) {
  panelTitle("warps × lanes  (lit = active this issue · color = op kind)", x0, y0);
  let y = y0 + 8;
  const WS = s.warp_size, nW = s.n_warps;
  const labelW = 52;
  const gridW = CW - x0 * 2 - labelW;
  const cell = Math.max(8, Math.min(20, Math.floor(gridW / WS) - 2));
  const gap = 2, rowH = cell + 8;
  laneGeom = { x0: x0 + labelW, y, cell, gap, rowH, WS };

  for (let w = 0; w < nW; w++) {
    const ry = y + w * rowH;
    const issuing = cur && cur.warp === w;
    if (issuing) {
      ctx.fillStyle = "rgba(88,166,255,0.06)";
      rr(x0, ry - 3, CW - x0 * 2, cell + 6, 5); ctx.fill();
    }
    ctx.fillStyle = issuing ? C.accent : C.muted;
    ctx.font = "11px ui-monospace, Consolas, monospace";
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText("warp " + w, x0, ry + cell / 2);
    for (let l = 0; l < WS; l++) {
      const cx = laneGeom.x0 + l * (cell + gap);
      let fill = C.idle, stroke = null;
      if (issuing && (cur.mask & (1 << l))) {
        fill = KIND_COLOR[kindOf(cur.op)];
        if (cur.diverged) stroke = C.diverge;
      }
      ctx.fillStyle = fill; rr(cx, ry, cell, cell, 3); ctx.fill();
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
    }
    if (issuing && cur.diverged) {
      ctx.fillStyle = C.diverge; ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText("⚡ diverged", laneGeom.x0 + WS * (cell + gap) + 6, ry + cell / 2);
    }
  }
  return y + nW * rowH;
}

// Panel B — memory-block map ---------------------------------------------------
// Draws the memory segments touched by the CURRENT memory op as blocks of `segment_words`
// word-cells; lit cells are the words active lanes address; connectors run from the issuing
// warp's lanes to the word they hit. Many lanes → one block = coalesced; fan-out = scattered.
function drawMemory(x0, y0, s, cur) {
  const segW = s.segment_words;
  const isMem = cur && cur.is_mem && cur.lane_addr;
  panelTitle(
    isMem ? `global memory  —  ${cur.op.toUpperCase()} : ${cur.txns} transaction${cur.txns === 1 ? "" : "s"} (${cur.txns <= 4 ? "coalesced" : cur.txns >= s.warp_size / 2 ? "scattered" : "partly coalesced"})`
          : "global memory  —  (no memory traffic this instruction)",
    x0, y0);
  const top = y0 + 10;
  const boxH = 46;
  ctx.fillStyle = C.panel2; rr(x0, top, CW - x0 * 2, boxH, 6); ctx.fill();
  ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();

  if (!isMem) {
    ctx.fillStyle = C.faint; ctx.font = "12px ui-monospace, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("memory is idle — this is an ALU/branch instruction", CW / 2, top + boxH / 2);
    return top + boxH;
  }

  // lanes → their addresses (active only), grouped into segments.
  const WS = s.warp_size;
  const laneAddr = [];  // {lane, addr}
  for (let l = 0; l < WS; l++)
    if ((cur.mask & (1 << l)) && cur.lane_addr[l] >= 0)
      laneAddr.push({ lane: l, addr: cur.lane_addr[l] });
  const segs = [...new Set(laneAddr.map((a) => Math.floor(a.addr / segW)))].sort((a, b) => a - b);

  const areaX = x0 + 8, areaW = CW - x0 * 2 - 16;
  const nSeg = segs.length;
  const blockGap = 8;
  const blockW = Math.max(10, Math.min(120, (areaW - blockGap * (nSeg - 1)) / nSeg));
  const showWords = blockW >= segW * 6;      // enough room to draw individual word cells
  const by = top + 8, bh = boxH - 16;
  const segX = {};                            // segment index -> center x of its block

  for (let i = 0; i < nSeg; i++) {
    const bx = areaX + i * (blockW + blockGap);
    segX[segs[i]] = bx + blockW / 2;
    ctx.fillStyle = C.idle; rr(bx, by, blockW, bh, 4); ctx.fill();
    ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();
    if (showWords) {
      const wCellW = (blockW - 6) / segW;
      for (let wi = 0; wi < segW; wi++) {
        const wx = bx + 3 + wi * wCellW;
        const addr = segs[i] * segW + wi;
        const hit = laneAddr.some((a) => a.addr === addr);
        ctx.fillStyle = hit ? C.mem : "#1b2129";
        rr(wx + 0.5, by + 6, wCellW - 1, bh - 20, 2); ctx.fill();
      }
    } else {
      // compact: fill intensity by how many lanes hit this segment
      const n = laneAddr.filter((a) => Math.floor(a.addr / segW) === segs[i]).length;
      const alpha = Math.min(1, 0.25 + n / WS);
      ctx.fillStyle = `rgba(240,136,62,${alpha})`;
      rr(bx + 2, by + 2, blockW - 4, bh - 16, 3); ctx.fill();
    }
    ctx.fillStyle = C.faint; ctx.font = "9px ui-monospace, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("seg " + segs[i], bx + blockW / 2, by + bh);
  }

  // connectors: issuing lane cell (panel A) → its word/segment block
  if (laneGeom && laneGeom.WS === WS) {
    const lg = laneGeom;
    const laneRowY = lg.y + cur.warp * lg.rowH + lg.cell;   // bottom of the issuing lane row
    ctx.lineWidth = 1;
    for (const a of laneAddr) {
      const lx = lg.x0 + a.lane * (lg.cell + lg.gap) + lg.cell / 2;
      let tx = segX[Math.floor(a.addr / segW)];
      if (showWords) {
        const wCellW = (blockW - 6) / segW;
        const bx = tx - blockW / 2;
        tx = bx + 3 + (a.addr % segW) * wCellW + wCellW / 2;
      }
      ctx.strokeStyle = "rgba(240,136,62,0.35)";
      ctx.beginPath(); ctx.moveTo(lx, laneRowY + 2); ctx.lineTo(tx, by); ctx.stroke();
    }
  }
  return top + boxH;
}

// Panel C — register data-flow -------------------------------------------------
// Shows the current instruction as a data-flow for one representative active lane:
// source values flow through the op into the destination register. Values are the REAL
// per-lane values replayed from the trace.
function drawDataflow(x0, y0, s, cur) {
  panelTitle("data flow  (one representative lane)", x0, y0);
  const top = y0 + 10, boxH = 92;
  ctx.fillStyle = C.panel2; rr(x0, top, CW - x0 * 2, boxH, 6); ctx.fill();
  ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();
  if (!cur) return top + boxH;

  const WS = s.warp_size;
  const lane = firstActiveLane(cur.mask, WS);
  const sem = OP_SEM[cur.op] || { src: [], dst: false, sym: "" };
  const regs = regfileBefore(anim.step);
  const before = (lane >= 0) ? regs[cur.warp][lane] : new Int32Array(16);

  // header: the assembly + which lane we're showing
  ctx.fillStyle = C.fg; ctx.font = "13px ui-monospace, Consolas, monospace";
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  const asm = asmText(cur);
  ctx.fillText(asm, x0 + 12, top + 10);
  if (lane >= 0) {
    ctx.fillStyle = C.faint; ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("lane " + lane + (cur.after_stall ? "   ·   ⏱ issued after a STALL (memory latency)" : ""),
      x0 + 12, top + 28);
  }

  // build the src → op → dst node chain
  const cy = top + 62;
  const nodes = [];
  for (const src of sem.src) {
    if (src === "imm") nodes.push({ label: "imm", val: cur.imm });
    else if (src === "tid") nodes.push({ label: "tid", val: (lane >= 0 ? cur.lane_val[lane] : 0) });
    else if (src === "mem") nodes.push({ label: "mem[" + (lane >= 0 && cur.lane_addr ? cur.lane_addr[lane] : "?") + "]", val: (lane >= 0 && cur.lane_val ? cur.lane_val[lane] : 0) });
    else if (src === "ra") nodes.push({ label: "r" + cur.ra, val: before[cur.ra] });
    else if (src === "rb") nodes.push({ label: "r" + cur.rb, val: before[cur.rb] });
  }
  let dst = null;
  if (sem.dst) dst = { label: "r" + cur.rd, val: (lane >= 0 && cur.lane_val ? cur.lane_val[lane] : 0) };
  else if (cur.is_store) dst = { label: "mem[" + (lane >= 0 && cur.lane_addr ? cur.lane_addr[lane] : "?") + "]", val: (lane >= 0 && cur.lane_val ? cur.lane_val[lane] : 0) };

  // layout nodes left→right: [src0] (sym [src1]) → [dst]
  const kind = kindOf(cur.op);
  let x = x0 + 16;
  const drawNode = (n, accent) => {
    const w = nodeW(n);
    drawReg(x, cy, w, n.label, n.val, accent);
    x += w;
    return x;
  };
  if (cur.op === "bra") {
    ctx.fillStyle = C.fg; ctx.font = "13px ui-monospace, monospace"; ctx.textBaseline = "middle";
    ctx.fillText("if r" + cur.ra + " ≠ 0 → then-path, else → else-path", x, cy + 14);
    if (cur.diverged) { ctx.fillStyle = C.diverge; ctx.fillText("   ⚡ lanes disagreed → warp serializes both paths", x + 320, cy + 14); }
    return top + boxH;
  }
  if (nodes.length === 0) {
    ctx.fillStyle = C.faint; ctx.font = "12px ui-monospace, monospace"; ctx.textBaseline = "middle";
    ctx.fillText(cur.op === "halt" ? "halt — warp finished" : "control flow", x, cy + 14);
    return top + boxH;
  }
  x = drawNode(nodes[0], KIND_COLOR[kind]);
  for (let i = 1; i < nodes.length; i++) {
    x = arrowSym(x, cy, sem.sym) ;
    x = drawNode(nodes[i], KIND_COLOR[kind]);
  }
  if (dst) {
    x = arrowSym(x, cy, "→");
    drawNode(dst, C.ok);
  }
  return top + boxH;
}

function asmText(e) {
  const r = (i) => "r" + i;
  switch (e.op) {
    case "mov": return `mov ${r(e.rd)}, ${e.imm}`;
    case "tid": return `tid ${r(e.rd)}`;
    case "iadd": return `iadd ${r(e.rd)}, ${r(e.ra)}, ${r(e.rb)}`;
    case "imul": return `imul ${r(e.rd)}, ${r(e.ra)}, ${r(e.rb)}`;
    case "slt": return `slt ${r(e.rd)}, ${r(e.ra)}, ${r(e.rb)}`;
    case "ld": return `ld ${r(e.rd)}, ${r(e.ra)}`;
    case "st": return `st ${r(e.ra)}, ${r(e.rb)}`;
    case "bra": return `bra ${r(e.ra)}, else, join`;
    case "jmp": return `jmp ${e.imm}`;
    default: return e.op;
  }
}
function nodeW(n) {
  const t = n.label + " = " + n.val;
  ctx.font = "12px ui-monospace, monospace";
  return Math.max(64, ctx.measureText(t).width + 20);
}
function drawReg(x, cy, w, label, val, accent) {
  const h = 30, y = cy;
  ctx.fillStyle = C.panel; rr(x, y, w - 12, h, 6); ctx.fill();
  ctx.strokeStyle = accent || C.border; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.textBaseline = "middle"; ctx.textAlign = "left";
  ctx.fillStyle = accent || C.fg; ctx.font = "bold 12px ui-monospace, monospace";
  ctx.fillText(label, x + 8, y + h / 2 - 6 + 6);
  ctx.fillStyle = C.fg; ctx.font = "12px ui-monospace, monospace";
  const lw = ctx.measureText(label).width;
  ctx.fillStyle = C.muted; ctx.fillText("= " + val, x + 12 + lw, y + h / 2);
}
function arrowSym(x, cy, sym) {
  ctx.fillStyle = C.muted; ctx.font = "15px ui-monospace, monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(sym || "→", x + 6, cy + 15);
  ctx.textAlign = "left";
  return x + 18;
}

function drawProgress() {
  const y = CH - 8, x0 = 14, w = CW - 28;
  ctx.fillStyle = C.idle; rr(x0, y, w, 5, 2); ctx.fill();
  const frac = anim.trace.length ? anim.step / anim.trace.length : 0;
  ctx.fillStyle = C.accent; rr(x0, y, w * frac, 5, 2); ctx.fill();
}
