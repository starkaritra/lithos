// lithos mini-GPU playground — loads the real WASM engine, runs kernels, and animates the
// enriched trace across three panels: (A) warps × lanes, (B) a word-level memory-block map
// that makes coalescing visible, and (C) a register-level data-flow view. Register/memory
// state shown is REPLAYED on the client from the engine's per-issue trace (every writing op
// emits its per-lane result), so nothing here re-implements the engine. A continuous render
// loop adds eased transitions, glow, and animated data-flow connectors.
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

// ---- palette ----------------------------------------------------------------
const C = {
  bg: "#0b0f16", panel: "#161b22", panel2: "#0e1420", grid: "#141b26",
  border: "#2a3340", fg: "#e6edf3", muted: "#8b949e", faint: "#5b6675",
  accent: "#58a6ff", alu: "#4c8dff", aluHi: "#8fbaff", mem: "#f0883e",
  memHi: "#ffb27a", branch: "#a371f7", branchHi: "#c9a9ff", idle: "#1b222c",
  diverge: "#f85149", ok: "#3fb950", okHi: "#7ee2a0",
};
const OP_KIND = { ld: "mem", st: "mem", bra: "branch", jmp: "branch" };
const kindOf = (op) => OP_KIND[op] || "alu";
const KIND = {
  alu: { base: C.alu, hi: C.aluHi }, mem: { base: C.mem, hi: C.memHi },
  branch: { base: C.branch, hi: C.branchHi },
};
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
let anim = {
  trace: [], step: 0, playing: false, acc: 0, stats: null,
  stepAt: 0, raf: null, pcLine: [],
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
let CW = 760, CH = 620;

// ---- boot -------------------------------------------------------------------
SimtModule().then((M) => {
  runKernel = M.cwrap("run_kernel", "string",
    ["string", "number", "number", "number", "number"]);
  buildExamples();
  $("editor").value = EXAMPLES["vector-add"];
  $("editor").addEventListener("input", () => setActiveExample(null));
  resizeCanvas();
  run();
  startRenderLoop();
});
window.addEventListener("resize", () => { resizeCanvas(); });

let activeExample = "vector-add";
function setActiveExample(name) {
  activeExample = name;
  for (const b of document.querySelectorAll(".examples button"))
    b.classList.toggle("active", b.dataset.name === name);
}
function buildExamples() {
  const box = $("examples");
  for (const name of Object.keys(EXAMPLES)) {
    const b = document.createElement("button");
    b.textContent = name;
    b.dataset.name = name;
    b.onclick = () => {
      $("editor").value = EXAMPLES[name];
      if (name === "latency-hiding") $("threads").value = 64;
      setActiveExample(name);
      run();
    };
    box.appendChild(b);
  }
  setActiveExample("vector-add");
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  CW = Math.max(360, canvas.clientWidth || 760);
  const nWarps = anim.stats ? anim.stats.n_warps : 1;
  CH = 320 + Math.max(0, nWarps - 1) * 28;
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
  anim.pcLine = mapPcToLine(src);
  resizeCanvas();
  renderStats(data.stats);
  setStep(0);
  anim.playing = true;
  $("play").textContent = "⏸ pause";
}

// map each instruction index (pc) to its 0-based source line (mirrors the assembler:
// skip blank lines, comment-only lines, and lone "label:" lines).
function mapPcToLine(src) {
  const out = [];
  const lines = src.split("\n");
  for (let i = 0; i < lines.length; i++) {
    let s = lines[i];
    const h = s.search(/[#;]/); if (h >= 0) s = s.slice(0, h);
    s = s.trim();
    if (!s) continue;
    if (/^\S+:$/.test(s)) continue;   // label-only
    out.push(i);
  }
  return out;
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

// ---- replay -----------------------------------------------------------------
function regfileBefore(step) {
  const s = anim.stats;
  const NR = 16, WS = s.warp_size, nW = s.n_warps;
  const regs = Array.from({ length: nW }, () =>
    Array.from({ length: WS }, () => new Int32Array(NR)));
  for (let i = 0; i < step - 1 && i < anim.trace.length; i++) {
    const e = anim.trace[i];
    if (e.writes_rd && e.lane_val && e.rd >= 0)
      for (let l = 0; l < WS; l++)
        if (e.mask & (1 << l)) regs[e.warp][l][e.rd] = e.lane_val[l];
  }
  return regs;
}
const firstActiveLane = (mask, WS) => { for (let l = 0; l < WS; l++) if (mask & (1 << l)) return l; return -1; };

// ---- animation clock --------------------------------------------------------
function now() { return (typeof performance !== "undefined" ? performance.now() : Date.now()); }
function setStep(n) { anim.step = n; anim.stepAt = now(); }
const easeOut = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);

function startRenderLoop() {
  if (anim.raf) return;
  const tick = (t) => { renderFrame(t); anim.raf = requestAnimationFrame(tick); };
  anim.raf = requestAnimationFrame(tick);
}
let lastT = 0;
function renderFrame(t) {
  t = t || now();
  if (anim.playing && anim.trace.length) {
    const eps = parseInt($("speed").value, 10) || 12;
    if (!lastT) lastT = t;
    anim.acc += ((t - lastT) / 1000) * eps;
    lastT = t;
    while (anim.acc >= 1 && anim.step < anim.trace.length) { setStep(anim.step + 1); anim.acc -= 1; }
    if (anim.step >= anim.trace.length) { anim.playing = false; $("play").textContent = "▶ play"; anim.acc = 0; }
  } else { lastT = t; }
  draw(t);
}
$("play").onclick = () => {
  if (anim.playing) { anim.playing = false; $("play").textContent = "▶ play"; }
  else { if (anim.step >= anim.trace.length) setStep(0); anim.playing = true; lastT = 0; $("play").textContent = "⏸ pause"; }
};
$("step").onclick = () => { anim.playing = false; $("play").textContent = "▶ play"; if (anim.step < anim.trace.length) setStep(anim.step + 1); };
$("reset").onclick = () => { anim.playing = false; $("play").textContent = "▶ play"; setStep(0); };
$("run").onclick = run;

// ---- drawing primitives -----------------------------------------------------
function rr(x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function vgrad(x, y, h, top, bot) { const g = ctx.createLinearGradient(x, y, x, y + h); g.addColorStop(0, top); g.addColorStop(1, bot); return g; }
function glow(color, blur, fn) { ctx.save(); ctx.shadowColor = color; ctx.shadowBlur = blur; fn(); ctx.restore(); }
function panelTitle(text, x, y) {
  ctx.fillStyle = C.accent; rr(x, y - 8, 3, 10, 1.5); ctx.fill();
  ctx.fillStyle = C.muted; ctx.font = "600 11px ui-monospace, Consolas, monospace";
  ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
  ctx.fillText(text.toUpperCase(), x + 9, y);
}
function bgGrid() {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, CW, CH);
  ctx.fillStyle = C.grid;
  for (let gx = 12; gx < CW; gx += 26)
    for (let gy = 12; gy < CH; gy += 26) { ctx.beginPath(); ctx.arc(gx, gy, 0.7, 0, 7); ctx.fill(); }
}

let laneGeom = null;

function draw(t) {
  t = t || now();
  bgGrid();
  const s = anim.stats;
  if (!s) return;
  const cur = anim.step > 0 ? anim.trace[anim.step - 1] : null;
  const enter = easeOut((t - anim.stepAt) / 200);   // 0→1 reveal since this issue
  const pulse = 0.5 + 0.5 * Math.sin(t / 420);        // breathing glow
  $("cyclereadout").textContent = cur
    ? `cycle ${cur.cycle}  ·  issue ${anim.step}/${anim.trace.length}`
    : "cycle —";

  const padX = 16;
  let y = 24;
  y = drawWarps(padX, y, s, cur, enter, pulse);
  y += 16;
  y = drawMemory(padX, y, s, cur, enter, pulse, t);
  y += 16;
  drawDataflow(padX, y, s, cur, enter, pulse);
  drawProgress();
}

// Panel A — warps × lanes ------------------------------------------------------
function drawWarps(x0, y0, s, cur, enter, pulse) {
  panelTitle("warps × lanes  ·  lit = active this issue · color = op kind", x0, y0);
  let y = y0 + 10;
  const WS = s.warp_size, nW = s.n_warps;
  const labelW = 52;
  const gridW = CW - x0 * 2 - labelW;
  const cell = Math.max(8, Math.min(20, Math.floor(gridW / WS) - 2));
  const gap = 2, rowH = cell + 9;
  laneGeom = { x0: x0 + labelW, y, cell, gap, rowH, WS };

  for (let w = 0; w < nW; w++) {
    const ry = y + w * rowH;
    const issuing = cur && cur.warp === w;
    if (issuing) {
      ctx.fillStyle = "rgba(88,166,255," + (0.10 * enter).toFixed(3) + ")";
      rr(x0 - 2, ry - 4, CW - x0 * 2 + 4, cell + 8, 6); ctx.fill();
    }
    ctx.fillStyle = issuing ? C.accent : C.muted;
    ctx.font = (issuing ? "bold " : "") + "11px ui-monospace, Consolas, monospace";
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.fillText("warp " + w, x0, ry + cell / 2);
    for (let l = 0; l < WS; l++) {
      const cx = laneGeom.x0 + l * (cell + gap);
      const active = issuing && (cur.mask & (1 << l));
      if (active) {
        const k = KIND[kindOf(cur.op)];
        const sc = 0.82 + 0.18 * enter;
        const inset = (cell * (1 - sc)) / 2;
        glow(k.base, 8 * enter * (0.7 + 0.3 * pulse), () => {
          ctx.fillStyle = vgrad(cx, ry, cell, k.hi, k.base);
          rr(cx + inset, ry + inset, cell * sc, cell * sc, 3); ctx.fill();
        });
        if (cur.diverged) { ctx.strokeStyle = C.diverge; ctx.lineWidth = 1.5; ctx.stroke(); }
      } else {
        ctx.fillStyle = C.idle; rr(cx, ry, cell, cell, 3); ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.03)"; ctx.lineWidth = 1; ctx.stroke();
      }
    }
    if (issuing && cur.diverged) {
      ctx.fillStyle = C.diverge; ctx.font = "bold 11px ui-monospace, monospace"; ctx.textAlign = "left";
      ctx.fillText("⚡ diverged", laneGeom.x0 + WS * (cell + gap) + 6, ry + cell / 2);
    }
  }
  return y + nW * rowH;
}

// Panel B — memory-block map ---------------------------------------------------
function drawMemory(x0, y0, s, cur, enter, pulse, t) {
  const segW = s.segment_words;
  const isMem = cur && cur.is_mem && cur.lane_addr;
  const quality = isMem ? (cur.txns <= 4 ? "coalesced" : cur.txns >= s.warp_size / 2 ? "scattered" : "partly coalesced") : "";
  panelTitle(
    isMem ? `global memory  ·  ${cur.op.toUpperCase()} — ${cur.txns} transaction${cur.txns === 1 ? "" : "s"} (${quality})`
          : "global memory  ·  no memory traffic this instruction",
    x0, y0);
  const top = y0 + 12, boxH = 52;
  ctx.fillStyle = vgrad(x0, top, boxH, C.panel2, C.bg);
  rr(x0, top, CW - x0 * 2, boxH, 8); ctx.fill();
  ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();

  if (!isMem) {
    ctx.fillStyle = C.faint; ctx.font = "12px ui-monospace, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("memory is idle — this is an ALU / branch instruction", CW / 2, top + boxH / 2);
    return top + boxH;
  }

  const WS = s.warp_size;
  const laneAddr = [];
  for (let l = 0; l < WS; l++)
    if ((cur.mask & (1 << l)) && cur.lane_addr[l] >= 0)
      laneAddr.push({ lane: l, addr: cur.lane_addr[l] });
  const segs = [...new Set(laneAddr.map((a) => Math.floor(a.addr / segW)))].sort((a, b) => a - b);

  const areaX = x0 + 10, areaW = CW - x0 * 2 - 20;
  const nSeg = segs.length, blockGap = 8;
  const blockW = Math.max(9, Math.min(130, (areaW - blockGap * (nSeg - 1)) / nSeg));
  const showWords = blockW >= segW * 7;
  const by = top + 9, bh = boxH - 26;
  const segX = {};

  for (let i = 0; i < nSeg; i++) {
    const bx = areaX + i * (blockW + blockGap);
    segX[segs[i]] = bx + blockW / 2;
    ctx.fillStyle = C.idle; rr(bx, by, blockW, bh, 4); ctx.fill();
    ctx.strokeStyle = "rgba(240,136,62,0.25)"; ctx.lineWidth = 1; ctx.stroke();
    if (showWords) {
      const wCellW = (blockW - 6) / segW;
      for (let wi = 0; wi < segW; wi++) {
        const wx = bx + 3 + wi * wCellW, addr = segs[i] * segW + wi;
        const hit = laneAddr.some((a) => a.addr === addr);
        if (hit) glow(C.mem, 6 * enter, () => { ctx.fillStyle = vgrad(wx, by + 5, bh - 18, C.memHi, C.mem); rr(wx + 0.5, by + 5, wCellW - 1, bh - 18, 2); ctx.fill(); });
        else { ctx.fillStyle = "#161d27"; rr(wx + 0.5, by + 5, wCellW - 1, bh - 18, 2); ctx.fill(); }
      }
    } else {
      const n = laneAddr.filter((a) => Math.floor(a.addr / segW) === segs[i]).length;
      const a = Math.min(1, 0.3 + n / WS) * (0.6 + 0.4 * enter);
      ctx.fillStyle = `rgba(240,136,62,${a.toFixed(3)})`;
      rr(bx + 2, by + 2, blockW - 4, bh - 4, 3); ctx.fill();
    }
    ctx.fillStyle = C.faint; ctx.font = "9px ui-monospace, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText("seg " + segs[i], bx + blockW / 2, by + bh + 3);
  }

  // curved, animated connectors: issuing lane → the word/segment it touches
  if (laneGeom && laneGeom.WS === WS) {
    const lg = laneGeom;
    const laneRowY = lg.y + cur.warp * lg.rowH + lg.cell;
    ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.lineDashOffset = -(t / 40) % 9;
    for (const a of laneAddr) {
      const lx = lg.x0 + a.lane * (lg.cell + lg.gap) + lg.cell / 2;
      let tx = segX[Math.floor(a.addr / segW)];
      if (showWords) { const wCellW = (blockW - 6) / segW; tx = tx - blockW / 2 + 3 + (a.addr % segW) * wCellW + wCellW / 2; }
      const g = ctx.createLinearGradient(lx, laneRowY, tx, by);
      g.addColorStop(0, "rgba(240,136,62,0.30)"); g.addColorStop(1, "rgba(255,178,122," + (0.55 + 0.35 * enter).toFixed(3) + ")");
      ctx.strokeStyle = g;
      const my = (laneRowY + by) / 2;
      ctx.beginPath(); ctx.moveTo(lx, laneRowY + 2);
      ctx.bezierCurveTo(lx, my, tx, my, tx, by); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  return top + boxH;
}

// Panel C — register data-flow -------------------------------------------------
function drawDataflow(x0, y0, s, cur, enter, pulse) {
  panelTitle("data flow  ·  one representative lane", x0, y0);
  const top = y0 + 12, boxH = 96;
  ctx.fillStyle = vgrad(x0, top, boxH, C.panel2, C.bg);
  rr(x0, top, CW - x0 * 2, boxH, 8); ctx.fill();
  ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();
  if (!cur) return top + boxH;

  const WS = s.warp_size;
  const lane = firstActiveLane(cur.mask, WS);
  const sem = OP_SEM[cur.op] || { src: [], dst: false, sym: "" };
  const kind = kindOf(cur.op), kc = KIND[kind];
  const regs = regfileBefore(anim.step);
  const before = (lane >= 0) ? regs[cur.warp][lane] : new Int32Array(16);

  // op pill + assembly + lane/line info
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  const pill = cur.op.toUpperCase();
  ctx.font = "bold 11px ui-monospace, monospace";
  const pw = ctx.measureText(pill).width + 16;
  ctx.fillStyle = kc.base; rr(x0 + 12, top + 8, pw, 18, 9); ctx.fill();
  ctx.fillStyle = "#0b0f16"; ctx.fillText(pill, x0 + 20, top + 17);
  ctx.fillStyle = C.fg; ctx.font = "13px ui-monospace, Consolas, monospace";
  ctx.fillText(asmText(cur), x0 + 20 + pw, top + 17);
  const srcLine = anim.pcLine[cur.pc];
  ctx.fillStyle = C.faint; ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText((srcLine != null ? "line " + (srcLine + 1) + "  ·  " : "") + "lane " + lane +
    (cur.after_stall ? "  ·  ⏱ after a STALL" : ""), CW - x0 - 12, top + 17);
  ctx.textAlign = "left";

  const cy = top + 52;
  if (cur.op === "bra") {
    ctx.fillStyle = C.fg; ctx.font = "13px ui-monospace, monospace"; ctx.textBaseline = "middle";
    ctx.fillText("if  r" + cur.ra + " ≠ 0  → then-path,  else → else-path", x0 + 16, cy + 12);
    if (cur.diverged) { ctx.fillStyle = C.diverge; ctx.fillText("⚡ lanes disagreed → warp serializes both paths", x0 + 16, cy + 32); }
    return top + boxH;
  }
  const nodes = [];
  for (const src of sem.src) {
    if (src === "imm") nodes.push({ label: "imm", val: cur.imm });
    else if (src === "tid") nodes.push({ label: "tid", val: (lane >= 0 ? cur.lane_val[lane] : 0) });
    else if (src === "mem") nodes.push({ label: "mem[" + (lane >= 0 && cur.lane_addr ? cur.lane_addr[lane] : "?") + "]", val: (lane >= 0 && cur.lane_val ? cur.lane_val[lane] : 0), mem: true });
    else if (src === "ra") nodes.push({ label: "r" + cur.ra, val: before[cur.ra] });
    else if (src === "rb") nodes.push({ label: "r" + cur.rb, val: before[cur.rb] });
  }
  let dst = null;
  if (sem.dst) dst = { label: "r" + cur.rd, val: (lane >= 0 && cur.lane_val ? cur.lane_val[lane] : 0) };
  else if (cur.is_store) dst = { label: "mem[" + (lane >= 0 && cur.lane_addr ? cur.lane_addr[lane] : "?") + "]", val: (lane >= 0 && cur.lane_val ? cur.lane_val[lane] : 0), mem: true };

  if (nodes.length === 0) {
    ctx.fillStyle = C.faint; ctx.font = "12px ui-monospace, monospace"; ctx.textBaseline = "middle";
    ctx.fillText(cur.op === "halt" ? "halt — warp finished" : "control flow", x0 + 16, cy + 12);
    return top + boxH;
  }
  let x = x0 + 16;
  const draw1 = (n, col, colHi, flash) => { x = drawReg(x, cy, n, col, colHi, flash, enter, pulse); };
  draw1(nodes[0], kc.base, kc.hi, false);
  for (let i = 1; i < nodes.length; i++) { x = arrowSym(x, cy, sem.sym); draw1(nodes[i], kc.base, kc.hi, false); }
  if (dst) { x = arrowSym(x, cy, "→"); draw1(dst, C.ok, C.okHi, true); }
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
function drawReg(x, cy, n, col, colHi, flash, enter, pulse) {
  const label = n.label, val = String(n.val);
  ctx.font = "bold 12px ui-monospace, monospace"; const lw = ctx.measureText(label).width;
  ctx.font = "12px ui-monospace, monospace"; const vw = ctx.measureText("= " + val).width;
  const w = Math.max(60, lw + vw + 22), h = 32, y = cy;
  const doFill = () => {
    ctx.fillStyle = vgrad(x, y, h, C.panel, C.panel2); rr(x, y, w, h, 7); ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = flash ? 1.5 + enter : 1.4; ctx.stroke();
  };
  if (flash) glow(col, 12 * (1 - enter) + 3 * pulse, doFill); else doFill();
  ctx.textBaseline = "middle"; ctx.textAlign = "left";
  ctx.fillStyle = colHi || col; ctx.font = "bold 12px ui-monospace, monospace";
  ctx.fillText(label, x + 9, y + h / 2);
  ctx.fillStyle = C.muted; ctx.font = "12px ui-monospace, monospace";
  ctx.fillText("= " + val, x + 9 + lw + 6, y + h / 2);
  return x + w + 6;
}
function arrowSym(x, cy, sym) {
  ctx.fillStyle = C.faint; ctx.font = "16px ui-monospace, monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(sym || "→", x + 8, cy + 16);
  ctx.textAlign = "left";
  return x + 20;
}
function drawProgress() {
  const y = CH - 9, x0 = 16, w = CW - 32;
  ctx.fillStyle = C.idle; rr(x0, y, w, 5, 2.5); ctx.fill();
  const frac = anim.trace.length ? anim.step / anim.trace.length : 0;
  const g = ctx.createLinearGradient(x0, 0, x0 + w, 0);
  g.addColorStop(0, C.accent); g.addColorStop(1, "#7ee2a0");
  ctx.fillStyle = g; rr(x0, y, Math.max(0, w * frac), 5, 2.5); ctx.fill();
}
