// lithos mini-GPU playground — loads the real WASM engine, runs kernels, animates the trace.
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
  "coalesced": `# addr = tid  -> 32 lanes hit 4 segments (few transactions)
tid r0
ld  r1, r0
halt`,
  "scattered": `# addr = tid*8 -> every lane in its own segment (32 transactions!)
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

const OP_KIND = {
  ld: "mem", st: "mem", bra: "branch", jmp: "branch",
};
function kindOf(op) { return OP_KIND[op] || "alu"; }
const COLORS = { alu: "#388bfd", mem: "#f0883e", branch: "#a371f7" };

let runKernel = null;      // the WASM-exported function
let anim = { trace: [], step: 0, playing: false, raf: null, acc: 0, stats: null };

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// ---- boot: load the WASM module ---------------------------------------------
SimtModule().then((M) => {
  runKernel = M.cwrap("run_kernel", "string",
    ["string", "number", "number", "number", "number"]);
  buildExamples();
  $("editor").value = EXAMPLES["vector-add"];
  run();
});

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

// ---- run ---------------------------------------------------------------------
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

// ---- animation ---------------------------------------------------------------
function startPlay() { anim.playing = true; $("play").textContent = "⏸ pause"; loop(); }
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
  const eps = parseInt($("speed").value, 10); // events per second
  if (!lastT) lastT = t || 0;
  const dt = ((t || 0) - lastT) / 1000;
  lastT = t || 0;
  anim.acc += dt * eps;
  while (anim.acc >= 1 && anim.step < anim.trace.length) { anim.step++; anim.acc -= 1; }
  draw();
  if (anim.step >= anim.trace.length) { stopPlay(); anim.acc = 0; lastT = 0; return; }
  anim.raf = requestAnimationFrame(loop);
}

function draw() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const s = anim.stats;
  if (!s) return;
  const nWarps = s.n_warps, WS = s.warp_size;
  const cur = anim.step > 0 ? anim.trace[anim.step - 1] : null;
  $("cyclereadout").textContent = cur ? `cycle ${cur.cycle}  ·  issue ${anim.step}/${anim.trace.length}` : "cycle —";

  // layout: warp rows of WS lane cells
  const padX = 70, padTop = 14;
  const gridW = W - padX - 16;
  const cell = Math.min(18, Math.floor(gridW / WS) - 2);
  const rowH = cell + 10;

  for (let w = 0; w < nWarps; w++) {
    const y = padTop + w * rowH;
    ctx.fillStyle = "#8b949e"; ctx.font = "11px monospace"; ctx.textBaseline = "middle";
    ctx.fillText("warp " + w, 8, y + cell / 2);
    for (let l = 0; l < WS; l++) {
      const x = padX + l * (cell + 2);
      let fill = "#21262d";          // idle
      if (cur && cur.warp === w && (cur.mask & (1 << l))) fill = COLORS[kindOf(cur.op)];
      ctx.fillStyle = fill;
      ctx.fillRect(x, y, cell, cell);
    }
    // divergence badge on the issuing warp
    if (cur && cur.warp === w && cur.diverged) {
      ctx.fillStyle = "#f85149"; ctx.font = "bold 11px monospace";
      ctx.fillText("⚡ diverge", padX + WS * (cell + 2) + 6, y + cell / 2);
    }
  }

  // current instruction label
  const infoY = padTop + nWarps * rowH + 12;
  if (cur) {
    ctx.fillStyle = "#c9d1d9"; ctx.font = "13px monospace"; ctx.textBaseline = "top";
    const stallTxt = cur.stall ? "   (after a STALL — nothing was ready; latency being paid)" : "";
    ctx.fillText(`warp ${cur.warp}  pc ${cur.pc}  ${cur.op}${stallTxt}`, 8, infoY);
  }

  // memory-transaction strip (only for memory ops): light `txns` of WS cells
  const memY = infoY + 26;
  ctx.fillStyle = "#8b949e"; ctx.font = "11px monospace"; ctx.textBaseline = "middle";
  ctx.fillText("memory", 8, memY + cell / 2);
  const txns = cur && cur.txns ? cur.txns : 0;
  for (let i = 0; i < WS; i++) {
    const x = padX + i * (cell + 2);
    let fill = "#21262d";
    if (i < txns) fill = txns <= 8 ? "#3fb950" : "#f85149"; // few=coalesced(green), many=red
    ctx.fillStyle = fill; ctx.fillRect(x, memY, cell, cell);
  }
  if (cur && cur.txns) {
    ctx.fillStyle = "#c9d1d9"; ctx.font = "12px monospace"; ctx.textBaseline = "middle";
    const q = cur.txns <= 8 ? "coalesced" : "uncoalesced";
    ctx.fillText(`${cur.txns} transactions (${q})`, padX + WS * (cell + 2) + 6, memY + cell / 2);
  }

  // progress bar
  const barY = H - 14;
  ctx.fillStyle = "#21262d"; ctx.fillRect(padX, barY, gridW, 6);
  ctx.fillStyle = "#58a6ff";
  const frac = anim.trace.length ? anim.step / anim.trace.length : 0;
  ctx.fillRect(padX, barY, gridW * frac, 6);
}
