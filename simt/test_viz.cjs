// Headless drawing harness: runs the SHIPPED app.js in a sandbox with a mock canvas and
// the REAL enriched traces, driving draw() across every issue of every example to catch
// runtime errors in the visualization code (which the browser can't be reached to test here).
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const SimtModule = require("./web/simt.js");

const EXAMPLES = {
  "vector-add": ["mov r1,0","mov r2,32","mov r3,64","tid r0","iadd r4,r1,r0","iadd r5,r2,r0","iadd r6,r3,r0","ld r7,r4","ld r8,r5","iadd r9,r7,r8","st r6,r9","halt"].join("\n"),
  "divergence": ["mov r5,16","tid r0","slt r1,r0,r5","bra r1,else,join","mov r2,100","iadd r3,r2,r0","st r0,r3","jmp done","else:","mov r2,200","iadd r3,r2,r0","st r0,r3","join:","done:","halt"].join("\n"),
  "coalesced": ["tid r0","ld r1,r0","halt"].join("\n"),
  "scattered": ["tid r0","mov r2,8","imul r3,r0,r2","ld r1,r3","halt"].join("\n"),
};

function mockCtx() {
  const noop = () => {};
  const grad = { addColorStop: noop };
  return new Proxy({
    measureText: (t) => ({ width: String(t).length * 7 }),
    createLinearGradient: () => grad, createRadialGradient: () => grad,
    setTransform: noop, clearRect: noop, fillRect: noop, strokeRect: noop,
    beginPath: noop, moveTo: noop, lineTo: noop, arcTo: noop, arc: noop,
    bezierCurveTo: noop, quadraticCurveTo: noop, setLineDash: noop,
    closePath: noop, fill: noop, stroke: noop, fillText: noop, save: noop, restore: noop,
    fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", textAlign: "", textBaseline: "",
    shadowColor: "", shadowBlur: 0, lineDashOffset: 0,
  }, { get: (o, k) => (k in o ? o[k] : noop), set: (o, k, v) => (o[k] = v, true) });
}
function elStub() {
  return { value: "", textContent: "", innerHTML: "", onclick: null,
    dataset: {}, style: {}, classList: { toggle: () => {}, add: () => {}, remove: () => {} },
    appendChild: () => {}, addEventListener: () => {}, querySelectorAll: () => [] };
}

SimtModule().then((M) => {
  const run = M.cwrap("run_kernel", "string", ["string","number","number","number","number"]);

  let errors = 0, drawn = 0;
  for (const [name, src] of Object.entries(EXAMPLES)) {
    const threads = name === "latency-hiding" ? 64 : 32;
    const json = run(src, threads, 200, 8, 8);
    const data = JSON.parse(json);
    if (data.error) { console.log("PARSE ERROR", name, data.error); errors++; continue; }

    // build a sandbox that runs the shipped app.js with mocks + this precomputed trace
    const ctx = mockCtx();
    const editor = elStub(); editor.value = src;
    const stub = elStub;
    const els = { editor, canvas: { getContext: () => ctx, clientWidth: 760, width: 0, height: 0, style: {} },
      threads: { value: String(threads) }, latency: { value: "200" }, segwords: { value: "8" },
      error: stub(), stats: stub(), examples: stub(), play: stub(), step: stub(),
      reset: stub(), run: stub(), speed: { value: "12" }, cyclereadout: stub() };
    const sandbox = {
      window: { devicePixelRatio: 1, addEventListener: () => {} },
      document: { getElementById: (id) => els[id] || stub(), createElement: () => stub(),
        querySelectorAll: () => [] },
      SimtModule: () => Promise.resolve({ cwrap: () => run }),
      requestAnimationFrame: () => 0, cancelAnimationFrame: () => {},
      performance: { now: () => 0 }, Date,
      parseInt, Int32Array, Set, Math, JSON, Array, console, isNaN, String, Number,
    };
    sandbox.globalThis = sandbox;
    const code = fs.readFileSync(path.join(__dirname, "web", "app.js"), "utf8")
      + "\n;globalThis.__api = { get anim(){return anim;}, draw, run };";
    vm.createContext(sandbox);
    try { vm.runInContext(code, sandbox); } catch (e) { console.log("LOAD ERROR", name, e.message); errors++; continue; }

    const api = sandbox.__api;
    api.anim.trace = data.trace; api.anim.stats = data.stats;
    // drive draw across every issue + reset state
    for (let step = 0; step <= data.trace.length; step++) {
      api.anim.step = step;
      try { api.draw(); drawn++; } catch (e) { console.log(`DRAW ERROR ${name} step ${step}:`, e.message); errors++; break; }
    }
    console.log(`  ${name}: ${data.trace.length} issues drawn OK`);
  }
  console.log(errors === 0 ? `VIZ HARNESS PASS (${drawn} draws)` : `VIZ HARNESS FAIL (${errors} errors)`);
  process.exit(errors === 0 ? 0 : 1);
});
