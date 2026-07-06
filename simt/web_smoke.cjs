// Node smoke test: load the WASM engine and run kernels through it, mirroring what the
// browser will do. Verifies the real simt_core produces correct stats + a trace in JS.
const SimtModule = require("./web/simt.js");

const vadd = [
  "mov r1,0","mov r2,32","mov r3,64","tid r0",
  "iadd r4,r1,r0","iadd r5,r2,r0","iadd r6,r3,r0",
  "ld r7,r4","ld r8,r5","iadd r9,r7,r8","st r6,r9","halt",
].join("\n");

const diverge = [
  "mov r5,16","tid r0","slt r1,r0,r5","bra r1,else,join",
  "mov r2,100","iadd r3,r2,r0","st r0,r3","jmp done",
  "else:","mov r2,200","iadd r3,r2,r0","st r0,r3","join:","done:","halt",
].join("\n");

SimtModule().then((M) => {
  const run = M.cwrap("run_kernel", "string",
    ["string", "number", "number", "number", "number"]);

  const a = JSON.parse(run(vadd, 32, 200, 8, 8));
  console.log("vector_add:", JSON.stringify(a.stats));
  console.log("  trace events:", a.trace.length, "first:", JSON.stringify(a.trace[0]));

  const b = JSON.parse(run(diverge, 32, 200, 8, 8));
  console.log("divergence:", JSON.stringify(b.stats));
  const div = b.trace.filter((e) => e.diverged).length;
  console.log("  diverged events in trace:", div);

  const err = JSON.parse(run("bogus r0", 32, 200, 8, 8));
  console.log("error handling:", JSON.stringify(err));

  const ok = a.stats.cycles === 681 && a.stats.mem_transactions === 12 &&
             b.stats.divergent_branches === 1 && !!err.error;
  console.log(ok ? "WASM SMOKE PASS" : "WASM SMOKE FAIL");
  process.exit(ok ? 0 : 1);
});
