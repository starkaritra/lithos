# Grove mini-GPU — browser playground

**Program a GPU in your browser.** Write a kernel in the tiny SIMT assembly, run it on the
**real cycle-accurate `simt_core` engine** (compiled to WebAssembly — the same C++ that passes
the native test suite), and *watch it execute*: warps issue in lockstep, memory coalesces,
branches diverge, and latency hides behind other warps.

Everything runs **client-side** (WASM) — no backend, $0 to host (e.g. GitHub Pages).

## What you see
- **Warp/lane grid** — each row is a warp, each cell a lane. On every issue, the active lanes
  light up, colored by op kind (ALU / memory / branch). Idle/masked lanes stay dark.
- **Memory strip** — a memory op lights up its *transactions*: few + green = coalesced, many +
  red = uncoalesced (chapter 04).
- **⚡ divergence** — flashes on a warp when a branch splits it (chapter 05).
- **stall notes** — when the scheduler had nothing ready and the clock jumped (latency being
  paid); run the `latency-hiding` example with `threads=64` to watch a 2nd warp fill the gaps.
- Live **stats** (cycles, transactions, divergences) and playback controls (play/step/reset/speed).

## Build & serve (local)
```powershell
# 1) build the engine to WASM ($0, Emscripten — see repo setup)
powershell -File ..\build_wasm.ps1      # -> simt.js + simt.wasm here

# 2) serve over HTTP (WASM can't load from file://)
python -m http.server 8099              # then open http://localhost:8099
```
Node smoke test of the WASM engine (correct stats + trace): `node ..\web_smoke.cjs`.

## Files
```
index.html   layout            app.js     load WASM, run, animate the trace
style.css    dark theme        simt.js/.wasm   built engine (gitignored — regenerate)
```
The engine is built from `../src/{isa,assembler,core,wasm_api}.cpp` via `../build_wasm.ps1`
(`wasm_api.cpp` is the only browser-specific glue; the simulator itself is unchanged and the
`enable_trace()` path is nullptr-guarded so the native build + tests are unaffected).
