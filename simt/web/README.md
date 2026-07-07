# Lithos mini-GPU — browser playground

**[▶ Live: lithos-c0w.pages.dev](https://lithos-c0w.pages.dev/)** — no install, runs in your browser.

**Program a GPU in your browser.** Write a kernel in the tiny SIMT assembly, run it on the
**real cycle-accurate `simt_core` engine** (compiled to WebAssembly — the same C++ that passes
the native test suite), and *watch it execute*: warps issue in lockstep, memory coalesces,
branches diverge, and latency hides behind other warps.

Everything runs **client-side** (WASM) — no backend, $0 to host (deployed to Cloudflare Pages via
`.github/workflows/deploy-cloudflare.yml`, which rebuilds the WASM from source on every push).

## What you see (three panels)
- **Warps × lanes** — each row is a warp, each cell a lane. On every issue the active lanes light
  up (with a glow), colored by op kind (ALU / memory / branch); idle/masked lanes stay dark, and a
  diverged branch outlines the split lanes in red.
- **Global memory (block map)** — a memory op draws the segments it touches as blocks of words,
  with animated connectors from each issuing lane to the exact word it addresses. Many lanes → a
  few blocks = **coalesced**; a fan-out to many blocks = **scattered** (chapter 04). The header
  shows the transaction count and verdict.
- **Data flow** — the current instruction for one representative lane as `source values → op →
  destination register`, using the *real* per-lane values (replayed on the client from the trace),
  with a value-flash on the destination and the executing source line / lane.
- Plus **⚡ divergence** flags (chapter 05), **stall** notes when the clock jumped (latency being
  paid — run `latency-hiding` with `threads=64` to watch a 2nd warp fill the gaps), live **stats**
  (cycles, transactions, divergences), and playback controls (play / step / reset / speed).

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
