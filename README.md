# Lithos

# Lithos

> **Lithos** (Greek *λίθος*, "stone") — the root of **lithography**, how chips are patterned onto
> silicon. A $0, from-scratch, simulation-first exploration of **computer architecture through
> measurement** — build the machine, measure why it stalls, then build the thing that fixes it.

Lithos is a solo, months-scale project whose real product is **understanding**: a cycle-accurate
**mini-GPU** and a **near-memory (PIM)** model, each built from scratch, each used to *measure* a
real architectural question rather than assert it. Everything here is honest and reproducible —
including a headline idea that was **tested and killed** before it wasted months.

---

## The journey (read this first — the repo is a story)

```
  1. A SPIKE, and an honest NO-GO.
     The project began as an EDGE dataflow accelerator betting it could beat a GPU at
     XGBoost tree-ensemble inference. A $0 pre-registered cost-model spike TESTED that
     bet and returned NO-GO: a competent branchless CPU already neutralizes the branching.
     → spike/ , spike-prereg.md , decisions.md D-008..D-014.  The thesis died cheaply. ✅

  2. THE PIVOT (D-015).
     New north star, owner-chosen: application-based learning + a use-case-driven result,
     with the design FORCED by an application. Concretely, a two-arm A→C arc.

  3. ARM A — a mini-GPU (SIMT). "Build a GPU; measure why it stalls."
     A cycle-accurate SIMT core (its own ISA, assembler, kernels) that MEASURES the effects
     that actually decide GPU performance — and a 9-chapter course teaching each one.
     → simt/ .  Finding: memory-bound kernels stall on the MEMORY WALL.

  4. ARM C — near-memory / PIM. "Stop moving the data; measure the win."
     A byte-accounting model of compute-in-memory, measured against a fair GPU baseline.
     → pim/ .  Finding (D-019): a real but BANKING-LIMITED data-movement win —
     it pays off for high-pooling recommendation features, not low-pooling ones.
```

**The arc worked:** *expose the bottleneck by measurement, then solve it — and quantify exactly
where the solution works and where it doesn't.* Both findings are honest and non-tautological.

---

## What's here (folder map)

| Path | What |
|---|---|
| **`simt/`** | **Arm A — the mini-GPU.** C++ cycle-accurate SIMT core + assembler + kernels + Python analysis. |
| **`simt/docs/`** | **A 9-chapter GPU-architecture course** (why-GPUs → SIMT → latency hiding → coalescing → divergence → cycle-accurate sim → ISA → reduction → the memory wall). Start here to learn. |
| **`pim/`** | **Arm C — the near-memory/PIM model.** C++ byte-accounting core + Python sweep/analysis. |
| **`pim/docs/`** | **A learning course on the memory wall & PIM** (near-memory computing, the bank model, the banking factor, byte-accounting/DMR, the crossover). |
| `spike/` | The concluded ($0) XGBoost cost-model spike (the NO-GO that triggered the pivot). |
| `architecture.md` | The technical design of both arms (mini-GPU + PIM), motivated then specified. |
| `decisions.md` | The full decision log (ADRs D-001…D-019) — every choice, its rationale, its status. |
| `handoff.md` | Master handoff: current state, task table, who does what next. |
| `spike-prereg.md`, `pim-prereg.md` | The pre-registrations (hypotheses, metrics, decision rules fixed *before* data). |
| `research/` | Early citation-backed landscape + use-case sweeps. |

---

## Quickstart (everything is $0)

**Toolchain:** MSVC + CMake + Ninja (Visual Studio Build Tools) for the C++ cores; Python
(numpy + matplotlib + pyyaml) for the analysis layers. This mirrors the gem5 pattern — a C++
engine scripted/analyzed by Python (decision D-016). From a shell where `vcvars64.bat` is sourced:

```powershell
# Arm A — the mini-GPU
cd simt
cmake -G Ninja -B build ; cmake --build build
build\test_simt.exe                        # 7 tests: correctness + cycle-accuracy + microarch
build\simt.exe kernels\vector_add.sasm 32  # run a kernel; see cycles / coalescing / divergence
python analysis\sweep.py                   # plot latency-hiding / coalescing / divergence / wall

# Arm C — near-memory / PIM
cd ..\pim
cmake -G Ninja -B build ; cmake --build build
build\test_pim.exe                         # 8 tests: byte-accounting + anti-tautology (crossover)
python run.py --config config.yaml         # sweep -> DMR crossover plot + decision_inputs.json
```

---

## The honest-scope contract
- **$0**, simulation-first, solo. Every phase scoped to finish.
- Every claim is tagged `[verified]` / `[believed]` / `[modelled]` / `[modelled-exact]`. Nothing here
  claims silicon performance — these are analytical/cycle-approximate teaching-and-measurement models,
  and each doc says exactly where it simplifies real hardware.
- Big decisions are **pre-registered** (hypotheses + metric + decision rule fixed before data), and a
  **NO-GO is always an acceptable, valuable outcome** — as the spike proved.
