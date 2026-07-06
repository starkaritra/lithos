# Grove — Arm C: near-memory / PIM (does compute-in-memory beat the memory wall?)

**Arm A** (`../simt/`) built a mini-GPU and *measured* that memory-bound kernels stall on the
memory wall. **Arm C** asks the follow-up: if we put small compute units **inside the memory
banks** so only results cross the off-chip link, how much data movement do we actually save?

This is a **byte-accounting model** (pre-registered in `../pim-prereg.md`): for a kernel + config
it *counts* the off-chip link bytes moved by a competent coalesced **GPU baseline** vs a **PIM**
machine that aggregates in-bank, and reports:

> **DMR = baseline off-chip bytes ÷ PIM off-chip bytes** — "×-fewer-bytes." Bytes are *counted*,
> not modelled, so the primary result is assumption-free.

## The honest result (the point of the whole design)
The naive intuition says a bag pooling `L` rows sends `1` result, so DMR = `L`. But banking forces
PIM to ship **one partial per distinct bank touched**, `k(L,B) = B·(1−(1−1/B)^L)`, so the *real*
win is **≈ L/k**, far below the naive line — plus indices and output are charged to both sides. The
sweep locates the **crossover** where PIM stops being worth it:

```
  L (pooling):   1     8      16     40     64     256
  DMR (B=16):   1.0x  1.20x  1.49x  2.53x  3.72x  12.4x
                └ pure gather: PIM can't help        └ high pooling: PIM wins
```

So PIM helps **high-pooling** embedding features and barely helps low-pooling ones — a nuanced,
non-tautological finding, not a rigged "PIM always wins." The final GO/NO-GO is experimentAS's call
from `out/decision_inputs.json` against the pre-registered `../pim-prereg.md` §5 rule.

## The model (pim-prereg.md §3)
- Memory in `B` banks; row/element `i` in bank `place(i)`; each bank has a tiny compute unit.
- One shared off-chip **bandwidth cap** limits *both* machines (the fairness spine).
- **Kernel A — embedding-bag sum-pooling** (headline, DLRM/recsys): gather `L` scattered rows, sum
  to one vector. PIM pre-sums co-banked rows → ships `k` partials, not `L` rows.
- **Kernel B — array reduction** (de-risk endpoint): DMR ≈ `N/B` (the extreme-aggregation corner;
  proves the plumbing, not the decision).

## Build & run ($0 — MSVC + CMake + Ninja core, Python analysis; the gem5 pattern, D-016)
```
# from a shell where vcvars64.bat is sourced:
cmake -G Ninja -B build
cmake --build build
build\test_pim.exe                       # 8 accounting/anti-tautology tests
build\pim.exe --json                     # canonical accounting as JSON
python run.py --config config.yaml       # full sweep -> out/ (plots, results.csv,
                                          #   decision_inputs.json, provenance.json)
```

## Layout
```
include/pim/model.hpp   src/model.cpp    the byte-accounting core (baseline vs PIM)
src/main.cpp            CLI (--json + config flags)
tests/test_pim.cpp      correctness + anti-tautology tests (crossover, DMR < naive L)
config.yaml             canonical + pre-registered sweep (pim-prereg.md §6)
run.py                  Python analysis: sweep, crossover, plots, decision_inputs.json
```
Everything is `[modelled]`; the primary DMR is `[modelled-exact]` (bytes counted). See
`../pim-prereg.md` and `../decisions.md` (D-017 scope, D-018 pre-registration).
