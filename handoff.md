# Grove — Master Handoff

**Read this first.** It orients any agent (human or AI) picking up the project.
For decisions see `decisions.md`; for the design see `architecture.md`; for research evidence see
`research/`.

---

## 1. One-line mission
Build a from-scratch **EDGE dataflow processor** (ISA + compiler + cycle-accurate sim, $0) as the
**first open, minimal, readable, reproducible full-stack EDGE implementation with a measured result**.
*(Note: the original AI/tree-inference win framing was tested and retired — D-014 NO-GO; see §2. The
mission is now the general-purpose EDGE contribution, pending the discussAS re-frame.)*

## 2. Current state — SPIKE COMPLETE → **NO-GO**; ML framing retired, re-frame pending
Direction was locked (D-004…D-007) **conditional on the de-risk spike (D-008)**. The spike has now run.
**Update (2026-07-06):** designed & pre-registered (D-012, `spike-prereg.md`); coderAS built the rig;
a pre-data fairness bug on the synthetic smoke (ρ structurally < 1) was fixed by timestamped
**Amendment A1** (D-013) before any canonical data; coderAS implemented A1 (17 tests) and ran canonical
HIGGS. **Outcome: NO-GO (D-014)** — primary ρ = 0.84 (< 1.5×), overhead-free ceiling only 1.92× (< 2×),
rival R-A confirmed (predictable branches → a branchless N-wide scalar harvests the same cross-tree
parallelism EDGE offers). **R7 retired.** The EDGE core is unaffected; only the ML/tree-inference
use-case narrative is dropped. **Next: discussAS re-frames to a general-purpose open full-stack EDGE
contribution** (D-004/D-005/D-009), no GBDT/AI win claim. Full reasoning: `spike/out/conclusion.md`.
**Do NOT start the simulator on the ML thesis** — the gate returned NO-GO by design.

## 3. What happens next (ordered)
| # | Task | Owner | Blocking? | Status |
|---|------|-------|-----------|--------|
| 1 | Design + pre-register the cost-model spike | **experimentAS** (`handoffs/experimentAS-handoff.md`) | — | ✅ done → `spike-prereg.md`, D-012 |
| 2 | Implement the cost model + run it | **coderAS** (`handoffs/coderAS-handoff.md`, Stage A) | after #1 | ✅ rig built + A1 (D-013) implemented (17 tests) + canonical HIGGS run |
| 3 | Analyse → **GO / NO-GO** decision | experimentAS | after #2 | ✅ **NO-GO (D-014)** → `spike/out/conclusion.md` |
| 4 | *If GO:* phased EDGE sim build | coderAS (Stage B) | after #3 = GO | ✖ not triggered (spike = NO-GO) |
| 4' | ***NO-GO:* drop ML framing, pursue general-purpose EDGE** | **discussAS re-frame** | after #3 = NO-GO | ▶ **ACTIVE — next action** |
| 5 | *(optional)* Covertype confirmatory robustness re-run | coderAS | — | optional; documents external validity of NO-GO |

**The gate returned NO-GO.** Do not build the simulator on the tree-inference thesis; the next move is
the discussAS re-frame to a general-purpose EDGE contribution.

## 4. The non-negotiables
- **$0** — free tools only (Python now; Verilator/Icarus later). No paid hardware.
- **Simulation-first** — cycle-accurate software model is the primary artifact.
- **Honest scope** — tree-ensemble *inference*, **not** DL/LLM *training*. Don't let it drift.
- **The compiler + measurement rig is the differentiator, not the RTL** — budget effort accordingly.
- **Keep `decisions.md` and this file in sync** as work proceeds (append ADRs; update the status table).

## 5. The novelty claim (state it honestly)
Not "we invented EDGE" (TRIPS/E2 did). The contribution is **the first open, minimal, readable,
reproducible full-stack EDGE implementation + a measured control-bound result**, plus our
micro-innovations and the dynamic-dataflow tree extension. Be ready for the reviewer critique in D-010.

## 6. Risk ledger (live)
See `decisions.md` §"Risk & assumption ledger". **R7 (GBDT win magnitude) is RETIRED → NO-GO (D-014):**
the spike showed EDGE does not beat a competent branchless N-wide scalar by a portfolio-worthy margin
on GBDT batch-1 inference (ρ = 0.84; overhead-free ceiling 1.92×). Next live concern is the
general-purpose EDGE novelty defense (R2 absence-claim, R8 fixed-function-FPGA critique, D-010).

## 7. Pointers
- Design: `architecture.md`
- Decisions/ADRs: `decisions.md`
- **Spike pre-registration (the D-008 gate design): `spike-prereg.md`** ← build Stage A to this
- Spike brief: `handoffs/experimentAS-handoff.md`
- Build brief: `handoffs/coderAS-handoff.md`
- Evidence: `research/landscape.md`, `research/usecase.md`

## 8. Provenance
Chosen via a discussAS strategy engagement + two researchAS citation-backed sweeps (2026-07-05).
Claims are tagged `[verified]`/`[believed]`/`[guess]`; fabricated citations were caught and discarded.
