# Grove — Master Handoff

**Read this first.** It orients any agent (human or AI) picking up the project.
For decisions see `decisions.md`; for the design see `architecture.md`; for research evidence see
`research/`.

---

## 1. One-line mission
**Application-driven, $0, from-scratch computer-architecture build to LEARN GPU architecture (and solve
a real bottleneck).** Two arms — **A→C**: (A) build a mini-**GPU (SIMT)** core + minimal ISA + tiny
CUDA-like compiler + cycle-accurate sim that runs real parallel kernels and *exposes the memory wall*;
(C) build a **near-memory / PIM** architecture + ISA that *solves* it and **measures** the data-movement
win. North star = *application-based learning + a use-case-driven contribution* (D-015).
*(History: the project began as an EDGE dataflow accelerator for XGBoost tree-inference; that thesis was
pre-registered, tested, and retired — D-014 NO-GO. The EDGE/dataflow + cycle-accurate-sim + measurement
discipline carries over. See §2.)*

## 2. Current state — PIVOTED (post-NO-GO). Arm A built; **Arm C measured → CONDITIONAL-GO (D-019).**
The original EDGE/tree-inference thesis (D-004…D-007) was gated on a de-risk spike (D-008); the spike
**ran and returned NO-GO** (D-012 pre-reg → D-013 Amendment A1 → **D-014 NO-GO**: canonical HIGGS
ρ = 0.84 < 1.5×, overhead-free ceiling only 1.92×, rival R-A confirmed; R7 retired; full reasoning in
`spike/out/conclusion.md`). The owner then rejected both the general-purpose-EDGE fallback and a
novelty-free playground, and chose the **A→C arc (D-015)**: a mini-GPU that exposes the memory wall,
then PIM that solves it. **Arm A (v1) is built** (SIMT sim + compiler + kernels; memory wall measured:
one access ≈ 225 arith ops; `simt/docs/` 9-chapter course). **Arm C (v2) is now MEASURED:** coderAS
built the byte-accounting model to the pre-registration (`pim-prereg.md`, D-018) and experimentAS
rendered the verdict — **CONDITIONAL-GO (D-019), calibrated confidence ~0.82** (canonical DMR = 2.53×,
gray zone; the data-movement win is real but **banking-limited** and scoped to **high-pooling embedding
features L ≳ 64**, failing < 1.5× for low-pooling L ≲ 16). Rivals: R-tautology / R-hidden-traffic /
R-baseline-strawman **killed**; R-banking-tax **confirmed** as the binding limiter. Risks **C1 & C4
resolved**, C2/C3 held (see `decisions.md` ledger + `pim/out/conclusion.md`). **Branch `exp/d017-pim`
is NOT merged** — merge-to-main is the owner's/coderAS's call, deferred to after this verdict. **Next:
owner/coderAS decide merge; if the Arm C write-up proceeds, first pin the target workload's
pooling-factor distribution** (the single observation that flips conditional-GO ↔ unconditional-GO or
NO-GO/weak).

## 3. What happens next (ordered)
| # | Task | Owner | Blocking? | Status |
|---|------|-------|-----------|--------|
| 1 | Design + pre-register the cost-model spike | **experimentAS** | — | ✅ done → `spike-prereg.md`, D-012 |
| 2 | Implement the cost model + run it | **coderAS** (Stage A) | after #1 | ✅ rig built + A1 (D-013) + canonical HIGGS run |
| 3 | Analyse → **GO / NO-GO** decision | experimentAS | after #2 | ✅ **NO-GO (D-014)** → `spike/out/conclusion.md` |
| — | ~~*If GO:* EDGE sim build~~ / ~~*NO-GO:* general-purpose EDGE~~ | — | — | ✖ superseded by the D-015 pivot |
| 4 | **Pivot to the A→C arc (mini-GPU → PIM)** | owner + coderAS | after #3 | ✅ decided (D-015) |
| 5 | **Settle v1 (Arm A) scope: ISA + kernel lang + first kernels + stack** (OQ-6,7,9) | **coderAS** (+owner sign-off) | after #4 | ▶ **ACTIVE — next action** |
| 6 | Build Arm A v1: SIMT sim + compiler + kernels; instrument divergence/coalescing/occupancy | coderAS | after #5 | ✅ SIMT spine, divergence, reduction; latency-hiding/coalescing/divergence/memory-wall all measured + a 9-chapter GPU-arch course (`simt/docs/`) |
| 6b | Expose the memory wall (the forcing function for Arm C) | coderAS | after #6 | ✅ measured: one access ≈ 225 arith ops (`cycles=451+K`); scattered gather = 6 instrs / 900 cyc / 64 txns. `simt/docs/09` |
| 7 | Arm C v2: PIM model + ISA on a memory-bound kernel; measure data-movement win (OQ-8) | coderAS (+ discussAS/experimentAS to scope) | after #6b | ✅ **scoped (D-017)** — bank+bandwidth-capped model, ~6-op PIM ISA, headline = embedding-bag (A) + reduction de-risk (B) + reduction-ratio sweep (D); metric = off-chip bytes moved (+energy). New `pim/` module. |
| 7a | **Pre-register the Arm C measurement** (params, sweep, metrics, NO-GO rule) | **experimentAS** | after #7 | ✅ **done → `pim-prereg.md`, D-018.** Primary = **DMR** (baseline÷PIM off-chip bytes, counted); H1 = DMR ≥ 3× canonical + ≥ 1.5× robust + honest crossover; reduction-ratio sweep kills R-tautology; fair coalesced baseline + shared bandwidth cap; **admits an honest NO-GO** (C4) |
| 7b | Build Arm C: bandwidth-capped model → baseline vs PIM bytes-moved on reduction, then embedding-bag; run the sweep | coderAS | after #7a | ✅ **BUILT + RUN** (branch `exp/d017-pim`, commit `e33b365`, 8/8 model tests pass). Bank+bandwidth-capped C++ byte-accounting model; canonical + full sweep; artifacts in `pim/out/` (`decision_inputs.json`, `results.csv`, `provenance.json`, 5 plots) |
| 7c | Render GO/NO-GO verdict from artifacts (apply frozen §5 rule) | **experimentAS** | after #7b | ✅ **CONDITIONAL-GO (D-019), conf ~0.82.** Canonical DMR=2.53× (gray zone); win real but banking-limited (R-banking-tax the limiter), scoped to high-pooling features (L≳64). R-tautology/hidden-traffic/baseline-strawman killed; C1 & C4 resolved. Verdict + rival scorecard → `pim/out/conclusion.md` |
| 7d | Decide merge of `exp/d017-pim` → main; if write-up proceeds, pin target pooling-factor distribution first | owner + coderAS | after #7c | ▶ **NEXT.** Branch not merged (deferred to owner/coderAS per pre-reg). The pooling-factor distribution is the single observation that flips conditional-GO ↔ unconditional-GO / NO-GO-weak |

**Optional Arm A extensions (not blocking Arm C):** tiled matmul (needs thread blocks + shared
memory), a bandwidth-capped memory model, warp-shuffle reduction.

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
See `decisions.md` §"Risk & assumption ledger". **R7 (GBDT win magnitude) is RETIRED → NO-GO (D-014).**
**Arm C risks resolved (D-019):** **C1 (tautology) KILLED** (honest DMR ≪ naive line; real crossover
L=1→1.000, L*(B=16)=32) and **C4 (marginal/no-honest-win) RESOLVED → CONDITIONAL-GO** (win real but
banking-limited, scoped to high-pooling L≳64); C2 (DRAM over-scope) and C3 (energy-assumption) held —
energy DMR is e-invariant. Live concerns are now the general-purpose/PIM write-up positioning and the
target-workload pooling-factor distribution (the observation that firms up the conditional-GO scope).

## 7. Pointers
- Design: `architecture.md`
- Decisions/ADRs: `decisions.md`
- **Spike pre-registration (the D-008 gate design): `spike-prereg.md`** ← build Stage A to this
- **Arm C pre-registration (the D-017 PIM measurement): `pim-prereg.md`** ← build Arm C to this (D-018)
- **Arm C verdict + rival scorecard + plots: `pim/out/conclusion.md`** ← the CONDITIONAL-GO result (D-019); `pim/out/` is gitignored, force-add to commit it
- Spike brief: `handoffs/experimentAS-handoff.md`
- Build brief: `handoffs/coderAS-handoff.md`
- Evidence: `research/landscape.md`, `research/usecase.md`

## 8. Provenance
Chosen via a discussAS strategy engagement + two researchAS citation-backed sweeps (2026-07-05).
Claims are tagged `[verified]`/`[believed]`/`[guess]`; fabricated citations were caught and discarded.
