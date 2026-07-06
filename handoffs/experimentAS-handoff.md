# Handoff → experimentAS: the Lithos de-risk spike

**You own the design.** This brief gives you the mission, constraints, and a *draft* decision rule.
Refine it, pre-register it, get it approved, then delegate the implementation to coderAS
(`coderAS-handoff.md`, Stage A) and analyse the result. Do **not** let Lithos's simulator get built
until this spike returns **GO**.

Full context: `../handoff.md`, `../architecture.md`, `../decisions.md` (esp. **D-006, D-008**),
`../research/usecase.md`.

---

## 1. Why this spike exists (the one-way door it guards)
Lithos's entire AI thesis rests on one empirical assumption: **decision-tree ensemble (XGBoost)
inference is control-bound with a small on-chip working set**, so an EDGE dataflow core can extract
far more parallelism per cycle than a scalar CPU and win where GPUs stall on divergence. If that's
false (e.g. the workload spills to memory, or dispatch/gather overhead dominates), we'd waste months
building a simulator around a dead claim. This spike is **$0 and days**, versus a multi-month build —
textbook "retire the biggest risk cheapest first."

## 2. Hypotheses (make them falsifiable — your call to finalise)
- **H0 (null):** On a real XGBoost ensemble, an EDGE-issue model sustains **no meaningful advantage**
  in achievable parallel-ops/cycle (functional-unit utilisation) over an equal-resource scalar
  RISC-V model, and/or the per-inference working set does **not** fit a plausible on-chip budget.
- **H1:** EDGE-issue utilisation is **substantially higher** than the scalar baseline **and** the
  working set fits on-chip → the win regime is real.

## 3. Method (a pure cost model — no RTL, no real hardware)
Build a Python **analytical cost model** over a **real, trained XGBoost model** (not a toy):
1. Train (or load) a realistic ensemble on a standard tabular dataset — e.g. Higgs, Covertype, or a
   Kaggle tabular set. Suggested config to start: ~500 trees, depth ~6–8. (You pick the canonical one — OQ-2.)
2. Extract the ensemble structure (nodes, thresholds, feature indices, leaves) from the model dump.
3. Model **two execution engines** analytically over batch-1 (and small-batch) inference:
   - **Scalar RISC-V baseline:** serial node visits; one compare/branch per cycle-ish; count cycles.
   - **EDGE + dynamic-dataflow model:** many independent tree traversals tagged and interleaved across
     N functional units; count achievable parallel node-evaluations/cycle given the operand-network /
     dispatch model, and the sustained FU utilisation.
4. Measure the **working-set footprint** per inference (active nodes + feature vector + partial
   scores) and compare to a plausible on-chip SRAM budget.
5. **Explicitly model the overheads that could kill the thesis:** feature-gather cost, dynamic
   dispatch cost, tag-matching cost, load imbalance across trees. Do **not** assume them away.

## 4. Draft decision rule (pre-register your final version)
- **GO** if EDGE-model sustained FU utilisation (or achievable parallel node-evals/cycle) is **≳ 3×**
  the scalar baseline on the same ensemble **AND** the per-inference working set fits on-chip
  (draft budget: ≤ ~256 KB) **AND** modelled dispatch/gather overhead does not erase the gain.
- **NO-GO / weak** if the advantage is **< ~1.5×** or the set spills → fall back to a general-purpose
  EDGE contribution without the ML framing (D-008).
- **Gray zone (1.5×–3×)** → report it honestly and investigate whether dispatch/gather overhead or
  load imbalance is the limiter before deciding.
*(These numbers are a starting point — set the final thresholds and justify them in the pre-registration.)*

## 5. Guardrails / threats to validity (your specialty)
- **Fair baseline (OQ-4):** normalise resources — the scalar and EDGE models must assume comparable
  compute budgets, or the comparison is rigged. State the normalisation explicitly.
- **Don't cherry-pick the ensemble** — use a realistic size/depth, report sensitivity to #trees & depth.
- **Batch regime matters** — batch-1 (online serving) is the target; note how the advantage changes with batch size.
- **Separate "modelled" from "measured"** — this is an analytical cost model; be clear it bounds, not proves, real silicon behaviour. State the assumptions that most affect the result.
- **Report effect size + a sensitivity range, not a single hero number.**

## 6. Deliverables
1. A short **pre-registration** (hypotheses, metric, baseline, decision rule, assumptions) — approved before running.
2. The cost-model spec handed to **coderAS** (Stage A).
3. An **analysis** with the utilisation ratio, working-set footprint, overhead breakdown, sensitivity
   sweep, and a clear **GO / NO-GO** call with calibrated confidence.
4. Append the outcome as a new ADR in `../decisions.md` and update the status table in `../handoff.md`.

## 7. Definition of done
A reproducible ($0) result that either **greenlights** the Lithos build with an evidence-backed
expected win, or **kills the ML framing** cheaply — with the reasoning recorded so it isn't re-litigated.
