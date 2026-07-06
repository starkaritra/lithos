# Lithos D-008 spike — CONCLUSION: **NO-GO** (drop the ML/tree-inference framing)

- **Experiment:** D-008 cost-model de-risk — "is XGBoost batch-1 inference control-bound & on-chip enough for an EDGE core to beat an equal-resource scalar by a portfolio-worthy margin?"
- **Pre-registration:** `spike-prereg.md` §0–§10 + **Amendment A1** (D-013, pre-data engine-fairness fix)
- **Anchor ADRs:** D-008 (gate), D-012 (design), D-013 (A1 amendment), **D-014 (this outcome)**
- **Backend / model:** analytical cost model (`spike/costmodel.py`), all numbers **[modelled]**, not [measured]
- **Dataset / config:** HIGGS (28 feat, binary), 500 trees, depth 6, `hist`, batch-1, N=16, seed 42
- **Baseline:** equal-resource N-wide in-order superscalar; decision baseline = **stronger** of {branchy-swpipe, branchless-opt}
- **Decision corner:** conservative overhead corner (τ_tag=1.0, R_factor=0.5) — the EDGE-unfavourable values
- **Provenance:** xgboost 3.2.0, numpy 2.4.6, py 3.12.10; data SHA-256 `79f2444f…`; git `8f9efbc` (+ impl `355dc7a`); `spike/out/provenance.json`
- **Date concluded:** 2026-07-06T14:16+05:30 · **Owner:** experimentAS

---

## 1. Verdict: **NO-GO**, calibrated confidence **~0.9 (high)**

Per the pre-registered §5 rule, the primary metric at the canonical config against the conservative
baseline is **ρ = 0.84**, which is **below the NO-GO threshold (ρ < 1.5×)**. This is not a gray-zone
call — it is a mechanical NO-GO, and it is **overdetermined** by four independent rails:

| Rail | Pre-reg rule | Result | Verdict |
|---|---|---|---|
| **Primary ρ (canonical, conservative)** | GO ≥ 3× · NO-GO < 1.5× | **0.84** | NO-GO |
| **Sweep-min ρ** | GO needs ≥ 2× everywhere | **0.74** (and sweep-**max** only **1.41**) | fails GO |
| **G3 overhead fraction** | must be < 0.50 | **0.561** at conservative | fails GO |
| **G1 resident footprint** | ≤ 1 MB on-chip | 898 KB at canonical (fits), but **spills** at depth≥8 or trees=1000 | fails off-canonical |

**The decisive fact:** even the **overhead-free EDGE ceiling** is `edge_ideal_rate / strong_scalar =
15.35 / 8.0 = 1.92×` — *below the 2× sweep floor before a single cycle of EDGE overhead is charged.*
So the shortfall is **structural to the workload-plus-baseline**, not an artifact of the A1 overhead
model. EDGE does fill its units well (utilisation ≈ 0.96 of N) — the parallelism thesis holds — but
the branchless/QuickScorer-style N-wide scalar **harvests the same cross-tree parallelism**, so EDGE
has no unique width advantage to convert into a margin.

## 2. Rival hypotheses (Platt strong inference — what the data excluded)

- **R-A "the scalar is already fine" → CONFIRMED (this is the decisive finding).** HIGGS branches are
  predictable (data-grounded `p_mis = 0.146`), so the binding baseline is the **branchless** N-wide
  scalar (branch-free QuickScorer-style), which captures the same independent-tree parallelism EDGE
  offers. Strong internal-validity signal: **ρ is invariant (1.20) across the whole `p_mis` sweep** —
  the decision does not depend on any branch-predictability assumption, because the binding baseline is
  branchless and p_mis-immune. H1 is refuted by a confirmed rival, not by a fragile parameter choice.
- **R-B "overhead eats the win" → partially confirmed, not the sole cause.** At the conservative corner
  G3 = 0.56 (> 0.50) and τ_tag=1.0 drives ρ to 0.92; but even the overhead-*light* end (τ_tag=0.25)
  reaches only ρ = 1.41, and the overhead-*free* ceiling is 1.92×. Overheads deepen the loss; they do
  not create it.
- **R-C "it spills" → confirmed for realistic larger models.** Canonical (898 KB) fits 1 MB, but
  depth-8/10 and 1000-tree ensembles spill — those regimes are bandwidth-bound, where EDGE's ILP edge
  is moot anyway.

**H1 (ρ ≥ 3× canonical & ≥ 2× sweep, footprint fits, overheads don't erase the gain) is refuted.**

## 3. What a GO would have licensed vs. what NO-GO means

NO-GO does **not** claim "EDGE loses in silicon on all tree workloads." It means: *under a fair,
self-skeptical analytical screen, the portfolio-worthy ≥3× margin the ML thesis needed is not present
for GBDT batch-1 inference against a competent branchless scalar* — so it is not worth the one-way door
of building the simulator **for the ML framing**. Because analytical models are optimistic (§0/§7-T1)
and A1 fixed the model toward fairness, the true silicon ρ is, if anything, ≤ the modelled ρ; a
modelled overhead-free ceiling under 2× makes a real ≥3× win implausible.

## 4. Residual uncertainty (why 0.9, not ~1.0)

- Everything is **[modelled]**, not [measured]; no cycle-accurate confirmation.
- A materially different EDGE mapping (e.g. a dispatch/tag mechanism far cheaper than modelled, or
  fusing multiple node-evals per token) or a *different* control-bound workload (deeper/branchier trees
  with a genuinely small footprint, or irregular non-GBDT control flow) could change the calculus — but
  that is a **different experiment**, not this one. The overhead-free-ceiling argument makes a flip on
  *this* target (GBDT, batch-1, vs branchless N-wide scalar) unlikely.
- Covertype (robustness set) was not needed to decide (pre-reg makes HIGGS the decision dataset) and
  would, if anything, worsen the footprint/spill picture. See §5.

## 5. Decision & next step

**Trigger the D-008 NO-GO branch:** drop the ML/tree-inference framing and pursue the **general-purpose
EDGE contribution** — the open, minimal, readable, reproducible full-stack EDGE reproduction +
micro-innovations (D-004/D-005/D-009), **without** the GBDT/AI win claim. Route the re-frame to
**discussAS** (handoff task 4'). The EDGE core itself is unaffected; only the *use-case narrative* is
retired.

**Optional confirmatory probe (not required to decide, would only document external validity):**
re-run the sweep on **Covertype** (multiclass, 7× trees, larger footprint) to show the NO-GO is not
HIGGS-specific. Expectation: equal-or-worse ρ and more spill. Flag to coderAS if we want it on record.

## 6. Honesty rails honored

Metric/threshold/baseline/analysis-plan were frozen pre-data (§5) and A1 was a **pre-data** fairness
fix with **no threshold moved** (D-013). This verdict applies the frozen rule mechanically, then adds
calibrated confidence. No post-hoc analysis was used to reach the decision; any further slicing lives
in `spike-exploratory.md`, clearly labelled. A negative result, cheaply obtained ($0, days), that
retires risk **R7** before the simulator one-way door — exactly what the gate was for.
