# Grove — Spike Pre-Registration (D-008 cost-model de-risk)

**Status:** pre-registered / awaiting build. **Owner:** experimentAS (design) → coderAS (Stage A build).
**Anchor decision:** `decisions.md` D-008, D-012. **Resolves:** OQ-1, OQ-2, OQ-4.
**Date pre-registered:** 2026-07-06. **Confidence tags:** `[verified]` primary-source checked ·
`[believed]` reasoned · `[guess]` low grounding.

> This document is **append-only from the moment data is generated**. The hypotheses, primary
> metric, baseline, decision rule, and analysis plan below are committed **before** any number is
> produced, so the conclusion cannot be bent to fit the result. Any analysis not listed here is
> **exploratory** and must be labelled as such in the write-up (a separate `spike-exploratory.md`),
> never reported as a confirmatory test. *(Nosek pre-registration; Kerr anti-HARKing.)*

---

## 0. What this spike is — and what a GO actually licenses (read this first)

Grove's entire AI thesis rests on **one** empirical assumption (D-006): *XGBoost tree-ensemble
inference, in the batch-1 online-serving regime, is **control-bound** with a **small on-chip working
set** — so an EDGE dataflow core can extract far more parallelism per cycle than an equal-resource
scalar CPU, precisely where GPUs stall on warp divergence.* If that assumption is false, months of
simulator work are wasted. This spike retires risk **R7** for **$0 and days** before the one-way door
of building the simulator.

**It is an analytical cost model, not silicon and not a cycle-accurate sim.** Be honest about what it
can and cannot do:

- It **can** *bound* the available parallelism and the working-set footprint of a **real trained
  XGBoost model** under an explicit, stated resource + overhead model. It is a **screen**: a
  necessary condition, computed cheaply.
- It **cannot** *prove* silicon behaviour. Analytical models tend to be **optimistic** — they omit
  second-order effects (contention, wire delay, control corner cases). We compensate by (a) requiring
  a comfortable margin at the model level, and (b) making the GO decision on the **conservative
  corner** of the parameter space (params favourable to the scalar, unfavourable to EDGE).

**Therefore a GO means:** *"the win regime is plausible enough — under a fair, self-skeptical model —
to justify building the cycle-accurate simulator that can actually measure it."* It does **not** mean
"we will win in silicon." Everywhere below, results are tagged **[modelled]**; nothing here is
**[measured]** in the silicon sense. The simulator (Stage B) produces the first `[measured]` numbers.

---

## 1. Hypotheses (falsifiable — Popper / Platt)

Let **ρ** = the *sustained parallel node-evaluations per cycle* of the EDGE-issue model divided by
that of an **equal-resource** scalar RISC-V baseline (defined precisely in §3–§4), on the **same
trained ensemble**, in the **batch-1** regime, **after** charging EDGE's own overheads.

- **H0 (null):** On a real XGBoost ensemble, EDGE holds **no meaningful sustained-parallelism
  advantage** over the equal-resource scalar baseline (**ρ ≤ 1.5×** at the canonical config against
  the *conservative* baseline), **and/or** the resident model footprint does **not** fit a plausible
  on-chip SRAM budget (it spills → bandwidth-bound → EDGE's ILP edge is irrelevant).
- **H1 (alternative):** EDGE's sustained parallelism is **substantially higher** (**ρ ≥ 3×** at the
  canonical config against the conservative baseline, and **ρ ≥ 2×** across the whole sensitivity
  sweep) **and** the resident model fits the on-chip budget with margin **and** modelled overheads
  reduce but do **not** erase the gain.

**Rival hypotheses this design is built to kill (Platt — strong inference).** The experiment must be
able to *exclude* at least one of these, not just confirm H1:

- **R-A "scalar is already fine":** tree branches are *predictable* (skewed splits a branch predictor
  learns), so an in-order superscalar sustains high IPC and ρ is small. → Killed/confirmed by the
  **data-grounded misprediction estimate** (§3) and the branchless-scalar variant.
- **R-B "EDGE overhead eats the win":** tag-match + dynamic-dispatch + operand-network +
  load-imbalance overheads dominate, so EDGE's *effective* throughput collapses. → Killed/confirmed by
  the **explicit overhead breakdown** (§3, §5) — we refuse to assume these away.
- **R-C "it spills":** the ensemble doesn't stay on-chip at realistic sizes → the workload is
  bandwidth-bound, not control-bound. → Killed/confirmed by the **working-set guardrail + #trees
  sweep** (§3, §6).

**The refuting observation (what would prove H1 wrong):** any one of — ρ < 1.5× at canonical config
against the conservative scalar; or the resident footprint exceeds the on-chip budget at the canonical
config; or overheads drive the *effective* ρ below 1.5×.

---

## 2. Primary metric, guardrails, and baseline (Kohavi — OEC + guardrails up front)

### 2.1 Primary metric (the OEC)
**ρ = sustained parallel node-evaluations per cycle (EDGE) ÷ sustained parallel node-evaluations per
cycle (scalar)**, at equal functional-unit budget *N*, equal clock, equal on-chip feature memory,
in the batch-1 regime, with EDGE overheads charged. A "node-evaluation" = one tree-node visit =
{indexed feature-gather + threshold compare + route/dispatch to child (or emit leaf)}.

We report ρ as an **effect size with a sensitivity range**, never a single hero number: the canonical
point estimate **plus** the min–max band over the pre-committed sweep, **plus** the value at the
conservative corner (which drives the decision).

*Why this metric, not raw "speedup":* it isolates the architectural claim — *can the machine keep N
units busy on this workload?* — from the trivially-riggable choice of *how many units N you were
given*. It is the direct, honest test of the ILP-extraction thesis (architecture.md §1).

### 2.2 Guardrail metrics (satisficing — must pass, but not maximised)
- **G1 — Resident model footprint ≤ on-chip SRAM budget.** The whole ensemble must stay resident on
  chip so it is *reused* across the inference stream rather than re-streamed from DRAM. Primary budget
  **1 MB**; reported against {256 KB, 512 KB, 1 MB, 2 MB} with the spill point identified. This is the
  literal test of "small on-chip working set" and of being **off the bandwidth roof**.
- **G2 — Per-inference *touched* working set ≤ 256 KB.** Nodes actually visited along this row's paths
  + feature vector + partial leaf scores. (Honors the draft's 256 KB budget; expected to pass easily.)
- **G3 — Overhead does not erase the gain.** With all EDGE overheads charged, modelled EDGE overheads
  consume **< 50%** of ideal (overhead-free) EDGE throughput. If overheads eat ≥ 50%, that is a red
  flag even if ρ nominally clears 3× — investigate before GO.

### 2.3 Baseline definition (OQ-4 — resolved; see §4 for full normalisation)
**Equal-resource scalar RISC-V = an *N*-wide *in-order* superscalar** with the **same** number of
compare/ALU functional units *N*, the **same** clock, and the **same** on-chip feature SRAM as the
EDGE model. Two variants are modelled and both reported; the **decision uses the stronger (more
conservative) of the two**:
1. **Branchy scalar** — data-dependent branch per node, with a modelled misprediction penalty.
2. **Branchless/predicated scalar** — QuickScorer-style, no mispredicts, but evaluates more of each
   tree (extra work). This is the *strong* CPU baseline the literature actually uses. `[verified: QuickScorer/RapidScorer exist precisely to remove branch cost — research/usecase.md §4]`

---

## 3. The analytical cost model (precise — this is what coderAS implements one-to-one)

Both engines are scored on the **same** parsed ensemble and the **same** sample of real input rows, so
the only differences are architectural. All parameters have a pre-registered **default** *and* a
**sweep range**, and a **conservative-corner** value (the value least favourable to H1).

### 3.1 Inputs parsed from the XGBoost model
From `booster.get_dump(with_stats=True, dump_format='json')` (or `booster.trees_to_dataframe()`):
per tree, every node's {split feature index, threshold, yes/no/missing child, leaf value}, and each
node's depth. Derived per-tree: node count, max depth. **Realized path lengths are data-dependent**,
so they are obtained by running *real inference on a sample of real test rows* (default **1,000 rows**,
fixed seed) and recording, per row per tree, the actual root→leaf path. We report the **distribution**
of path lengths, not just the max — load imbalance lives in that distribution.

### 3.2 Work and critical path (engine-independent)
For a given input row:
- **Total work** `W = Σ_trees (realized path length of this row in that tree)` node-evals.
- **Ideal latency (critical path)** `L* = max_trees(path length) + ceil(log2(T))` — trees are
  independent (run concurrently); the trailing `log2(T)` is the leaf-score **reduction tree**
  (explicitly modelled, not free).
- Report W and L* averaged over the 1,000 sampled rows (batch-1 = per-row, averaged).

### 3.3 Scalar engine — sustained node-evals/cycle
Bottleneck (Little's-law style) model: `scalar_rate = min(issue_ceiling, control_ceiling)`.
- **issue_ceiling = N** (an *N*-wide core retires ≤ N node-evals/cycle).
- **Branchy variant control_ceiling = N / (1 + p_mis · B)** where `B` = pipeline flush penalty
  (cycles) and `p_mis` = per-node misprediction probability. An in-order pipeline flushes on a
  mispredict, so mispredicts throttle the whole front-end.
  - **`p_mis` is data-grounded, not assumed.** For each internal node, compute the empirical
    left-branch frequency `f` over the 1,000 sampled rows. Under a bias/2-bit-saturating predictor the
    achievable mispredict rate ≈ `min(f, 1−f)` (a near-50/50 split is unpredictable; a skewed split is
    learned). `p_mis` = the path-visit-weighted mean of `min(f,1−f)` across nodes. **Also swept** over
    a fixed range as robustness (see §6).
  - `B` default **12** cycles `[believed: typical modern in-order/short-OoO flush]`; swept {6, 12, 20}.
- **Branchless variant:** `control_ceiling = N` (no mispredicts) **but** work inflates to
  `W_bl = Σ_trees (nodes evaluated under branchless scheme)`. Default model: branchless evaluates the
  full set of split nodes per tree (upper bound ≈ `2^depth − 1`), i.e. `W_bl ≫ W`. Report both the
  optimistic (path-only + predication masking, ×~2) and pessimistic (full-tree) branchless costs.
- **Scalar cycles** `= W / scalar_rate` (branchy) or `W_bl / N` (branchless). Report node-evals/cycle
  `= W / scalar_cycles` for a like-for-like ρ numerator/denominator.

### 3.4 EDGE engine — sustained node-evals/cycle (list-scheduling + explicit overheads)
Model the ensemble as **T independent serial chains** (one per tree; chain length = that row's realized
path length in that tree) plus a trailing `log2(T)` reduction DAG. **Greedily list-schedule** these
chains onto **N** dataflow units, each node-eval costing `c_edge` cycles of unit occupancy. The
resulting **makespan M_edge** yields `edge_rate = W / M_edge` and **utilisation = W / (N · M_edge)**.
This scheduling *directly captures load imbalance* — when short trees finish, fewer than N independent
tokens remain in flight and units idle at the tail.

**EDGE overheads — modelled explicitly, never assumed away (kills rival R-B):**
| Overhead | Symbol | Where it enters | Default | Sweep | Conservative corner |
|---|---|---|---|---|---|
| Token tag-match | `t_tag` | per node-eval occupancy adder | 1 cyc | {0.5, 1, 2} | 2 |
| Dynamic dispatch (route to child) | `d_disp` | per node-eval occupancy adder | 1 cyc | {0.5, 1, 2} | 2 |
| Feature-gather (on-chip indexed load) | `g` | per node-eval occupancy adder, pipelined | 1 cyc | {1, 2, 4} | 4 |
| Operand-network bandwidth | `R` | global cap: `edge_rate ≤ R` operands/cyc | N | {N/2, N, 2N} | N/2 |
| Load imbalance | — | *emergent* from the list-schedule of unequal chains | — | — | — |

So `c_edge = 1 + t_tag + d_disp + g` (occupancy per node-eval), the schedule is capped at `R`
operands/cycle, and load imbalance falls out of the makespan. **Ideal (overhead-free) EDGE rate** is
also computed (`c_edge=1`, `R=∞`) so G3 (overhead fraction) can be reported.

### 3.5 Working-set footprint (guardrail G1/G2)
- **Node size** (compact encoding): {feature_idx 2 B, threshold 4 B, left 2 B, right 2 B / leaf 4 B} =
  **16 B/node** (default; also report 12 B optimistic and 24 B pessimistic). `[believed]`
- **Resident footprint (G1)** `= (Σ_trees node_count) · bytes_per_node + feature-vector + small
  buffers`. Compared to the SRAM budget grid.
- **Touched footprint (G2)** `= (Σ_trees realized_path_len) · bytes_per_node + F·4 B (features) +
  T·4 B (partial scores)`, averaged over sampled rows.

### 3.6 Batch regime
- **Batch-1 is primary and decision-driving** (matches D-006 online-serving target).
- For batch ∈ {8, 64, 256}: EDGE gains more independent tokens (better utilisation), **but the scalar
  baseline is credited with SIMD amortisation** across rows (vectorise node-evals over the batch, up to
  N lanes) — this *shrinks* EDGE's edge and is modelled so we do **not** over-claim. Report ρ vs batch
  and the crossover where the advantage fades. This is honest scoping, not a result to maximise.

---

## 4. OQ-4 resolved — the fair, non-rigged baseline normalisation

**The failure mode we are guarding against:** giving EDGE *N* units and the scalar 1 unit, so a "3×"
is really just "N-wide vs 1-wide." That would be rigged.

**Normalisation committed (equal-resource):**
1. **Equal functional units.** Both engines get the **same N** compare/ALU-capable units. Primary
   `N = 16`; swept {4, 8, 16, 32}. ρ isolates *architecture's ability to fill N units*, not N itself.
2. **Equal clock.** Same cycle time; we compare **node-evals/cycle**, so absolute frequency cancels.
3. **Equal on-chip feature memory.** Both read features from the same on-chip SRAM at the same latency
   `g`. EDGE gets **no** free memory advantage.
4. **Scalar is a real, optimised baseline, not a strawman.** The decision uses the **stronger** of
   {branchy-with-predictor, branchless/QuickScorer-style}. `p_mis` is **estimated from the data**, and
   swept — we do not hand the scalar an artificially high mispredict rate.

**Stated asymmetry & its justification (honest):** we compare EDGE to an **in-order** superscalar, not
an idealised **infinite-window out-of-order** core. This is deliberate and fair to the thesis, because
EDGE's entire pitch is *achieving ILP without the OoO renaming/scheduler machinery* (architecture.md
§1) — comparing to a hypothetical zero-cost OoO core would be comparing to a different, more expensive
design point. **Threat noted (§7):** an aggressive OoO core could also mine cross-tree ILP; we flag
this as a stronger-baseline caveat and a Stage-B follow-up, and we do **not** claim victory over OoO
from this spike. We additionally report ρ against an *idealised in-order with perfect prediction*
(`p_mis=0`) as an upper-bound sanity check on how much of EDGE's edge is "just" branch handling.

---

## 5. OQ-1 resolved — pre-committed GO / NO-GO decision rule

Decided **before** any data. The **canonical config** (§6) is the primary decision point; the sweep is
for **robustness/description**, not for hunting a favourable point (anti-p-hacking / anti-HARKing). All
thresholds are on ρ computed against the **conservative baseline** at the **conservative overhead
corner**.

### GO — build the simulator (Stage B). **All** must hold:
1. **Primary:** ρ ≥ **3×** at the canonical config (Higgs, 500 trees, depth 6, batch-1, N=16), against
   the stronger scalar baseline; **and** ρ ≥ **2×** across the *entire* core sensitivity sweep (not a
   knife-edge).
2. **Guardrail G1:** resident footprint ≤ on-chip budget (1 MB primary) at the canonical config — off
   the bandwidth roof.
3. **Guardrail G3:** with all overheads charged, effective ρ still clears the bar **and** overheads
   consume < 50% of ideal EDGE throughput.

### NO-GO — kill the ML framing, fall back to general-purpose EDGE (D-008). **Any** triggers:
1. ρ < **1.5×** at the canonical config against the conservative baseline; **or**
2. Resident footprint **spills** the on-chip budget at the canonical config (bandwidth-bound → EDGE
   irrelevant); **or**
3. Modelled overheads drive **effective ρ < 1.5×**.

### GRAY ZONE — ρ ∈ [1.5×, 3×): **do not greenlight blindly.**
Report honestly and diagnose the **binding limiter** from the overhead breakdown + a targeted probe
(is it scalar predictability R-A? EDGE overhead R-B? load imbalance?), then make a judgement call with
**explicitly stated calibrated confidence**. Legitimate outcomes: (a) *conditional-GO* with a scoped
Phase-1 sim to verify the specific uncertain mechanism before full commitment, or (b) *NO-GO*. The
gray-zone outcome must name which rival hypothesis remained un-killed.

### Threshold justification `[believed]`
- **3× GO:** analytical models are optimistic; a 3× model result plausibly survives to **~1.5–2×** in a
  careful cycle-accurate sim — still a genuine, portfolio-worthy result that clears R7's "only meh"
  bar. The extra `ρ ≥ 2× across the sweep` guard prevents greenlighting a config-fragile win.
- **1.5× NO-GO:** below ~1.5× at the *optimistic* model level, silicon would likely land ≤ 1× — not
  worth months, and the general-purpose EDGE fallback is the better use of effort.
- **Guardrails are physically motivated:** G1 is the literal bandwidth-roof test the whole thesis
  rests on (research/usecase.md §0, §5); G3 stops a "win" that is really an artifact of ignoring
  EDGE's own costs.

---

## 6. OQ-2 resolved — canonical dataset + XGBoost config, and the sensitivity sweep

### Canonical benchmark (decision-driving)
**Dataset: HIGGS** (UCI / OpenML, 11 M rows, **28 dense numeric features**, binary classification).
`[verified: standard XGBoost/LightGBM benchmark dataset, fully public, no auth]`
- **Why Higgs:** (a) it is the *de-facto* standard GBDT benchmarking dataset (used in the XGBoost and
  LightGBM papers), so results are comparable and reproducible with **no Kaggle auth**; (b) **binary
  classification** is the exact shape of the target online-serving use cases — fraud scoring, ad-click,
  real-time bidding (D-006) — each a single batch-1 binary score; (c) 28 dense numeric features give a
  clean, unambiguous feature-gather model.
- **Canonical XGBoost config:** `n_estimators=500, max_depth=6, tree_method="hist",
  objective="binary:logistic"`, default `learning_rate`, `random_state=42`. Train on a fixed
  subsample (e.g. 1 M rows) for speed; **evaluate the cost model on 1,000 held-out test rows** (fixed
  seed). Log xgboost version + data hash + full param dict.
- **Why 500 / depth-6:** matches real production GBDTs (fraud/ranking models are typically hundreds of
  trees, depth 4–8); depth-6 ⇒ ≤ 63 nodes/tree — realistic, not a toy. `[believed]`

### Secondary / robustness dataset (guards against cherry-picking)
**Dataset: Covertype** (UCI, 581 K rows, **54 features, 7-class**). `[verified: standard UCI dataset]`
- **Why also Covertype:** its **multiclass** nature makes XGBoost train `T × 7` trees, which **stresses
  the working-set guardrail G1** (larger resident footprint) and tests a **different feature count**
  (54 vs 28). If ρ and the guardrails hold on *both*, the result is not dataset-specific.

### Pre-committed sensitivity sweep (report ρ + guardrails at every point; decision on canonical)
| Axis | Values | What it probes |
|---|---|---|
| **#trees** `T` | {100, 300, **500**, 1000} | Working-set spill point (G1); ρ stability |
| **Depth** `max_depth` | {4, **6**, 8, 10} | Path length ↑ ⇒ load imbalance, critical path, footprint |
| **Batch** | {**1**, 8, 64, 256} | Where EDGE's batch-1 edge fades vs SIMD scalar |
| **N units** | {4, 8, **16**, 32} | Is ρ an artifact of a single N? |
| **`p_mis`** | data-grounded + {0.1, 0.3, 0.5} | Rival R-A: are trees predictable? |
| **EDGE overheads** | defaults + conservative corner (§3.4) | Rival R-B: do overheads erase the win? |
| **bytes/node** | {12, **16**, 24} | Footprint sensitivity for G1/G2 |

Bold = canonical value. **The decision is made at the canonical point against the conservative corner;
the sweep characterises robustness and is reported as the sensitivity band. No threshold moves after
data is seen.**

---

## 7. Threats to validity & how each is handled (Shadish/Cook/Campbell; Feynman)

| # | Threat | Handling |
|---|---|---|
| T1 | **Modelled ≠ measured** — analytical model is optimistic | Everything tagged `[modelled]`; GO requires 3× margin + conservative-corner decision; GO only *licenses* the sim, doesn't claim silicon victory |
| T2 | **Scalar strawman** | Decision uses the *stronger* of branchy/branchless; conservative params favour the scalar; report vs perfect-prediction upper bound |
| T3 | **`p_mis` assumed** (rival R-A) | Data-grounded per-node empirical entropy estimate **and** a sweep |
| T4 | **EDGE overheads assumed away** (rival R-B) | 5 overheads modelled explicitly (§3.4); G3 caps overhead fraction; conservative corner drives the decision |
| T5 | **Working-set ambiguity** | Two footprints defined precisely (resident G1 vs touched G2); budget grid + spill point reported |
| T6 | **Cherry-picked ensemble** | Two datasets + full sweep; ρ must hold across the sweep, not just at one point |
| T7 | **Batch regime over-claim** | Batch-1 primary; scalar credited with SIMD at batch>1; crossover reported |
| T8 | **Memory hierarchy simplified** | G1 spill = proxy for bandwidth-bound; if spill, gather ≠ on-chip → NO-GO by rule |
| T9 | **Reduction cost ignored** | `log2(T)` reduction DAG modelled in L* and the EDGE schedule |
| T10 | **Stronger OoO baseline** | Explicitly out of scope for the spike; flagged as Stage-B follow-up; not claimed |
| T11 | **Multiplicity / forking paths** from the sweep | Decision pre-committed to the canonical config; sweep is descriptive; extra analyses labelled exploratory |
| T12 | **Non-reproducibility** | Fixed seeds, pinned versions, data hash, one-command run, `provenance.json` |

---

## 8. Reproducibility & provenance (an unreproducible result is not evidence)

- **One command:** `python spike/run.py --config spike/config.yaml` (train-or-load → dump → cost-model
  → sweeps → plots → results → GO/NO-GO summary). Dataset cached on first run.
- **Provenance logged** to `spike/out/provenance.json`: xgboost/numpy/python versions, `random_state`,
  dataset name + SHA-256 hash, full hyperparameter dict, full cost-model parameter dict, timestamp,
  and (if the repo is under git) commit hash.
- **Pre-registration integrity:** this file is frozen at data-generation time. Post-hoc analyses go in
  `spike-exploratory.md`, clearly labelled exploratory.
- **Storage:** experiment code + outputs under `C:\Code\Self\projects\grove\spike\`; a pointer entry is
  recorded in the experimentAS index. **coderAS should build on a dedicated local branch
  `exp/d008-cost-model-spike`** (do not pollute the working line; merge decision deferred to CONCLUDE).

---

## 9. Deliverables coderAS must return (so experimentAS can render GO/NO-GO)

1. `spike/out/results.csv` (or `.json`) — for every sweep point: ρ (vs branchy & branchless scalar),
   EDGE utilisation, ideal-EDGE rate, scalar rate(s), W, L*, `p_mis` (data-grounded), resident & touched
   footprints, per-overhead cycle contribution.
2. **Plots:** ρ vs #trees; ρ vs depth; ρ vs batch; ρ vs N; EDGE utilisation bar; resident footprint vs
   #trees with the SRAM-budget lines; **stacked overhead breakdown** (ideal vs charged EDGE throughput).
3. A machine-readable `spike/out/decision_inputs.json` reporting the three GO-gate quantities at the
   canonical config + conservative corner (primary ρ, ρ-sweep-min, G1 margin, G3 overhead fraction).
4. `provenance.json` (§8).

**Definition of done:** experimentAS can read those artifacts and apply §5's rule mechanically to
produce a **GO / NO-GO** with a calibrated-confidence statement, then append the outcome ADR.

---

## 10. Approval gate

This is the pre-registration. **coderAS Stage A is unblocked to implement to this spec.** No thresholds
or metric definitions may be changed after data is generated; if a definition proves unimplementable,
coderAS returns to experimentAS to amend the pre-registration **before** producing numbers, and the
amendment is recorded with a timestamp.
