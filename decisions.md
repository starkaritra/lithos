# Grove — Decision Log (ADRs)

Each entry is a decision, its rationale, and its status. Confidence tags:
`[verified]` = checked against a primary source · `[believed]` = reasoned inference · `[guess]` = low grounding.
Decisions are a **mutable snapshot** — challenge and update them; don't defend them.

---

**D-001 — Goal & constraints.** The project optimizes for **genuine novelty + portfolio/research
signal**, not beginner learning (the builder already knows CS fundamentals). Hard constraints: **$0,
simulation-first, solo, months-scale.** `[verified: builder statement]`

**D-002 — Compute-not-graphics scope cut** *(from the initial GPU exploration; now historical).* Any
accelerator here targets *compute*, never a graphics pipeline (rasteriser/textures). Recorded because
the reasoning still applies to scope discipline. `[known]`

**D-003 — The original GPU/CUDA/tensor dream is NOT a novelty path.** As of Dec 2025 the open
full-stack exists: **Vortex** (RISC-V SIMT GPGPU + LLVM + OpenCL/CUDA), **Ten-Four** (open
tensor/MMA unit), **Gemmini** (open systolic matmul). Building it would be *learning*, not novelty —
competing with funded labs on their turf. **Pivoted away.** `[verified: repos + arXiv 2002.12151, 2512.00053]`

**D-004 — Architecture chosen: EDGE, built from scratch.** An **EDGE** (Explicit Data Graph
Execution) **block-atomic dataflow** core — its own ISA, a hand-rolled compiler, and a cycle-accurate
simulator. Dataflow *inside* a block, control flow *between* blocks; instructions name their
consumers ("target form") and fire when operands arrive, so there is no shared-register-file traffic
inside a block. `[known]`

**D-005 — Novelty framing (be honest).** EDGE is **prior art** (TRIPS, UT Austin ~2003–2009;
Microsoft E2 ~2013). We are **not** inventing it. The contribution is *the first open, minimal,
readable, reproducible **full-stack** EDGE implementation with a **measured** result*, plus our own
micro-innovations (block-formation heuristic, operand-network / dispatch design). Frame it as
reproduction + extension + measurement, never as invention. `[verified: TRIPS/E2 published]`

**D-006 — AI angle (use-case-grounded): tree-ensemble INFERENCE.** The only ML workload that is
genuinely **control-bound with a small on-chip working set** — where EDGE can win — is
**gradient-boosted decision-tree ensemble inference** (XGBoost / LightGBM), especially **batch-1
online serving** (fraud, ad/search ranking, real-time bidding). GPUs lose here to *warp divergence*;
that is why Tahoe / QuickScorer / RapidScorer exist. **Explicitly NOT deep-learning / LLM training** —
that is memory-bandwidth / GEMM / interconnect-bound and already owned. `[verified: DLRM 1906.00091, HyGCN 2001.02514, MegaBlocks 2211.15841, Tahoe EuroSys'21]`

**D-007 — Extension keeps the EDGE core (not a pivot).** Base EDGE is *static* block-atomic
dataflow. Add three small *dynamic* mechanisms: (1) **data-dependent next-block dispatch** (route by
a comparison result), (2) **Monsoon-style token tagging** so thousands of independent tree
traversals run concurrently on one fabric, (3) a **lightweight on-chip indexed feature-load**.
Prior-art spine: **tagged-token dataflow** (Arvind/Monsoon, ISCA 1990), **WaveScalar** (MICRO 2003).
`[verified]`

**D-008 — GATE before building (the one-way-door guard).** Do **not** build the simulator until a
$0, days-long **Python cost-model spike** over a *real trained XGBoost model* confirms the win. See
`handoffs/experimentAS-handoff.md`. **GO** if EDGE-issue functional-unit utilisation ≫ scalar
baseline **and** the per-inference working set fits on-chip; **NO-GO** → fall back to a
general-purpose EDGE contribution without ML framing. `[believed]`

**D-009 — Build vehicle.** Cycle-accurate **software** simulator first (fast to instrument for the
co-design experiments that are the whole point). **Hand-rolled compiler**, *no LLVM* for v1.
Synthesizable Verilog (Verilator) is a **Phase-4 stretch**, not v1. The **compiler + measurement rig
is the real differentiator, not the RTL.** `[believed]`

**D-010 — Reviewer critique to pre-empt.** A sharp reviewer will ask *"why not a fixed-function FPGA
tree pipeline that beats your programmable core?"* Our only honest defense: **programmability /
generality** (one fabric for trees *and* other control-heavy code) + the **open full-stack**
contribution. If we can't defend that, novelty shrinks — keep this in view. `[believed]`

**D-011 — Parked next project.** After Grove: an **"AI-architecture-bottleneck-focused"** build
targeting the *real* deep-learning bottlenecks — memory bandwidth/capacity, irregular-memory latency,
MoE all-to-all interconnect (e.g. near-memory / PIM or interconnect architecture; LLM KV-cache,
recsys embeddings). Complements Grove, which by design cannot touch these. `[verified: builder request]`

**D-012 — Spike design pre-registered (resolves OQ-1, OQ-2, OQ-4).** The D-008 gate is now designed
and **pre-registered** in `spike-prereg.md` (append-only once data is generated). Summary of the
committed design:
- **Falsifiable claim.** Primary metric **ρ = sustained parallel node-evals/cycle (EDGE) ÷ (scalar)**,
  at equal functional-unit budget *N*, equal clock, equal on-chip feature memory, batch-1, **with EDGE
  overheads charged**. H1: ρ ≥ 3× at the canonical config against the *conservative* baseline and
  ρ ≥ 2× across the sweep **and** the resident model fits on-chip. H0: ρ ≤ 1.5× and/or it spills.
  Three rival hypotheses are pre-committed to be killed: scalar-is-predictable, overhead-eats-the-win,
  and it-spills.
- **OQ-2 (dataset/config) resolved:** canonical = **HIGGS** (28 features, binary) at
  `n_estimators=500, max_depth=6, hist, binary:logistic, seed=42`, cost-modelled over 1,000 real test
  rows; **Covertype** (54 feat, 7-class) as the multiclass robustness dataset. Sweep: #trees {100,300,
  500,1000}, depth {4,6,8,10}, batch {1,8,64,256}, N {4,8,16,32}, plus `p_mis` and overhead corners.
- **OQ-4 (fair baseline) resolved:** equal-resource scalar = an **N-wide *in-order* superscalar
  RISC-V** with the same N compare units, clock, and on-chip feature SRAM; decision uses the *stronger*
  of a branchy (data-grounded misprediction penalty) and a branchless/QuickScorer-style variant.
  Comparison to an idealised OoO core is explicitly out of scope and flagged as a Stage-B caveat.
- **OQ-1 (thresholds) resolved:** **GO** iff ρ ≥ 3× (canonical) **and** ρ ≥ 2× (all sweep points)
  against the conservative baseline **and** resident footprint ≤ 1 MB on-chip **and** overheads < 50%
  of ideal EDGE throughput. **NO-GO** if ρ < 1.5× **or** it spills **or** overheads drive effective
  ρ < 1.5×. **Gray zone** 1.5×–3× → diagnose the binding limiter, name the un-killed rival, decide with
  stated confidence (conditional-GO or NO-GO). Decision is on the **canonical config at the
  conservative corner**; the sweep is descriptive; no threshold moves after data.
- **Honesty rail.** This is an analytical cost model: every number is `[modelled]`, not `[measured]`.
  A GO *licenses building the simulator*, it does not claim a silicon win. Report effect size + a
  sensitivity range, never a single hero number. `[believed]`

---

**D-013 — Spike pre-reg Amendment A1 (pre-data engine cost-model fairness fix).** On the coderAS
pre-data **synthetic smoke** (40 trees, depth 4, N=16 — *not* decision data; no canonical HIGGS numbers
generated), the primary metric **ρ = EDGE ÷ strong-scalar node-evals/cycle was structurally < 1 for any
data** (smoke: charged `edge_rate` 2.63/1.17 vs `strong_scalar` 8.0 → ρ 0.33/0.15), so the spike could
return *only* NO-GO and could **not discriminate H0 from H1** — a modelling artifact, not a result.
experimentAS ruled on the two asymmetries **before** any decision data (per `spike-prereg.md` §10) and
recorded a timestamped **Amendment A1** (2026-07-06T12:41:41+05:30):
- **Asymmetry 1 (per-node cost) — fixed as genuine unfairness.** Both engines were N-wide but only EDGE
  paid a multi-cycle occupancy `c_edge = 1+t_tag+d_disp+g` (4/9 cyc), while the scalar was credited
  ~1 node-eval/cycle/lane. Two sub-faults: (a) feature-gather `g` and child-select are done by *both*
  engines yet charged only to EDGE; (b) charging `c_edge` as exclusive occupancy models EDGE as
  *un-pipelined*, contradicting its tagged-token/Monsoon pipelined basis (D-007). Fix: `g, d_disp` and
  the pipeline-depth part of `t_tag` become **latency** (hidden by token parallelism → critical-path /
  reduction only, `c_edge_lat = 1+t_tag+d_disp+g`); EDGE keeps two genuine throughput taxes —
  operand-network cap `R = R_factor·N` (conservative 1/2) and an un-pipelinable tag-match tax
  `τ_tag` (occupancy `c_edge_occ = 1+τ_tag`; default 0.5, conservative 1.0). Rival R-B stays killable.
- **Asymmetry 2 (scalar cross-tree width) — ruled *keep the strong baseline*.** A competent scalar
  kernel *does* get width via SIMD-across-trees (that is why QuickScorer/RapidScorer exist,
  `research/usecase.md §4`); EDGE's claim is "width *without* the branch/predication tax," not "only
  EDGE finds width." So the decision baseline stays the **width-N** `strong_scalar =
  max(branchy_swpipe, branchless_opt)` (unchanged — weakening it would strawman H0 and *help* EDGE). A
  dependency-honest `scalar_branchy_naive = 1/(1+p_mis·B)` is **added for R-A characterization only**,
  not the decision.
- **No threshold moved.** GO (ρ≥3× canonical & ≥2× sweep & G1≤1 MB & G3<50%), NO-GO (ρ<1.5× or spill),
  gray [1.5×,3×), canonical config, and the conservative-corner decision rule are all unchanged; only
  the §3.3/§3.4 engine internals changed. The fair model is **not** GO-biased: canonical-like ρ ≈ 1.27
  (default) / 0.95 (conservative) → NO-GO/low-gray, reaching GO only where the *data* inflates the
  scalar's branchless work (deep trees). H1 is now testable. coderAS to implement A1 one-to-one, then
  run canonical HIGGS. `[believed]`

---

**D-014 — Spike outcome: NO-GO on the ML/tree-inference framing (retires R7).** coderAS implemented
Amendment A1 one-to-one (17 tests) and ran the canonical HIGGS decision. Applying the frozen §5 rule:
**primary ρ = 0.84 (< 1.5× NO-GO bar) → NO-GO**, overdetermined across four independent rails —
sweep-min ρ = 0.74 and sweep-**max** only 1.41 (never nears the 2×/3× GO bars); G3 overhead fraction
0.56 (> 0.50) at the conservative corner; and G1 spills the 1 MB budget at depth≥8 or 1000 trees
(canonical 898 KB fits). **Decisive fact:** even the *overhead-free* EDGE ceiling is
`edge_ideal_rate/strong_scalar = 15.35/8.0 = 1.92×` — below the 2× sweep floor **before any EDGE
overhead is charged** — so the shortfall is structural to the workload+baseline, not an artifact of the
A1 overhead model. Strong-inference reading: **rival R-A ("the scalar is already fine") is CONFIRMED** —
HIGGS branches are predictable (data-grounded `p_mis = 0.146`), so the binding baseline is the
branchless/QuickScorer-style N-wide scalar, which harvests the *same* cross-tree parallelism EDGE
offers; ρ is invariant (1.20) across the entire p_mis sweep, proving the decision does not rest on any
branch-predictability assumption. R-B (overhead) is partially confirmed (deepens but doesn't create the
loss); R-C (spill) confirmed for larger models. **H1 refuted.** Verdict rendered with **calibrated
confidence ≈ 0.9 (high)**: it is [modelled] not [measured], and analytical models are optimistic, so
true silicon ρ is if anything ≤ modelled — a sub-2× overhead-free ceiling makes a real ≥3× win on this
target (GBDT, batch-1, vs a competent branchless scalar) implausible. **Consequence:** drop the ML/
tree-inference use-case narrative; pursue the **general-purpose open full-stack EDGE contribution**
(D-004/D-005/D-009) *without* the GBDT/AI win claim. The EDGE core is unaffected; only the use-case
framing is retired. Re-frame routed to **discussAS** (handoff task 4'). Full reasoning:
`spike/out/conclusion.md`; artifacts: `spike/out/{results.csv,decision_inputs.json,provenance.json,
plots}`. `[modelled]` (a $0/days negative result that retires R7 before the simulator one-way door). `[believed]`

---

## Open questions (decide before or during the relevant phase)
- **OQ-1** — Exact GO/NO-GO thresholds for the spike. **RESOLVED (D-012, `spike-prereg.md` §5):** GO iff
  ρ ≥ 3× (canonical) & ρ ≥ 2× (all sweep) vs the conservative baseline & resident ≤ 1 MB & overhead
  < 50%; NO-GO if ρ < 1.5× or spill; gray zone 1.5×–3× → diagnose & decide with stated confidence.
- **OQ-2** — Which tabular dataset + XGBoost config is the canonical benchmark? **RESOLVED (D-012,
  `spike-prereg.md` §6):** HIGGS (28 feat, binary), 500 trees / depth 6 / hist; Covertype as multiclass
  robustness set; full sensitivity sweep pre-committed.
- **OQ-3** — Restricted input language for the Phase-2 compiler: compile *from an XGBoost model dump*
  directly (recommended) vs a general DSL (scope risk). *(Still open — Phase-2 concern, not the spike.)*
- **OQ-4** — Baseline definition: "equal-resource scalar RISC-V." **RESOLVED (D-012, `spike-prereg.md`
  §4):** N-wide in-order superscalar, same N compare units / clock / on-chip feature SRAM; decision uses
  the stronger of branchy (data-grounded p_mis) and branchless variants; OoO comparison out of scope.
- **OQ-5** — Project name (Grove is a placeholder). *(Still open.)*

## Risk & assumption ledger
| ID | Risk / assumption | Basis | L | I | One-way? | Cheapest test | Status |
|----|----|----|----|----|----|----|----|
| R2 | "No open full-stack EDGE" is an *absence* claim | [believed] | M | H | no | Focused prior-art pass (partly done in research/) | open |
| R3 | Phase-2 block-forming compiler sinks the project | [known] hard | H | H | no | Phase-1 hand-written blocks prove microarch first | mitigated |
| R6 | AI angle is DL-training (not deliverable by EDGE) | [verified] | — | — | — | Accepted tree-inference framing instead | retired |
| R7 | GBDT win magnitude is only "meh" | [believed] Med | M | H | no | The cost-model spike (D-008) | **RETIRED → NO-GO (D-014).** Canonical HIGGS ρ=0.84 (<1.5×); overhead-free ceiling only 1.92× (<2×); rival R-A confirmed (predictable branches → branchless N-wide scalar harvests the same parallelism). ML framing dropped; general-purpose EDGE fallback (→ discussAS) |
| R8 | "Fixed-function FPGA beats you" critique | [believed] | M | M | no | Frame as programmability + open stack (D-010) | open |
