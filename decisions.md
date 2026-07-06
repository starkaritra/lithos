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

**D-015 — Direction pivot: application-driven "expose→solve the memory wall" (supersedes the D-014 general-purpose-EDGE fallback; re-scopes D-001, D-006).** After the NO-GO, the owner rejected both (a) the general-purpose-EDGE fallback (reduces to reproducing known prior art TRIPS/E2 → fails the owner's bar) and (b) a novelty-free "watchable dataflow playground" (visualization is a byproduct, not a use case). **New north star, stated by the owner:** *"either novelty or a use case — application-based learning,"* explicitly *"help me learn GPU architecture and related,"* with the design **forced by an application** (the owner's analogy: photonic computing needed a new substrate → it forced a new ISA). This **re-scopes D-001** (the project is now first-class *application-based learning*, no longer "not-a-learning-exercise") and **supersedes D-006** (tree-inference use-case, dead per D-014). Chosen shape is a two-phase **A→C arc**:
- **Arm A — a from-scratch mini-GPU (SIMT).** Build our own SIMT core + a minimal ISA + a tiny CUDA-like language → hand-rolled compiler → cycle-accurate software simulator. **Use case:** run real parallel kernels (vector-add → reduction → tiled matmul). **Payoff:** learn GPU microarchitecture *by building it* — warp execution, **warp divergence**, memory **coalescing**, **occupancy**, **latency hiding** via many resident warps. **Novelty:** ~none (Vortex, GPGPU-Sim, Accel-Sim exist) — this arm is *pure application-based GPU-architecture learning*, and it also produces the honest **baseline** that exposes the forcing function for Arm C.
- **Arm C — near-memory / Processing-In-Memory (PIM).** Motivated by a *measured* problem from Arm A: memory-bound kernels (embedding gather, SpMV, large reductions) **stall the SIMT core on DRAM bandwidth — the memory wall.** Build a PIM architecture + its own ISA where compute is moved into/near the memory banks, and **measure the reduction in off-chip bytes moved / modelled energy** vs the Arm-A baseline. **Use case + novelty:** PIM is a live research + commercial area (UPMEM, Samsung HBM-PIM) and ties to the parked **D-011** ("the real DL bottleneck is memory bandwidth"); an open, minimal ISA + compiler + a measured data-movement win is a *defensible, use-case-driven* contribution — the photonics-style "application forced a new ISA."

The arc is deliberate: **A exposes the bottleneck; C solves it.** Grove's reusable assets carry over — the cycle-accurate-simulator discipline, the hand-rolled-compiler concept, the pre-registration/measurement rigor from the spike, and the ISA-design experience. Hard constraints unchanged: **$0**, simulation-first, solo, months-scale, honest-scope (the over-scope one-way door — building a full language, a real OoO core, or a voxel engine — is the top risk, not novelty). Phasing: **Arm A = v1** (achievable, teaches GPU arch, yields the baseline); **Arm C = v2** (the novel, use-case-driven fix). `[verified: owner decision]` — User input: *"i like A and C … either novelty or use case. application based learning"* and *"the core thing should be the cs fundamentals rigor, this should help me learn gpu architecture."`

---

## Open questions (decide before or during the relevant phase)
- **OQ-6 (Arm A)** — mini-GPU ISA scope. **RESOLVED (built):** minimal 10-op ISA
  (mov/tid/iadd/imul/slt/ld/st/jmp/bra/halt) + labels, hand-written `.sasm` kernels, no general
  language/LLVM. See `simt/`.
- **OQ-7 (Arm A)** — what v1 computes/measures. **RESOLVED (built):** vector-add → reduction, with
  latency-hiding, coalescing, divergence, and the memory wall all instrumented + plotted. See
  `simt/docs/`.
- **OQ-8 (Arm C)** — PIM model + memory-bound kernel + headline metric. **RESOLVED (D-017):**
  bank + off-chip-bandwidth-capped model; **headline kernel = recommendation-style embedding-bag
  sum-pooling (A)**, with array-reduction (B) as the de-risk first slice and a reduction-ratio
  **sweep (D)** for honesty; **primary metric = off-chip bytes moved** (+ modelled energy secondary).
- **OQ-9** — Is there a visualization layer (browser, reusing the owner's loupe experience) or is v1 headless-sim + CLI/plots only? **RESOLVED (D-016):** headless **C++ cycle-accurate core + Python analysis layer** (the gem5/GPGPU-Sim industry pattern); browser viz is optional later.

---

**D-016 — Arm A stack: C++ cycle-accurate core + Python analysis (resolves OQ-9).** The owner asked for the *industry standard*, not the convenient choice. The standard for cycle-accurate architecture simulators is **C++** (gem5, GPGPU-Sim/Accel-Sim, SystemC); Python's real role is the config/analysis layer around it (gem5 is literally a C++ core scripted by Python). Given the north star (CS-fundamentals rigor + learn GPU architecture + portfolio signal, D-015), the mini-GPU **engine is C++17** built with **CMake + Ninja + MSVC** (VS 2022 Build Tools — already installed, $0), and the **analysis/plots layer is Python** (reusing the spike's matplotlib rig). Clean engine/render split mirrors the loupe ingest/engine separation and keeps the core headless + unit-testable. A browser visualization is an optional later wrapper, not v1. Rust was considered (modern, memory-safe) but the established arch-sim ecosystem is C++, so C++ is the standard answer. `[verified: gem5/GPGPU-Sim/SystemC are C++; VS2022 BuildTools + bundled CMake/Ninja verified present]` — User input: *"why python? what is the industry standard?"*

---

**D-017 — Arm C scope: near-memory / PIM to solve the memory wall (resolves OQ-8).** Arm A
*measured* the wall (D-015; `simt/docs/09`): ~one memory access ≈ 225 arithmetic ops, and a
scattered gather spends ~99% of its cycles moving data. Arm C's thesis: **stop moving the data —
put small compute units in the memory banks so only results cross the off-chip link.** Honest
framing (as with Arm A): PIM is prior art (UPMEM, Samsung HBM-PIM, Mutlu/Ghose's line); the
contribution is *an open, minimal, cycle-accurate near-memory model + a measured, fair
data-movement result that shows where PIM wins **and where it doesn't**.* `[verified: PIM is a live
research + commercial area]`

- **Architecture (concrete, $0):** global memory split into `B` **banks**, each owning a contiguous
  address slice; each bank has a tiny **PIM compute unit (PCU)** doing simple ops (add / MAC /
  compare) on data resident in *its* bank. An off-chip **link with a bandwidth cap** (bytes/cycle)
  connects banks↔host. Baseline (the Arm-A GPU) drags all operands across the link; PIM aggregates
  locally and sends only the small result.
- **Minimal PIM ISA (~6 ops, mirrors Arm A's discipline):** `pim_load` (bank-local), `pim_add` /
  `pim_mac`, `pim_reduce` (fold a bank's slice), `host_collect` (gather `B` partials), `halt`.
- **Bandwidth-capped model (the `simt/docs/09` prerequisite):** add ONE off-chip bytes/cycle ceiling
  to a *shared* memory model, so **both** the GPU baseline and PIM are limited by the same link —
  the fairness spine (echoing the spike's OQ-4 "don't rig the baseline" lesson).
- **Headline kernel = recommendation-style embedding-bag sum-pooling (owner choice "A")** — grab a
  handful of scattered rows from a huge table and sum them into one vector (the #1 memory-bound
  workload in industrial recsys / DLRM; ties to D-011). **De-risk first slice = array reduction
  ("B")** (reuses `simt` ch.08). **Honesty layer = a reduction-ratio sweep ("D")** across kernels
  from "boils a lot down to a little" (PIM wins) to "needs all the data" / pure gather (PIM barely
  helps) — this makes the result non-tautological by showing the crossover.
- **Metric (OQ-8):** primary = **off-chip bytes moved** (assumption-free; the thing PIM reduces),
  baseline vs PIM as a reduction factor + sensitivity; secondary = **modelled data-movement energy**
  (bytes × a cited pJ/byte order-of-magnitude). Effect size + range, never a hero number.
- **Stack:** a new `pim/` module beside `simt/`, sharing a common C++ bandwidth/bank model + the
  Python analysis layer (D-016 pattern).
- **Parked (scope traps):** general PIM compiler/language; DRAM-faithful timing (refresh, row
  buffers); RTL; elaborate energy modelling; multi-level (near-cache + near-DRAM) PIM.
- **Next:** experimentAS **pre-registers** the measurement (bank/bandwidth params, the reduction-
  ratio sweep, primary/secondary metrics, and a decision rule that admits an honest NO-GO if the
  useful regime is too narrow) **before** coderAS generates any numbers.
`[believed — owner-approved shape]` — User input: chose *"A as headline + B to de-risk + D for honesty."`

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

---

**D-018 — Arm C measurement pre-registered (near-memory / PIM data-movement study).** experimentAS
pre-registered the Arm C measurement in **`pim-prereg.md`** before any number is generated (mirrors the
spike's pre-reg discipline; append-only-from-data; amend-with-timestamp if unimplementable). Key
commitments:
- **Primary quantity = DMR (data-movement reduction) = off-chip link bytes(baseline GPU) ÷ off-chip
  link bytes(PIM)**, same kernel/data/bandwidth-cap for both. Bytes are *counted*, not modelled
  (`[modelled-exact]`) — this makes the primary metric assumption-free (mitigates C3). Reported as an
  effect-size **range** + the **crossover**, never a hero number.
- **H1:** DMR ≥ 3× at the canonical DLRM config **and** ≥ 1.5× across the realistic sweep **and** the
  crossover sits outside the realistic regime. **H0:** DMR < 1.5× once *all* link traffic is counted, or
  the useful regime is an implausible corner. **Rivals to kill:** R-tautology (aggregation trivially
  sends less → killed by the reduction-ratio sweep + naive-vs-honest DMR + mandatory crossover),
  R-hidden-traffic (indices + `k(L,B)` per-bag partials + output erase the win → charged to both sides),
  R-baseline-strawman (baseline is the *coalesced byte floor* — moves each byte once), R-banking-tax
  (more banks → more partials → smaller byte-win; surfaced, decided at realistic `B`).
- **Fair baseline (anti-rig spine):** a competent coalesced Arm-A GPU that moves each needed byte
  exactly once, under the **same single off-chip bandwidth cap** as PIM (closes the `simt/docs/09`
  missing-bandwidth-ceiling gap with ONE shared cap — no DRAM model, per C2).
- **Anti-tautology core (C1):** a reduction-ratio sweep over pooling factor `L` (pure-gather `L=1` →
  aggressive `L≫B`), reporting the crossover `L*` where DMR falls to 1.5×, with real DLRM params placed
  relative to it. The honest DMR = `L / k(L,B)` (banking factor `k` = distinct banks a bag touches), far
  below the naive `1/RR = L` line — the gap is the evidence it is not a definitional win.
- **Decision rule admits an honest NO-GO (C4):** GO (write up Arm C) iff canonical DMR ≥ 3× + robust
  ≥ 1.5× + crossover-honest; NO-GO/weak iff < 1.5× once fully counted or the useful regime is a narrow
  corner; gray [1.5×,3×) → diagnose the binding limiter + name the un-killed rival + decide with stated
  confidence. Kernels: reduction (de-risk smoke + tautological endpoint, **not** the decision kernel) →
  embedding-bag sum-pooling (headline, the decision kernel) → optional SpMV. Secondary = modelled energy
  (Horowitz ~160 pJ/byte, swept {50,160,640}, framed as an *upper bound* on PIM's advantage — C3).
- **Stack/repro (D-016):** C++ bank/bandwidth/byte-counter core + Python sweep/analysis; one-command run;
  seeds + `provenance.json`; build on dedicated local branch **`exp/d017-pim`** (all Arm C runs share it).
`[believed — experimentAS design]` Resolves the OQ-8 measurement plan. **Next:** coderAS builds to
`pim-prereg.md`; experimentAS renders GO/NO-GO from the returned `decision_inputs.json`.

**D-019 — Arm C measured: CONDITIONAL-GO (data-movement win is real but banking-limited; scoped to
high-pooling features).** coderAS built the byte-accounting model to the D-018 spec and ran the full
sweep (branch `exp/d017-pim`, commit `e33b365`, 8/8 model tests pass); experimentAS verified the
artifacts (Twyman's-law re-derivations all pass — see `pim/out/conclusion.md` §7) and applied the
**frozen §5 rule mechanically**: no threshold moved after data. **Outcome = GRAY ZONE → CONDITIONAL-GO,
calibrated confidence ~0.82.** The full reasoning + rival scorecard + plots are in
**`pim/out/conclusion.md`** (the CONCLUDE artifact).
- **The numbers (`[modelled-exact]` bytes):** canonical embedding-bag (`d=64, L=40, B=16, nb=1024`,
  round-robin, coalesced baseline, all link traffic charged) → **DMR = 2.53×** (`k_empirical=14.84` ≈
  closed-form 14.79; overhead fraction 0.099). `2.53× ∈ [1.5×,3×)` ⇒ GO-primary (≥3×) fails, NO-GO-primary
  (<1.5×) not triggered, and the realistic-range sweep-min (**1.20×** at L=8) fails GO-robustness ⇒
  **squarely the §5 gray zone.**
- **Binding limiter = the banking factor `k` (R-banking-tax).** `DMR ≈ L/k(L,B)`; at B=16, L=40 nearly
  all banks are touched (`k=14.84/16`), so PIM ships ~B partials/bag. The B-sweep at L=40 is the smoking
  gun: DMR = 7.40× (B=4) → 4.34× (B=8) → **2.53× (B=16)** → 1.68× → 1.32× → 1.15× (B=128). Hidden traffic
  is small (overhead 0.099) and clustered placement does **not** rescue it (2.540× vs 2.529× — random
  DLRM indices defeat locality). The crossover `L*(B=16)=32` sits just left of the DLRM L=40 point (thin
  margin); low-pooling features (L=8→1.20×, L=16→1.49×) fall *below* the worth-it line.
- **Rival scorecard (Platt):** **R-tautology KILLED** (honest DMR ≪ naive `DMR=L` line; real crossover
  L=1→DMR=1.000 exactly — this retires C1); **R-hidden-traffic KILLED** (charged both sides, overhead
  0.099, win survives); **R-baseline-strawman KILLED** (baseline = coalesced floor, each row once);
  **R-banking-tax CONFIRMED** as the honest, un-killed binding limiter. 3 of 4 killed; H1 (≥3× + robust +
  crossover-outside) not fully supported, H0 (no win / pure tautology) refuted → the nuanced middle.
- **Decision:** conditional-GO **licenses a use-case-driven, honestly-scoped *data-movement* write-up** —
  "near-memory sum-pooling gives a substantial off-chip-byte reduction (≥3×, rising to >12×) for
  **high-pooling embedding features (L ≳ 64 at realistic B=16)**, is fundamentally banking-limited
  (DMR ≈ L/k, k → B), and crosses below the worth-it line for low-pooling features (L ≲ 16)." It claims
  **no** silicon speedup and **no** flat win across all features (C2/T1 honored). Energy DMR is
  e-invariant (=byte DMR) and framed as an upper bound (C3 honored).
- **The one observation that would change the call (confidence 0.82, not higher):** the **pooling-factor
  distribution of the target DLRM workload.** If a meaningful mass of features sit right of `L*=32`,
  conditional-GO strengthens toward unconditional GO; if the mass is low-pooling (L ≲ 16), the useful
  regime is an implausible corner ⇒ tips to **NO-GO/weak** (§5 NO-GO cond. 2). Cheapest next test: cite
  published DLRM/MLPerf per-feature pooling shapes and mark their mass vs `L*(B)`.
`[believed — experimentAS verdict on `[modelled-exact]` data]` Resolves OQ-8. **Branch `exp/d017-pim`
is NOT merged** — the merge-to-main decision is the owner's/coderAS's, deferred to after this verdict.
**Next:** owner/coderAS decide merge; if the write-up proceeds, pin the pooling-factor distribution first.

## Risk & assumption ledger
| ID | Risk / assumption | Basis | L | I | One-way? | Cheapest test | Status |
|----|----|----|----|----|----|----|----|
| R2 | "No open full-stack EDGE" is an *absence* claim | [believed] | M | H | no | Focused prior-art pass (partly done in research/) | open |
| R3 | Phase-2 block-forming compiler sinks the project | [known] hard | H | H | no | Phase-1 hand-written blocks prove microarch first | mitigated |
| R6 | AI angle is DL-training (not deliverable by EDGE) | [verified] | — | — | — | Accepted tree-inference framing instead | retired |
| R7 | GBDT win magnitude is only "meh" | [believed] Med | M | H | no | The cost-model spike (D-008) | **RETIRED → NO-GO (D-014).** Canonical HIGGS ρ=0.84 (<1.5×); overhead-free ceiling only 1.92× (<2×); rival R-A confirmed (predictable branches → branchless N-wide scalar harvests the same parallelism). ML framing dropped; general-purpose EDGE fallback (→ discussAS) |
| R8 | "Fixed-function FPGA beats you" critique | [believed] | M | M | no | Frame as programmability + open stack (D-010) | open |
| C1 | Arm C PIM win is a construction tautology (aggregation trivially sends less) | [believed] | H | H | no | Reduction-ratio sweep: show where PIM wins AND fails (crossover) | **RESOLVED → KILLED (D-019).** Honest DMR (2.53× canonical) sits far below the naive `DMR=L` line; a real crossover exists (L=1→DMR=1.000 exactly, L*(B=16)=32). The win is bounded by `L/k`, not the definitional `L`. Tautology refuted; C1 retired |
| C2 | Over-scoping the DRAM model (refresh/row-buffers) sinks Arm C | [believed] | M | H | no | Minimal bank + single bandwidth cap only | **held (D-019).** Model stayed minimal (one shared cap, no refresh/row-buffers); the decision rested only on countable link bytes (`[modelled-exact]`). No over-scope; CONDITIONAL-GO claims no silicon speedup (T1) |
| C3 | Arm C win rests entirely on the energy-per-byte assumption | [believed] | M | M | no | Report bytes-moved (assumption-free) as the primary metric | **held (D-019).** Primary = counted bytes; energy DMR proved **e-invariant** (=byte DMR, ratio cancels e) across e∈{50,160,640} and framed as an upper bound. Verdict does not depend on the pJ/byte assumption |
| C4 | Arm C PIM byte-reduction is marginal even for aggregation → no honest win | [guess] | L | H | no | experimentAS pre-registered measurement admits a NO-GO | **RESOLVED → CONDITIONAL-GO (D-019).** Canonical DMR=2.53× (gray zone); win is real but **banking-limited** (R-banking-tax the binding limiter, DMR≈L/k, k→B) and robust only for **high-pooling features (L≳64)**, failing <1.5× for low-pooling (L≲16). Honestly scoped, not a flat win. Calibrated confidence ~0.82; the pooling-factor distribution of the target workload is the observation that would flip it to unconditional-GO or NO-GO/weak. See `pim/out/conclusion.md` |
