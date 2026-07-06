# Grove — Arm C Pre-Registration (near-memory / PIM data-movement measurement)

**Status:** pre-registered / awaiting build. **Owner:** experimentAS (design) → coderAS (build).
**Anchor decision:** `decisions.md` D-015 (A→C arc), D-016 (C++ core + Python analysis),
**D-017 (Arm C scope)**, **D-018 (this pre-registration)**. **Resolves:** the OQ-8 measurement plan.
**Addresses risks:** C1 (tautology), C2 (DRAM over-scope), C3 (energy-assumption), C4 (honest NO-GO).
**Date pre-registered:** 2026-07-06. **Confidence tags:** `[verified]` primary-source checked ·
`[believed]` reasoned · `[guess]` low grounding.

> This document is **append-only from the moment data is generated.** The hypotheses, primary
> quantity, byte-accounting rules, sweep, and decision rule below are committed **before** any number
> is produced, so the conclusion cannot be bent to fit the result. Any analysis not listed here is
> **exploratory** and must be labelled as such in a separate `pim-exploratory.md`, never reported as a
> confirmatory test. If a definition proves unimplementable, coderAS returns here to **amend the
> pre-registration with a timestamp** *before* generating numbers — exactly as the spike's Amendment
> A1 worked. *(Nosek pre-registration; Kerr anti-HARKing; Feynman "don't fool yourself.")*

---

## 0. What this is — and what a GO actually licenses (read this first)

Arm A built a mini-GPU and **measured** the memory wall (`simt/docs/09`): ~one memory access ≈ 225
arithmetic ops, and a scattered gather spends ~99 % of its cycles moving data, not computing. Arm C's
thesis is the only real escape from that: **stop moving the data — put tiny compute units in the
memory banks so that, for aggregation kernels, mostly *results* cross the off-chip link instead of raw
operands.** This document pre-registers the measurement that decides whether that thesis holds *under
a fair comparison* — or is a construction tautology (risk C1) / a marginal win (risk C4).

**It is an analytical, cycle-approximate architectural model — not silicon, not a full DRAM
simulator.** Be honest about the epistemic status of each number:

- **Off-chip bytes moved is a faithful, assumption-light architectural quantity.** Given a kernel, a
  data placement, and the algorithm each machine runs, the number of bytes that must cross the
  bank↔host link is *counted deterministically*, not estimated. This is the primary metric precisely
  because it does **not** depend on a timing or energy model (mitigates C3). We tag it `[modelled-exact]`.
- **Cycles and energy are genuinely modelled** (a single bandwidth cap; a cited pJ/byte constant).
  They are **secondary**, reported with sensitivity, and tagged `[modelled]`.
- Like all analytical models it omits second-order effects (bank conflicts, refresh, real DRAM row
  buffers, in-bank compute latency). Per D-017/C2 those are **deliberately parked**; we do not model a
  DRAM-faithful timing stack, and we do not let the win rest on anything we cannot count.

**Therefore a GO means:** *"under a fair, bandwidth-capped comparison against a competent coalesced
GPU baseline, near-memory aggregation delivers a substantial and robust reduction in off-chip bytes on
a realistic recommendation-style workload, with a clearly characterised regime where it stops
winning — enough to justify writing up Arm C as a use-case-driven contribution."* It does **NOT**
claim a silicon speedup, does **NOT** claim victory over a real HBM-PIM product, and does **NOT** claim
the win holds outside the characterised regime. Everything decision-driving is tagged `[modelled…]`.

A **NO-GO is a fully legitimate, valuable outcome** (risk C4): if the byte reduction is marginal once
*all* link traffic is honestly counted, or the useful regime is implausibly narrow, we say so and Arm C
is scoped down or reframed. Arm C is **not** assumed to succeed.

---

## 1. Hypotheses (falsifiable — Popper / Platt)

### 1.1 The primary quantity — DMR (defined precisely)

Let the **off-chip link** be the single channel connecting the memory banks to the host/compute. For a
given kernel run, let `Bytes_link` = the total number of bytes that must cross that link to complete
the kernel (operands in/out + indices + partial/final results — see §3 for the full accounting).

> **DMR — Data-Movement Reduction — is the primary quantity:**
> **`DMR = Bytes_link(baseline GPU) ÷ Bytes_link(PIM)`**, both machines running the **same kernel on
> the same data under the same bandwidth cap.**

`DMR > 1` means PIM moves *less* across the link; `DMR = 10` means "10× fewer off-chip bytes." We choose
this orientation (baseline ÷ PIM, "reduction factor") over its inverse because (a) it reads as an
intuitive "×-fewer-bytes" number, (b) it is the direct architectural quantity PIM is built to shrink,
and (c) it is scale-free — absolute byte counts cancel, so the result does not depend on table size or
batch size except where those genuinely change the ratio. We always report DMR as an **effect size with
a range** across the sweep, never a single hero number, plus the **crossover** where DMR falls to the
"not worth it" line (§4).

### 1.2 Hypotheses

On the **headline kernel (embedding-bag sum-pooling)** at the canonical DLRM-style config (§6),
under the fair shared bandwidth cap (§2.3) and the full link-traffic accounting (§3):

- **H0 (null):** Near-memory PIM holds **no useful, robust data-movement advantage.** Concretely,
  **DMR < 1.5×** at the canonical config once *all* link traffic is counted (indices, gathered
  rows/partials, and results on **both** sides); **and/or** any apparent advantage is a **tautology**
  that evaporates in the pure-gather regime — i.e. the useful (DMR ≥ 1.5×) region is confined to
  parameter corners that a real recommendation workload does not occupy.
- **H1 (alternative):** PIM achieves a **substantial and robust** data-movement reduction on
  aggregation kernels: **DMR ≥ 3×** at the canonical config against the competent coalesced baseline;
  **and** DMR ≥ 1.5× across the realistic DLRM parameter range (not a knife-edge); **and** the
  **crossover** where PIM stops winning is clearly characterised and sits *outside* the realistic
  regime (i.e. real recsys parameters land in the winning region, not the gather region).

**The refuting observation (what would prove H1 wrong):** DMR < 1.5× at the canonical config against the
competent coalesced baseline; **or** the DMR ≥ 1.5× region requires parameters a real DLRM workload
does not hit (e.g. wins only when the per-bag pooling factor `L` vastly exceeds the bank count `B`, or
only when `B` is implausibly small); **or** once index + result traffic is fully charged to the PIM
side, the ratio collapses to ~1.

### 1.3 Rival hypotheses this design is built to KILL (Platt — strong inference)

The design must be able to *exclude at least one* of these, not merely confirm H1:

- **R-tautology — "PIM wins only because aggregation trivially outputs less."** The naive expectation
  is `DMR = 1/RR` (the kernel's aggregation factor: `L` rows → 1, or `N` elements → 1). If that were the
  whole story, the "win" would be a definitional artefact of picking an aggregating kernel.
  → **Killed/characterised by** the reduction-ratio sweep (§4): we plot the naive tautological line
  `DMR_naive = 1/RR` *against* the honestly-accounted `DMR = 1/RR` moderated by banking + hidden
  traffic, and **report the crossover** where PIM stops being worth it. The *gap* between the two lines
  is the anti-tautology evidence: banking forces `k` partials out (not 1), so the real DMR is far below
  the naive line and *has* a crossover.
- **R-hidden-traffic — "the indices, the scattered rows, and the partial results that still cross the
  link erase the win."** → **Killed/characterised by** the full link-traffic accounting (§3): every
  byte crossing the link is charged to **both** machines, including the `k(L,B)` per-bag partials PIM
  must still ship out, the input indices, and the final result. We sweep the embedding dimension `d`
  down and index width up to expose where this overhead dominates.
- **R-baseline-strawman — "the GPU baseline wasn't a competent, coalesced machine."** → **Killed by
  the baseline definition** (§2.3, §3): the baseline moves each needed byte across the link **exactly
  once** (perfectly coalesced, no re-reads, no wasted transactions), which is the *theoretical floor*
  for the baseline's byte count. We deliberately hand the baseline its best case, so any PIM win is not
  an artefact of a crippled baseline. (Bytes, unlike cycles, are not inflated by poor coalescing — so
  using bytes as the primary metric is itself the strongest anti-strawman choice.)
- **R-banking-tax — "more banks help throughput but *hurt* the byte win."** A genuine architectural
  tension we surface honestly: with `B` banks, PIM must ship one partial per *touched* bank, so large
  `B` (good for compute parallelism) shrinks DMR. → **Characterised by** the `B` sweep (§4, §6); the
  decision is made at a realistic `B`, and the parallelism↔byte-win tradeoff is reported, not hidden.

---

## 2. Primary metric, guardrails, and the fair baseline (Kohavi — OEC + guardrails up front)

### 2.1 Primary metric (the OEC)
**DMR = off-chip link bytes (baseline) ÷ off-chip link bytes (PIM)** (§1.1), on the headline
embedding-bag kernel at the canonical config, reported as: the canonical point estimate **plus** the
min–max band over the pre-committed realistic sweep **plus** the crossover location (§4). Bytes are
counted, not modelled (`[modelled-exact]`).

### 2.2 Guardrail metrics (must be satisfied / reported — not maximised)
- **G1 — Count ALL link traffic on BOTH sides; never undercount PIM.** The byte accounting (§3) must
  include, for each machine: input **indices**, **gathered rows / operands**, **partial results**, and
  the **final output**. Report the **PIM overhead fraction** = (index bytes + partial-collect bytes +
  output bytes) ÷ total PIM link bytes. If PIM's win rests on ignoring any of these, that is a defect,
  not a result. *(kills R-hidden-traffic)*
- **G2 — Baseline is the coalesced floor.** The baseline must move exactly the *useful* bytes once
  (Σ needed rows/elements + result + indices), i.e. the minimum any competent coalesced GPU could
  achieve. Report it explicitly so the ratio is auditable. *(kills R-baseline-strawman)*
- **G3 — Crossover exists and is located.** The reduction-ratio sweep must exhibit a regime where
  DMR → ~1 (PIM not worth it). A design with no crossover anywhere is a red flag for a tautology and
  must be investigated before any GO. *(kills R-tautology)*
- **G4 — Fair bandwidth cap is shared.** The *same* off-chip bytes/cycle cap limits both machines in
  the secondary cycle model (§2.3). Report the cap used and confirm it is identical for both.

### 2.3 Baseline definition (the fairness spine — echoes the spike's OQ-4 "don't rig the baseline")
**Baseline = a competent, coalesced Arm-A-style SIMT GPU** that:
1. Moves each operand it needs across the link **exactly once**, perfectly coalesced (no redundant
   re-reads, no wasted transactions). This is the *lower bound* on the baseline's byte count — we make
   the baseline as strong as physically possible on the primary metric.
2. Is subject to the **same single off-chip bandwidth cap** (bytes/cycle) as PIM — one shared ceiling
   limits both (the anti-rigging commitment; the `simt/docs/09` prerequisite that the current mini-GPU
   lacks a cross-warp bandwidth roof is closed here by adding **one** shared cap, not a DRAM model —
   per C2).
3. Does its aggregation on-chip after the data arrives (that is exactly the traffic PIM avoids).

PIM gets **no** free memory advantage: it reads its bank-local data (in-bank, off the link — that is
the point) but pays for every byte that still crosses the link (partials, indices, output).

### 2.4 Secondary metric — modelled energy (C3: explicitly secondary, sensitivity-swept)
**Energy_dm = Bytes_link × e**, where `e` = data-movement energy per off-chip byte. We report the
**energy DMR** (which, since energy ∝ link bytes, ≈ the byte DMR) and its **sensitivity to `e`**.
- Cited order of magnitude: off-chip DRAM access ≈ **~160 pJ/byte** (Horowitz, ISSCC 2014,
  "Computing's Energy Problem" — a 32-bit DRAM access ≈ 640 pJ ⇒ ~160 pJ/byte; on-chip SRAM/compute is
  1–2 orders of magnitude cheaper). `[believed — order-of-magnitude; coderAS to pin the exact citation]`
- **Swept** over `e ∈ {50, 160, 640}` pJ/byte. **Honesty note:** we do **not** credit PIM's in-bank
  data movement as free — but since we count only *link* bytes for energy, the energy DMR is an
  *upper bound* on PIM's true energy advantage. We state this so energy is never over-claimed; the
  assumption-free **byte** DMR remains the primary, decision-driving quantity (mitigates C3).

---

## 3. The model coderAS implements one-to-one (precise, auditable byte accounting)

A shared C++ **bank + bandwidth-capped memory model** underlies all kernels (new `pim/` module beside
`simt/`, sharing the memory/bandwidth concepts and the Python analysis layer — D-016). Global memory is
split into **`B` banks**; row/element `i` is placed in bank `place(i)` (default **round-robin**,
`i mod B`; a **clustered** placement is a swept alternative — §6). Each bank has a tiny PCU
(`pim_add`/`pim_mac`/`pim_reduce`). One off-chip link carries **`cap` bytes/cycle** (shared, §2.3).

**Notation (all pre-registered; defaults + sweep in §6):**
| Symbol | Meaning | Default |
|---|---|---|
| `b` | bytes per data element (fp32) | 4 |
| `idx_b` | bytes per index | 4 |
| `B` | number of banks | 16 |
| `d` | embedding dimension (elements per row) | 64 |
| `L` | rows per bag (pooling factor) | 40 |
| `nb` | number of bags (batch) | 1024 |
| `R_tab` | rows in the embedding table | 4,000,000 |
| `N` | array length (reduction kernel) | 16,000,000 |
| `cap` | off-chip link bytes/cycle (shared) | 32 |
| `e` | data-movement energy per off-chip byte (pJ) | 160 |

**Bank-occupancy function** `k(L, B)` = the number of *distinct banks* touched by a bag's `L` rows —
i.e. the number of partials PIM must ship out for that bag. Under random indices + round-robin
placement its expectation is `k(L,B) = B·(1 − (1 − 1/B)^L)` (a coupon-collector occupancy). coderAS
computes `k` **empirically** per bag from the actual sampled indices + placement (fixed seed), and also
reports the closed-form expectation as a cross-check. `k` is the crux of the anti-tautology result:
naive intuition assumes PIM sends **1** result per bag; banking forces it to send **`k` ≥ 1**.

### 3.1 Kernel B — array reduction (de-risk first slice; reuses `simt` ch.08)
Sum `N` elements → 1 scalar. Contiguous, no indices.
- **Baseline link bytes** `= N·b + b` (read every element once across the link; write the scalar) ≈ `N·b`.
- **PIM link bytes** `= B·b + b` (each bank `pim_reduce`s its `N/B` local elements → 1 partial; host
  collects `B` partials → 1 result) ≈ `B·b`.
- **`DMR_reduction = N·b / (B·b) = N/B`.**
> **Role:** reduction is the **extreme aggregation endpoint** (`RR = 1/N`) — it is expected to show a
> *huge* DMR and is therefore **near the tautological corner**, deliberately. Its job is to (a) prove
> the bank/reduce/collect plumbing and the byte counter are correct (a de-risk **smoke**), and (b)
> anchor the far-left ("boils everything down") end of the sweep. **The GO decision does NOT rest on
> reduction** — it rests on the headline embedding-bag kernel, whose win is bounded and non-trivial.

### 3.2 Kernel A — embedding-bag sum-pooling (HEADLINE)
For each of `nb` bags: gather `L` scattered rows (each a `d`-vector) from the `R_tab`-row table and sum
them element-wise into one `d`-vector output.

**Per bag, the full link accounting (charge every crossing to both machines — G1):**

| Traffic | Baseline GPU | PIM |
|---|---|---|
| Indices in (host → memory: which rows) | `L·idx_b` | `L·idx_b` *(same — cancels, but counted)* |
| Operands across link | `L·d·b` (gather all `L` rows to compute) | `k(L,B)·d·b` (each touched bank pre-sums its rows → ships **one** `d`-partial) |
| Result out (write `d`-vector) | `d·b` | `d·b` |
| **Per-bag total** | **`L·idx_b + L·d·b + d·b`** | **`L·idx_b + k·d·b + d·b`** |

- **Whole-workload link bytes** = `nb ×` the per-bag total for each machine.
- **`DMR_embedding = (L·idx_b + L·d·b + d·b) / (L·idx_b + k(L,B)·d·b + d·b)`.**
  For large `d` and small `idx_b` this ≈ `L / k(L,B)` — **not** the naive `L`. The banking factor `k`
  and the (index + output) hidden traffic are exactly what separate an honest result from a tautology.

**Why this is not rigged, stated plainly:** the baseline moves the *minimum* possible bytes (each row
once, coalesced — §2.3/G2). PIM's saving comes solely from summing co-banked rows *before* they cross
the link. When `L ≤ B` (few, widely scattered rows — each lands in its own bank), `k ≈ L`, every bank
ships a full row-sized partial, and **DMR → 1: PIM barely helps** (this is the pure-gather regime, and
it is the crossover). When `L ≫ B`, many rows collapse per bank, `k ≈ B`, and **DMR → L/B**.

### 3.3 Optional Kernel C — SpMV (sparse matrix-vector), only if cheap after A+B
`y = A·x`, `A` sparse (`nnz` nonzeros, `nr` rows). Each `y_i` = MAC over the nonzeros of row `i`.
Baseline gathers the scattered `x[j]` operands + `A` values across the link; PIM (`pim_mac`) accumulates
per row in-bank and ships one scalar per bank-resident output. Accounting analogous to §3.2 with the
per-row nonzero count playing the role of `L`. **Scope guard:** SpMV is a *stretch* item to broaden the
sweep's middle; it is **not required** for the decision and is deferred unless A+B land cheaply (C2).

---

## 4. The reduction-ratio sweep — the anti-tautology core (addresses C1 directly)

This is the heart of an honest result. We define an explicit sweep axis and **pre-commit to reporting
the crossover** where PIM stops being worth it.

### 4.1 The sweep axis
**Kernel reduction ratio `RR = output_bytes / input_bytes`** (how aggressively the kernel aggregates).
Equivalently we sweep the **aggregation factor `1/RR`**:
- **`RR = 1` (aggregation factor 1) — pure gather / copy** (e.g. embedding-bag with `L = 1`, or a
  scatter/copy kernel): output = input, nothing to pre-aggregate → **PIM cannot help, DMR → 1.**
- **`RR = 1/L` — embedding-bag sum-pooling** (`L` rows → 1 row): the headline, tunable via `L`.
- **`RR = 1/N` — full array reduction** (`N` elements → 1 scalar): extreme aggregation → **PIM wins
  hugely** (`N/B`), the tautological corner.

The primary realistic sweep varies **`L` from 1 (gather) upward through the DLRM range and beyond**, at
several bank counts `B`, so the aggregation factor moves continuously from "no win" to "big win."

### 4.2 What we pre-commit to plot and report
1. **DMR vs aggregation factor `1/RR`** (log-x), one curve per `B`, over `L ∈ {1,2,4,8,16,32,40,64,128,256}`.
2. **On the same axes, the naive tautological line `DMR_naive = 1/RR`.** The visible **gap** between
   `DMR_naive` and the honestly-accounted `DMR` *is* the anti-tautology evidence (banking + hidden
   traffic pull the real curve far below the naive line).
3. **The crossover `L*`** — the smallest `L` at which `DMR ≥ 1.5×` (the "worth it" line) — reported per
   `B`, with the realistic DLRM operating point marked on the axis. If the DLRM point sits *right of*
   `L*` (in the winning region) with margin, H1's regime claim holds; if it sits *left of* `L*`, that is
   evidence for H0 / R-tautology.
4. **The `B`-tension curve:** DMR vs `B` at fixed `L`, annotated with the note that small `B` inflates
   the byte win but reduces compute parallelism (R-banking-tax) — so the decision `B` must be realistic.

> **The crossover is the deliverable that makes this not a rigged demo.** A result that only ever shows
> "PIM wins big" would be a tautology; a result that shows *exactly where PIM wins and where it does
> not*, and places real DLRM parameters relative to that boundary, is honest science.

---

## 5. Pre-committed GO / NO-GO decision rule (justified thresholds; admits an honest NO-GO — C4)

Decided **before** any data. The decision is made on the **headline embedding-bag kernel** at the
**canonical config** (§6), against the **competent coalesced baseline** under the **shared bandwidth
cap**, with **all** link traffic charged to both sides (§3). The sweep characterises robustness and the
crossover; it is **not** mined for a favourable point (anti-p-hacking / anti-HARKing). No threshold
moves after data is seen.

### GO — write up Arm C as a use-case-driven contribution. **All** must hold:
1. **Primary:** `DMR ≥ 3×` at the canonical config (embedding-bag, `d=64, L=40, B=16, nb=1024`, random
   indices + round-robin placement) against the coalesced baseline, with all link traffic counted.
2. **Robustness:** `DMR ≥ 1.5×` across the *entire realistic DLRM sweep range* (§6) — not a knife-edge.
3. **Regime honesty (G3):** a crossover `L*` exists and the realistic DLRM operating point sits in the
   **winning** region (right of `L*`) with margin; i.e. the useful regime is not an implausible corner.
4. **Accounting integrity (G1/G2):** the PIM overhead fraction is reported and the win survives it; the
   baseline is the coalesced floor.

### NO-GO / WEAK — scope down or reframe Arm C. **Any** triggers:
1. `DMR < 1.5×` at the canonical config once all link traffic (indices + `k` partials + output) is
   counted; **or**
2. The `DMR ≥ 1.5×` region requires parameters a real DLRM workload does not occupy — e.g. it needs
   `L ≫ B` well beyond realistic pooling factors, or an implausibly small `B` — i.e. the useful regime
   is a narrow corner; **or**
3. There is **no** crossover / the "win" tracks the naive `1/RR` line so closely that it is a
   definitional tautology with no honest operating regime.

**A NO-GO is a real, publishable finding** ("near-memory sum-pooling's data-movement win is
banking-limited and marginal at realistic recsys pooling factors") and is an acceptable end state.
Arm C is not assumed to succeed (C4).

### GRAY ZONE — `DMR ∈ [1.5×, 3×)` at the canonical config: **do not greenlight blindly.**
Report honestly and **diagnose the binding limiter** from the accounting + the sweep: is it the
banking factor `k` (⇒ revisit `B`, or a locality-aware/clustered placement)? the index/output hidden
traffic at small `d`? the crossover sitting too close to the DLRM point? Name **which rival hypothesis
remained un-killed**, then make a judgement call with **explicitly stated calibrated confidence**.
Legitimate gray outcomes: (a) *conditional-GO* scoped to the specific regime where DMR clears the bar
(e.g. high-pooling features), stated as such; or (b) *NO-GO/weak*.

### Threshold justification `[believed]`
- **3× GO:** a 3× off-chip byte reduction is a *defensible, non-trivial* architectural result that
  clears the "is this worth writing up?" bar, and it demands the win survive the banking factor `k`
  (which pulls the honest DMR well below the naive aggregation factor `L`). It is high enough that the
  result is not a knife-edge artefact of one parameter.
- **1.5× NO-GO:** below ~1.5× once all hidden traffic is charged, the near-memory story is marginal —
  the plumbing, indices, and per-bank partials have eaten most of the aggregation benefit, and the
  honest conclusion is "PIM helps only in the extreme-aggregation corner (reduction), not on realistic
  pooling."
- **Regime honesty (G3):** the crossover requirement is what separates a genuine result from picking an
  aggregating kernel and declaring victory — the whole point of C1's mitigation.

---

## 6. Canonical config + the pre-committed sweep (guards against cherry-picking)

### Canonical operating point (decision-driving) `[believed — from DLRM literature]`
Embedding-bag, **`d = 64`** (typical DLRM embedding dim), **`L = 40`** (a representative pooling factor;
real DLRM pooling factors span ~1 to hundreds across features), **`B = 16`** banks, **`nb = 1024`**
bags, **`R_tab = 4,000,000`** rows, random indices (fixed seed), round-robin placement, `b = idx_b = 4`
bytes, shared `cap = 32` bytes/cycle, `e = 160` pJ/byte. `random_state = 42`.
> Chosen from published DLRM/MLPerf shapes, **not** tuned to a target DMR. The likely honest outcome may
> sit near the GO/gray boundary — that is expected and is itself evidence the design is not rigged.

### Pre-committed sweep (report DMR + guardrails at every point; decision on canonical)
| Axis | Values | What it probes |
|---|---|---|
| **`L` rows/bag (aggregation factor)** | {1, 2, 4, 8, 16, 32, **40**, 64, 128, 256} | The crossover `L*`; the reduction-ratio axis (§4); R-tautology |
| **`B` banks** | {4, 8, **16**, 32, 64, 128} | R-banking-tax: byte-win vs parallelism tension |
| **`d` embedding dim** | {16, 32, **64**, 128} | R-hidden-traffic: index/output amortisation |
| **`idx_b` index width** | {**4**, 8} | R-hidden-traffic: index-traffic sensitivity |
| **placement** | {**round-robin**, clustered/locality-aware} | Does placement move the crossover? (best/worst `k`) |
| **`e` energy/byte (pJ)** | {50, **160**, 640} | Secondary energy sensitivity (C3) |
| **kernel** | {reduction (endpoint), **embedding-bag** (headline), SpMV (optional)} | Sweep endpoints + honesty middle |

Bold = canonical. **The decision is made at the canonical point; the sweep characterises robustness and
locates the crossover. No threshold moves after data is seen.** *(anti-forking-paths — Gelman & Loken.)*

---

## 7. Threats to validity & how each is handled (Shadish/Cook/Campbell; Feynman)

| # | Threat | Handling |
|---|---|---|
| T1 | **Modelled ≠ measured** — analytical, not silicon | Bytes tagged `[modelled-exact]` (counted, not estimated); cycles/energy `[modelled]`, secondary; GO only *licenses a write-up*, claims no silicon speedup (§0) |
| T2 | **Baseline strawman** (R-baseline-strawman) | Baseline is the coalesced **floor** — moves each byte once (§2.3/G2); bytes (not cycles) as primary is itself the strongest anti-strawman choice |
| T3 | **Undercounting PIM link traffic** (R-hidden-traffic) | Full accounting charges indices + **`k` partials** + output to PIM (§3/G1); overhead fraction reported; `d`↓ and `idx_b`↑ swept |
| T4 | **Tautology — "aggregation trivially sends less"** (R-tautology, C1) | Reduction-ratio sweep + naive-vs-honest DMR lines + mandatory crossover (§4/G3); reduction (the tautological corner) is explicitly *not* the decision kernel |
| T5 | **Energy-assumption dependence** (C3) | Bytes are the assumption-free primary; energy is secondary, `e` swept, and framed as an *upper bound* on PIM's advantage (in-bank energy not credited as free) |
| T6 | **DRAM model over-simplification** (C2) | Deliberately parked: one shared bandwidth cap, no refresh/row-buffers; win rests only on countable link bytes, not on timing fidelity |
| T7 | **Cherry-picked kernel/params** | Canonical from DLRM literature (not tuned); full sweep; DMR must hold across the realistic range, and the crossover must place real params in the winning region |
| T8 | **Banking tension hidden** (R-banking-tax) | `B` sweep + explicit report that small `B` inflates byte-DMR but cuts parallelism; decision at realistic `B` |
| T9 | **Placement rigging** | Both round-robin (neutral) and clustered (favourable) placements swept; crossover reported under each; canonical uses neutral round-robin |
| T10 | **Multiplicity / forking paths** | Decision pre-committed to the canonical point; sweep is descriptive; any extra analysis labelled exploratory in `pim-exploratory.md` |
| T11 | **Non-reproducibility** | Fixed seeds, pinned versions, data/param hashes, one-command run, `provenance.json` (§8) |

---

## 8. Reproducibility & provenance (an unreproducible result is not evidence)

- **One command:** `python pim/run.py --config pim/config.yaml` (build/invoke the C++ core → count
  bytes for baseline & PIM across kernels + sweeps → compute DMR + energy → plots → results →
  GO/NO-GO summary inputs).
- **C++/Python split (D-016):** the byte/bandwidth/bank model + counters are the **C++ core** (CMake +
  Ninja + MSVC, $0); the sweep driver, DMR/energy analysis, and matplotlib plots are the **Python
  layer** (reusing the spike's rig).
- **Fixed seeds/params:** `random_state = 42`; all §6 params logged; index draws and placement are
  seeded and reproducible.
- **Provenance logged** to `pim/out/provenance.json`: compiler + Python/numpy/matplotlib versions, seed,
  full parameter dict, the pJ/byte constant + its citation, timestamp, and (if under git) commit hash.
- **`[modelled]` / `[modelled-exact]` tags** on every reported quantity per §0.
- **Storage:** experiment code + outputs under `C:\Code\Self\projects\grove\pim\`; a pointer entry is
  recorded in the experimentAS index. **coderAS builds on a dedicated local branch `exp/d017-pim`**
  (do not pollute the working line; merge decision deferred to CONCLUDE). Per the experiment-branch
  discipline, all Arm C sub-experiments and runs share this one branch (the "PIM" research thread).

---

## 9. Deliverables coderAS must return (so experimentAS can render GO/NO-GO mechanically)

1. **`pim/out/results.csv`** — for every sweep point: kernel, `L, B, d, idx_b, nb`, placement,
   baseline link bytes, PIM link bytes, `k` (empirical + closed-form), **DMR**, per-machine
   index/operand/partial/output byte breakdown, PIM overhead fraction, modelled link cycles under
   `cap`, and energy at each `e`.
2. **Plots:** (a) **DMR vs aggregation factor `1/RR`** per `B`, with the **naive tautological line**
   overlaid and the **crossover `L*`** + DLRM operating point marked (the headline honesty plot);
   (b) DMR vs `B` (banking tension); (c) DMR vs `d` (hidden-traffic amortisation); (d) stacked
   link-byte breakdown (indices/operands/partials/output) baseline vs PIM at canonical; (e) energy-DMR
   sensitivity to `e`.
3. **`pim/out/decision_inputs.json`** — the GO-gate quantities: canonical DMR; DMR-sweep-min across the
   realistic range; crossover `L*` per `B`; whether the DLRM operating point is right of `L*`; PIM
   overhead fraction at canonical; energy-DMR range over `e`.
4. **`pim/out/provenance.json`** (§8).

**Definition of done:** experimentAS reads those artifacts and applies §5's rule mechanically to
produce a **GO / NO-GO / gray-with-diagnosis** verdict with a calibrated-confidence statement, then
appends the outcome ADR and updates the risk ledger (C1/C4).

---

## 10. Approval gate

This is the pre-registration. **coderAS is unblocked to implement Arm C to this spec** (build order in
the handoff: shared bandwidth-capped bank model → reduction de-risk slice → embedding-bag headline →
the reduction-ratio sweep). No thresholds or accounting rules may change after data is generated; if a
definition proves unimplementable, coderAS returns here to **amend the pre-registration with a
timestamp** *before* producing numbers (as Amendment A1 did for the spike).

---

<!-- Amendments (pre-data only) are appended below this line, each with a timestamp and anchor ADR. -->
