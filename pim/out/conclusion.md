# Lithos Arm C (PIM) — CONCLUSION: **CONDITIONAL-GO** (data-movement win is real but banking-limited; scoped to high-pooling features)

- **Experiment:** Arm C (D-017/D-018) — "does near-memory / PIM cut off-chip bytes (DMR = baseline ÷ PIM link bytes) *substantially and robustly* on DLRM embedding-bag sum-pooling, under a fair shared bandwidth cap, with an honest crossover?"
- **Pre-registration (frozen from data-generation):** `pim-prereg.md` §0–§8 (§2 metric/guardrails, §4 crossover, §5 GO/NO-GO rule, §6 canonical+sweep, §7 threats)
- **Anchor ADRs:** D-017 (Arm C scope), D-018 (pre-registration), **D-019 (this outcome)**
- **Backend / model:** analytical bank + shared-bandwidth-capped byte-accounting model (`pim/src/model.cpp`, C++17; Python analysis layer). Primary metric **bytes = `[modelled-exact]`** (counted, not estimated); cycles/energy = `[modelled]` (secondary).
- **Kernel / canonical config:** embedding-bag sum-pooling, `d=64, L=40, B=16, nb=1024, R_tab=4,000,000`, random indices (seed 42), round-robin placement, coalesced baseline, `b=idx_b=4`, `cap=32` B/cyc, `e=160` pJ/B.
- **Decision kernel:** embedding-bag (headline). Reduction (`DMR≈N/B≈941,177×`) is the deliberate tautological endpoint — a plumbing/byte-counter smoke, **not** the decision kernel.
- **Provenance:** `pim/out/provenance.json` — numpy 2.4.6, matplotlib 3.10.9, py 3.12.10; seed 42; energy cite Horowitz ISSCC 2014 (~160 pJ/byte). Run recorded at git `33419f9` (prereg commit; build + outputs committed together at `e33b365`).
- **Date concluded:** 2026-07-06T18:54+05:30 · **Owner:** experimentAS · **Branch:** `exp/d017-pim` (not merged)

---

## 1. Verdict: **CONDITIONAL-GO**, calibrated confidence **~0.82**

The frozen §5 rule, applied mechanically:

| §5 rail | Rule | Result | Status |
|---|---|---|---|
| **Primary (canonical DMR)** | GO needs ≥ 3× | **2.53×** | GO-primary **fails** |
| **Primary (canonical DMR)** | NO-GO if < 1.5× | **2.53×** | NO-GO-primary **not triggered** |
| **Robustness** | GO needs ≥ 1.5× across realistic sweep | sweep-min **1.20×** (at L=8) | GO-robustness **fails** |
| **Regime honesty (G3)** | crossover exists; DLRM point right of `L*` with margin | `L*(B=16)=32`; DLRM L=40 just right of it (**thin margin**) | holds but **thin** |
| **Accounting integrity (G1/G2)** | overhead reported, win survives; baseline = coalesced floor | overhead **0.099**; baseline verified as floor | **passes** |

Canonical DMR = **2.53× ∈ [1.5×, 3×)** → this is squarely the **§5 GRAY ZONE**. coderAS's mechanical
echo (`GRAY-ZONE`) is confirmed. Per the §5 gray-zone protocol I diagnose the binding limiter, score
the rivals, and render a judgement call: **conditional-GO, scoped to the high-pooling regime where the
win clears the bar** — *not* an unconditional GO (canonical 2.53× < 3×), and *not* a NO-GO (2.53× > 1.5×,
three of four rivals killed, and a clean ≥3× regime exists).

> **What this conditional-GO licenses (and what it does not).** It licenses writing up Arm C as a
> *use-case-driven data-movement result*, **with its scope stated on the tin**: near-memory sum-pooling
> delivers a substantial, banking-limited off-chip-byte reduction **for high-pooling embedding features
> (L ≳ 64 at realistic B=16, DMR ≥ 3× rising to >12×)**, and honestly crosses *below* the worth-it line
> (1.5×) for low-pooling features (L ≲ 16). It does **NOT** claim a silicon speedup, does **NOT** claim a
> flat win across all DLRM features, and does **NOT** claim victory over a real HBM-PIM product.
> Everything decision-driving is `[modelled-exact]` (bytes) or `[modelled]` (energy/cycles).

## 2. Binding-limiter diagnosis: **the banking factor `k` (R-banking-tax)**

The honest DMR is `DMR ≈ L / k(L,B)`, moderated down by index+output traffic, where `k(L,B) =
B·(1−(1−1/B)^L)` is the number of *distinct banks* a bag's L rows touch — i.e. the partials PIM must
still ship across the link. At the canonical point `k(40,16) = 14.84` (of a max 16): **nearly every bank
is touched**, so PIM ships ~B partials per bag and the raw ratio is `L/k = 40/14.84 = 2.70`, pulled to
**2.53×** by the (small) hidden index+output traffic.

The **B-sweep at fixed L=40 is the smoking gun** — the win is bounded by banking, not by anything hidden:

| B (banks) | 4 | 8 | **16** | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| k_empirical | 4.00 | 7.96 | **14.84** | 23.1 | 30.0 | 34.6 |
| **DMR** | 7.40× | 4.34× | **2.53×** | 1.68× | 1.32× | 1.15× |

As `B` grows toward and past `L`, `k → min(L,B)` and DMR collapses toward 1. Small `B` (4, 8) clears the
3× bar — but small `B` is exactly the parallelism-sacrificing corner R-banking-tax warned about, so the
decision correctly stays at the realistic `B=16`. **The limiter is structural: PIM can only pre-sum rows
that share a bank, and with random DLRM indices over a 4M-row table, L=40 rows scatter across ~all 16
banks.** Two secondary checks confirm the limiter is *not* elsewhere:

- **Hidden traffic (R-hidden-traffic) is small and charged.** Overhead fraction (idx+output share) =
  **0.099** at canonical; DMR rises only mildly with d (2.37× at d=16 → 2.56× at d=128) as indices
  amortise. Not the binding constraint at d=64.
- **Placement cannot rescue it.** Clustered/locality-aware placement gives **2.540×** vs round-robin
  **2.529×** (k 14.76 vs 14.84) — a rounding-error difference. With scattered random indices, locality
  engineering does *not* lower `k`. This kills the "just place data better" escape hatch honestly.

**The crossover sits close to the operating point.** `L*(B=16) = 32` (smallest L with DMR ≥ 1.5×); the
DLRM canonical L=40 sits just right of it, but the common **low-pooling tail (L=8→1.20×, L=16→1.49×)
falls left of the worth-it line.** That closeness is why the realistic-range sweep-min (1.20×) fails the
GO robustness rail — the win is genuine for high-pooling features and genuinely absent for low-pooling
ones.

## 3. Rival-hypothesis scorecard (Platt strong inference)

| Rival | Verdict | Evidence |
|---|---|---|
| **R-tautology** ("PIM wins only because aggregation trivially sends less") | **KILLED** | Honest DMR sits *far below* the naive `DMR=1/RR=L` line (2.53× vs 40 at L=40; 12.4× vs 256 at L=256) and a real crossover exists: **L=1 → DMR = 1.000 exactly** (pure gather, PIM = baseline). The win is bounded by `L/k`, not the definitional `L`. See `dmr_crossover.png`. **This retires C1.** |
| **R-hidden-traffic** ("indices + k partials + output erase the win") | **KILLED (charged, survives)** | Every crossing charged to *both* sides; PIM = idx + `k·d·b` partials + output. Overhead fraction reported = **0.099** at canonical; win survives. Grows at small d (0.19 at d=16) but never dominates. |
| **R-baseline-strawman** ("the GPU baseline was crippled") | **KILLED** | Baseline byte count = `L·d·b·nb` gathered once + indices + output = the **coalesced floor** (each row crosses exactly once). Re-derived from the CSV breakdown. Using *bytes* (not coalescing-sensitive cycles) as primary is itself the anti-strawman choice. |
| **R-banking-tax** ("more banks → more partials → smaller byte-win") | **NOT killed — CONFIRMED as the binding limiter** | The B-sweep above: DMR falls 7.40× → 1.15× as B goes 4 → 128. This is the honest, un-killed rival — and characterising it *is* the result, not a failure. |

**Net: 3 of 4 rivals killed; R-banking-tax stands as the honest, characterised limiter.** H1 (≥3×
canonical **and** ≥1.5× robust **and** crossover outside the realistic regime) is **not** fully
supported — canonical is 2.53× and low-pooling features fall below 1.5×. But H0 (no useful/robust win,
or a pure tautology) is **also refuted** — the win is real, bounded, non-tautological, and clears 3×
for high-pooling features. The truth is the nuanced middle the gray zone was designed to surface.

## 4. Energy (secondary, `[modelled]`, C3-honest)

Energy DMR is **e-invariant** (= byte DMR = 2.53× at canonical) because `e` cancels in the ratio — so it
is robust to the pJ/byte assumption across the swept `e ∈ {50,160,640}`. Per §2.4 it is framed as an
**upper bound** on PIM's true energy advantage (we count only link bytes and do not credit in-bank
movement as free). No energy over-claim. See `energy_dmr.png`.

## 5. Calibrated confidence: **~0.82** — and the one observation that would move it

The gray-zone *classification* is certain (mechanical, `[modelled-exact]` bytes, all re-derivations
pass — §7). The **~0.82** attaches to the *conditional-GO judgement* — that the win is real,
banking-limited, and worth writing up scoped to high-pooling features. Residual uncertainty (why not
higher):

- **Workload plausibility is `[believed]`, not measured.** The conditional-GO's value rests on real
  target DLRM workloads actually *having* a meaningful mass of high-pooling (L ≳ 32–64) features. Long
  user-behaviour / click-history embeddings do reach those pooling factors, but many categorical
  features are low-pooling (L ≲ 16), which sit in the losing region.
- **[modelled], not silicon.** Second-order DRAM effects (bank conflicts, refresh, row buffers, in-bank
  compute latency) are parked (C2). A GO licenses a *data-movement* write-up only.

> **The single observation that would change the call:** the **pooling-factor distribution of the target
> DLRM workload.** If a meaningful fraction of features sit right of the crossover (L ≳ 32), conditional-GO
> strengthens toward an unconditional GO. If the mass is low-pooling (L ≲ 16, left of `L*=32`), the useful
> regime becomes an implausible corner → this tips to **NO-GO / weak** per §5 NO-GO condition 2. A locality
> structure in the indices (correlated co-access) would also help — but *only* if it lets clustered
> placement lower `k`, which random indices do not.

## 6. Decision & next step

**Conditional-GO:** write up Arm C as a **use-case-driven, honestly-scoped data-movement result** — *"near-memory
sum-pooling gives a substantial off-chip-byte reduction (≥3×, rising to >12×) for high-pooling embedding
features, is fundamentally banking-limited (DMR ≈ L/k, k → B), and crosses below the worth-it line for
low-pooling features — with the crossover L*(B) fully characterised."* That nuance is a stronger,
more honest contribution than a bare hero number.

- **Before over-committing the narrative:** pin the target workload's pooling-factor distribution (the
  §5 observation above). Cheapest test: cite published DLRM/MLPerf per-feature pooling shapes and mark
  where their mass falls relative to `L*(B)`. That converts the `[believed]` workload-plausibility into
  `[verified]` and decides conditional-GO vs NO-GO/weak definitively.
- **Branch:** all Arm C work is on `exp/d017-pim` (not merged). The merge-to-main decision is the
  owner's/coderAS's, deferred to *after* this verdict.

## 7. Integrity rails honored (Twyman's law checks done before deciding)

Metric, thresholds, baseline, sweep, and analysis plan were frozen in `pim-prereg.md` **before** any
number was generated; **no threshold moved after data**. This verdict applies the frozen §5 rule
mechanically, then adds calibrated confidence and the scope judgement. Verification re-derivations
(all pass):

- Canonical DMR = **10,911,744 / 4,315,136 = 2.52871** re-derived from the byte breakdown. ✓
- Baseline = idx (163,840) + operand (`L·d·b·nb`=10,485,760, each row once = coalesced floor) + output (262,144). ✓ (G2)
- PIM = idx (163,840) + `k·d·b·nb` partials (3,889,152) + output (262,144); all link traffic charged. ✓ (G1)
- `k_empirical` 14.836 ≈ closed-form `B(1−(1−1/B)^L)` = **14.789** (empirical *slightly higher* → conservative, does not inflate DMR). ✓
- DMR (2.53) < `L/k` (2.70) — confirms hidden idx+output traffic *is* charged and pulls the ratio down. ✓
- Overhead fraction (idx+output share) = **0.0987** re-derived. ✓
- Crossover anchor: L=1 → DMR = **1.000** exactly. ✓  Reduction endpoint = (N+1)/(B+1) = **941,177** ≈ N/B. ✓

**Minor provenance note (non-blocking):** `provenance.json` records git `33419f9` (the prereg commit —
HEAD at run time), while the model source + outputs were committed together at `e33b365`. The audited
source (`model.cpp`, 8/8 tests pass) matches the committed build; no accounting inconsistency found.

Any analysis beyond the frozen plan is exploratory and belongs in `pim-exploratory.md`, clearly labelled.
A nuanced, cheaply-obtained result that characterises *exactly where PIM wins and where it does not* —
that is honest science, and the gray zone did its job.

## 8. Plots (in `pim/out/`)

- `dmr_crossover.png` — honest DMR vs L per B, with the naive tautology line `DMR=L` and the crossover `L*` (the anti-tautology evidence, C1).
- `dmr_vs_banks.png` — the R-banking-tax curve (DMR vs B at fixed L): the binding limiter, visualised.
- `dmr_vs_dim.png` — DMR vs d (R-hidden-traffic amortisation).
- `byte_breakdown.png` — baseline vs PIM link-byte composition (idx / operand-or-partial / output).
- `energy_dmr.png` — energy DMR vs e (e-invariant; C3-honest).
