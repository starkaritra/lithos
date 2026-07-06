# 06 — The Crossover & Honest Science

> **Goal:** see how Arm C avoids fooling itself. You'll learn the **reduction-ratio sweep** and the
> **crossover** (where PIM stops being worth it), why **pre-registration** matters, and how these
> produced the honest final verdict: **CONDITIONAL-GO** — a real but banking-limited win.

---

## 1. The problem: how do you *not* rig your own experiment?

By chapter 05 we could compute an honest DMR. But there's a subtler trap — the **tautology**. If you
only ever test kernels that aggregate hard (like array reduction, DMR ≈ 941,177×), you'll always
"prove" PIM wins, because you *chose* kernels where it must. That's not science; it's a definition
dressed as a result.

The fix is to test the *whole spectrum* — from "aggregates everything" to "aggregates nothing" — and
report **exactly where the win appears and where it disappears.** That boundary is the honest deliverable.

---

## 2. The sweep axis: how much does the kernel aggregate?

We sweep the **aggregation factor** (for embedding-bag, this is just the pooling factor `L`), from no
aggregation to extreme aggregation:

```
   aggregation:  none ─────────────────────────────────────────► extreme
   L = 1         L = 40 (canonical)                    ...        full reduction (N→1)
   pure gather   embedding-bag pooling                            array sum
   DMR → 1       DMR = 2.53×                                       DMR ≈ N/B (huge)
   PIM useless   PIM helps, banking-limited                        PIM "wins" trivially
```

- **Left end (`L=1`, pure gather):** every row in its own bank, `k ≈ 1`, so PIM ships one partial per
  row — no saving. `DMR → 1`. PIM is useless here, and *the model correctly says so.*
- **Right end (reduction):** everything collapses to `B` partials. `DMR` is astronomical — the
  **tautological corner**. We include it only as a plumbing check and endpoint, **never** as the decision.
- **The middle (embedding-bag):** the real, interesting regime, where `k` (chapter 05) sets the win.

---

## 3. The crossover `L*` — the honesty deliverable

The **crossover** `L*` is the smallest pooling factor at which PIM becomes "worth it" (we pre-defined
"worth it" as DMR ≥ 1.5×). Below `L*`, PIM doesn't help; above it, it does. Measuring `L*` — and placing
the *real* workload's operating point relative to it — is what makes the result honest.

Our measured DMR vs `L` at `B = 16`:
```
   L:     1     4      8      16     32     40     64     128    256
   DMR:  1.00  1.08   1.20   1.49   2.16   2.53   3.72   6.90   12.43
                              └── crossover L* = 32 ──┘  └─ clears 3× "GO" bar ─┘
   naive DMR = L:  1     4      8      16     32     40     64     128    256   ← far above
```

Two things leap out:
1. The honest curve sits **far below** the naive `DMR = L` line (the chapter-05 banking factor at work)
   — the visible gap *is* the anti-tautology evidence.
2. There's a real **crossover at L* = 32**: below it PIM isn't worth it; the canonical DLRM point
   (`L = 40`) sits *just* right of it (a thin margin), and the win only becomes strong (≥3×) for
   high-pooling features (`L ≳ 64`).

This is the headline plot: `pim/out/dmr_crossover.png`.

---

## 4. Pre-registration: deciding before you see the data

The other guard against self-deception is **pre-registration**: we wrote down the hypotheses, the
primary metric (DMR), the exact byte accounting, and the **GO / NO-GO decision rule** *before*
generating a single number (`../../pim-prereg.md`, decision D-018). After data, **no threshold moved.**
This is the same discipline that let the earlier spike honestly kill its own thesis
(`../../spike/out/conclusion.md`). It's why the result is trustworthy rather than a story fit to the
numbers after the fact.

The pre-registered rule (paraphrased):
- **GO** if canonical DMR ≥ 3× *and* ≥ 1.5× across the realistic range *and* the crossover is honest.
- **NO-GO** if DMR < 1.5× once all traffic is counted, or the winning regime is an implausible corner.
- **Gray zone** in between → diagnose the binding limiter and decide with stated confidence.
- **A NO-GO was explicitly allowed to be the answer.**

---

## 5. The verdict: CONDITIONAL-GO (confidence ~0.82)

The canonical DMR came out at **2.53×** — in the gray zone (below the 3× GO bar, above the 1.5× NO-GO
bar), and the realistic-range minimum (1.20× at low pooling) failed the robustness bar. Applying the
gray-zone protocol, the project (decision **D-019**) resolved it to:

> **CONDITIONAL-GO** — PIM's data-movement win is **real and non-tautological, but banking-limited**. It
> pays off for **high-pooling** embedding features (`L ≳ 64` → DMR ≥ 3×, up to 12×) and is **absent**
> (<1.5×) for **low-pooling** ones (`L ≲ 16`). The binding limiter is the banking factor `k`
> (chapter 05); more banks only make it worse (the banking tax).

The rival-hypothesis scorecard (Platt's strong inference):

| Rival | Verdict | Why |
|---|---|---|
| "It's a tautology" | **KILLED** | Honest DMR ≪ naive `L`; a real crossover exists; `L=1 → DMR = 1.000` |
| "Hidden traffic erases it" | **KILLED** | Indices + `k` partials + output all charged; overhead only 0.099 |
| "The baseline is a strawman" | **KILLED** | Baseline is the coalesced byte floor |
| "It's banking-limited" | **CONFIRMED** | The `B`-sweep: 7.4× (B=4) → 1.15× (B=128) — *this is the result* |

**What would change the call:** the *actual* pooling-factor distribution of a target recsys workload. If
many features pool heavily (right of `L*`), the story strengthens toward unconditional-GO; if most are
low-pooling, it tips to NO-GO/weak. That's the honest next measurement.

---

## 6. Why a conditional result is a *good* result

It would have been easy to report "PIM gives a 40× data-movement win!" (naive) or "941,177×!"
(reduction). Both are technically true and completely misleading. Instead, Arm C says: *here is exactly
the regime where near-memory pooling helps, here is where it doesn't, and here is the single number
(`k`) that decides it.* That nuance is more useful — and more credible — than any hero number, and it's
the kind of result that survives a skeptical reviewer.

> **Reality check.** This is a **data-movement** result `[modelled-exact]`, not a silicon speedup. It
> says PIM moves fewer bytes in a specific regime; translating that to real end-to-end performance would
> need the timing/energy fidelity we deliberately didn't model (chapter 03). The honest claim is scoped
> to what we measured.

---

## Check your understanding
1. What is the tautology trap, and how does sweeping the aggregation factor + reporting the crossover
   defeat it?
2. Why does pre-registering the decision rule (before data) make the verdict trustworthy?
3. Restate the CONDITIONAL-GO verdict in one sentence, and name the single quantity that would change it.

---

## References
- B. Nosek et al., "The preregistration revolution," *PNAS*, 2018; J. Platt, "Strong Inference,"
  *Science*, 1964.
- A. Gelman & E. Loken, "The garden of forking paths," 2013 (why fixing analyses before data matters).
- Lithos: `../../pim-prereg.md` (§4 sweep, §5 decision rule), `../out/conclusion.md` (the verdict),
  `../../decisions.md` D-018/D-019.

→ Previous: [05 — The Banking Factor `k`](05-the-banking-factor-k.md) · Back to [index](README.md)
