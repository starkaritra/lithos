# 04 — Byte-Accounting & the DMR

> **Goal:** understand the metric that decides everything in Arm C — **DMR**, the ratio of off-chip
> bytes the baseline moves to the bytes PIM moves — and why we *count* bytes instead of modelling
> cycles or energy. You'll see the full, auditable accounting for the headline embedding-bag kernel.

---

## 1. The problem: what should we even measure?

We could measure cycles, or energy, or a "speedup." But each of those depends on assumptions we'd have
to defend — clock speeds, pipeline details, pJ-per-operation constants. The one thing PIM
fundamentally changes, and the one thing we can count *without any assumption*, is:

> **How many bytes cross the off-chip link?**

That is the memory wall in its purest form (chapter 01), and it's the quantity PIM exists to shrink.
So we make it the primary metric.

---

## 2. The metric: DMR (Data-Movement Reduction)

> **DMR = off-chip link bytes (baseline GPU) ÷ off-chip link bytes (PIM)**, both running the *same*
> kernel on the *same* data under the *same* bandwidth cap.

`DMR = 10` reads as "**10× fewer off-chip bytes**." It's a clean effect size, it's scale-free (absolute
counts cancel), and — the key move — the bytes are **counted, not modelled**. We tag this
`[modelled-exact]`: the model is an abstraction, but within it the byte counts are exact arithmetic, not
estimates. That defuses any "your energy constant is wrong" objection: bytes don't need a constant.

Energy is reported too, but only as a **secondary** metric: `Energy = bytes × e` (pJ/byte). Since the
DMR ratio cancels `e`, the energy DMR *equals* the byte DMR — so energy adds no new information and no
new assumption. We report it for completeness and flag that it's an *upper bound* on PIM's true energy
edge (we don't even credit PIM's cheaper in-bank movement). Bytes stay the decision-driver.

---

## 3. The rule that keeps it honest: count ALL traffic, on BOTH sides

The fastest way to lie with PIM is to forget the traffic it *doesn't* avoid. So the accounting charges,
to **both** machines, every category of byte that crosses the link:

```
   for each machine, link bytes =  indices in  +  operands/partials  +  result out
```

If PIM's "win" came from ignoring the indices it must still fetch, or the output it must still write,
that would be a defect, not a result. Our `Accounting` struct (`../include/pim/model.hpp`) tracks each
category separately so the breakdown is auditable, and reports a **PIM overhead fraction** = the share
of PIM's link traffic that is *not* the pre-aggregated partials (i.e. indices + output).

---

## 4. The headline kernel: embedding-bag sum-pooling

This is the recommendation-system workload (DLRM). For each **bag**: gather `L` scattered rows (each a
`d`-element vector) from a huge table and **sum** them into one `d`-vector.

```
   bag: pick L=40 scattered rows from a 4,000,000-row table, sum -> one d=64 vector

   GPU baseline:  fetch ALL L rows across the link, then sum on-chip
   PIM:           each bank sums the rows IT holds, ships ONE d-vector partial per bank touched
```

**Per-bag byte accounting (charged to both — this is the whole ballgame):**

| Traffic | Baseline GPU | PIM |
|---|---|---|
| Indices in | `L · idx_b` | `L · idx_b` *(same)* |
| Operands over link | `L · d · b` (all `L` rows) | `k · d · b` (one partial per **touched bank**) |
| Result out | `d · b` | `d · b` *(same)* |
| **Per-bag total** | **`L·idx_b + L·d·b + d·b`** | **`L·idx_b + k·d·b + d·b`** |

The only difference is the operand row: the baseline moves **`L`** row-vectors; PIM moves **`k`** of
them, where `k` = the number of distinct banks the bag's `L` rows land in. That single symbol `k` is
the entire story, and chapter 05 is devoted to it.

- **Array reduction** (the other kernel) is the extreme case: baseline moves all `N` elements, PIM moves
  `B` partials → `DMR = N/B`. It's a plumbing check and the far endpoint of the sweep, **not** the
  decision kernel (its win is trivially huge — the tautological corner).

---

## 5. In our model: where the counting lives

`account_embedding()` in `../src/model.cpp` does exactly the table above. The subtlety is that `k`
**varies per bag** (some bags' rows happen to cluster in fewer banks), so we can't just multiply — we
**sample `nb` bags of `L` seeded random indices, place each, and count the distinct banks per bag**,
summing the real `k` values. That's why the model runs actual (seeded, reproducible) sampling rather
than only a formula — the empirical `k` is the honest number, and we cross-check it against the
closed-form expectation (chapter 05).

---

## 6. Measure it yourself

```
build\pim.exe --json                    # canonical embedding-bag accounting
```
You'll see (canonical `d=64, L=40, B=16, nb=1024`):
```
  baseline_bytes = 10,911,744   pim_bytes = 4,315,136   dmr = 2.529
  base_operand   = 10,485,760   pim_partial = 3,889,152   (the L-vs-k difference)
  k_empirical    = 14.84        pim_overhead_fraction = 0.099
```
The baseline moves ~10.9 MB; PIM moves ~4.3 MB → **2.53× fewer bytes**. Note `pim_partial` (3.9 MB) is
much smaller than `base_operand` (10.5 MB) — that gap *is* the win, and it's set by `k ≈ 14.84`, not by
the naive `L = 40`. Onward to why.

> **Try it:** run `build\pim.exe --kernel reduction --json` and see `dmr ≈ 941,177` (~N/B). Then
> `build\pim.exe --L 1 --json` and see `dmr = 1.000`. Those are the two endpoints; the interesting
> physics is between them.

---

## Check your understanding
1. Why is "bytes moved" a better primary metric here than cycles or energy?
2. In the per-bag table, which single row differs between baseline and PIM, and by what factor?
3. Why does the model *sample* bags instead of just multiplying by a formula?

---

## References
- S. Williams et al., "Roofline," *CACM*, 2009 (data movement as the performance axis).
- M. Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems
  (DLRM)," arXiv:1906.00091, 2019 (the embedding-bag workload).
- Grove: `../../pim-prereg.md` §2 (DMR + guardrails), §3 (the accounting), `../src/model.cpp`.

→ Previous: [03 — The Bank Model & a Fair Fight](03-the-bank-model-and-fairness.md) · Next: [05 — The Banking Factor `k`](05-the-banking-factor-k.md)
