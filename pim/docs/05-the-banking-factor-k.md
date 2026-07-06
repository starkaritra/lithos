# 05 — The Banking Factor `k`

> **Goal:** understand the single most important quantity in Arm C — the **banking factor `k`** — and
> why it turns a naive "40× win" into an honest "2.5× win." If you understand `k`, you understand the
> entire result. This is the crux chapter.

---

## 1. The problem: the naive win is a lie

Chapter 04 showed the win comes from PIM shipping `k` partials instead of `L` rows. The *naive*
intuition — the one a careless analysis would use — says: "a bag pools `L` rows into 1 result, so PIM
sends **1** thing instead of `L`; the data-movement win is `L`." At the canonical `L = 40`, that would
claim a **40×** reduction.

That is wrong, and seeing *why* is the whole point of this arm.

---

## 2. Intuition: how many drawers do 40 socks land in?

You have a dresser with **`B = 16` drawers** and you throw in **`L = 40`** socks at random. To collect
your socks, you must open every drawer that has at least one — how many is that?

Not 1 (the naive answer, "they're all my socks"). And not 40 (there are only 16 drawers). With 40 socks
in 16 drawers, **almost every drawer ends up with at least one** — you'll open ~15 of the 16.

That count — the number of *distinct drawers touched* — is the **banking factor `k`**. In PIM terms:
each **bank** that holds *any* of a bag's rows must ship a partial. So PIM sends **`k` partials per
bag**, not 1. The rows collapse *within* a bank (that's the saving), but every *touched* bank still
costs a full row-sized message across the link.

```
   L = 40 rows, B = 16 banks, round-robin placement:

   bank:   0   1   2   3   4  ...  15
   rows:   ●●  ●   ●●● ●   ●●      ●●    ← ~every bank gets ≥1 row
           └── each touched bank ships ONE d-vector partial ──┘
   k ≈ 15  (NOT 1, and NOT 40)
```

---

## 3. The mechanism: `k(L, B)` is coupon-collector occupancy

The math is a classic occupancy problem. With `L` rows placed uniformly at random into `B` banks, the
probability a given bank gets *no* row is `(1 − 1/B)^L`, so the expected number of **non-empty** banks is

```
        k(L, B) = B · ( 1 − (1 − 1/B)^L )
```

Sanity-check the extremes (both matter):
- **`L = 1`** → `k = 1`. One row touches one bank → PIM ships 1 partial for 1 row → **no saving,
  DMR → 1.** (This is pure gather; PIM can't help — the crossover.)
- **`L ≫ B`** → `k → B`. Many rows, every bank touched → PIM ships `B` partials → the saving saturates
  at **`DMR → L/B`.**
- **`L = 40, B = 16`** → `k = 16·(1 − (15/16)^40) ≈ 14.79`. So PIM sends ~14.8 partials, not 1.

Now substitute back into the DMR from chapter 04 (for large `d`, indices/output are small):

```
        DMR  ≈  L / k(L, B)      ← the HONEST win

   naive:  L = 40         → "40×"
   honest: 40 / 14.8      → ~2.7× (and 2.53× once indices+output are also charged)
```

**That factor-of-15 gap between the naive `L` and the honest `L/k` is the result of Arm C.** The win is
real, but banking eats most of the naively-imagined benefit.

---

## 4. In our model: empirical `k`, cross-checked

We don't just trust the formula — `account_embedding()` (`../src/model.cpp`) **measures** `k` by
sampling `nb` seeded bags, placing each row via `place(i)`, and counting distinct banks per bag:

```cpp
for each bag:                      // nb = 1024 seeded bags
    banks.clear();
    for l in L:  banks.insert(place(random_row(), B, placement, R_tab));
    total_k += banks.size();       // this bag's distinct-bank count
k_empirical = total_k / nb;        // averaged; used for the actual byte totals
```

and cross-checks against `closed_form_k(L,B) = B·(1−(1−1/B)^L)`. At canonical: **`k_empirical = 14.84`
vs closed-form `14.79`** — a tight match (empirical slightly higher, which is *conservative*: it makes
PIM look slightly worse, never better). The test `test_k_empirical_matches_closed` pins this.

---

## 5. Why this makes the result honest (not a tautology)

A tautological result would track the naive `DMR = L` line — "I picked an aggregating kernel, of course
it wins." Our honest curve sits **far below** that line precisely because of `k`. And it produces a real
**crossover**: at low `L`, `k ≈ L`, so `DMR ≈ 1` and **PIM doesn't help**. A metric that can come out
saying "no win here" is a metric that isn't rigged — which is exactly what chapter 06 exploits.

The banking factor also creates a genuine **tension** you can measure: more banks `B` means more
parallelism (good for compute) but a *larger* `k` for a given `L` (more partials → *smaller* byte win).
So you can't just crank `B`. Run the sweep and watch it:

```
  DMR at L=40:   B=4 → 7.40×    B=8 → 4.34×    B=16 → 2.53×    B=64 → 1.32×    B=128 → 1.15×
```

More banks → smaller data-movement win. That's the **banking tax**, and it's the binding limiter on
Arm C's result.

---

## 6. On real hardware

- The occupancy math is real: scattered accesses across banked memory genuinely touch many banks, and
  each bank-level access has a cost. This is why real PIM systems care enormously about **data placement**
  and about co-locating data that's accessed together.
- Our model assumes uniform-random indices (a fair, neutral assumption for recsys embedding access). If
  a real workload had strong access locality, a locality-aware placement could lower `k` — but we
  measured that random DLRM indices defeat clustered placement (chapter 06), so you can't generally
  "place your way out" of the banking tax.

> **Reality check.** `k` is an *expected* occupancy under random placement; a real system's `k` depends
> on its actual index distribution and layout. We report the empirical `k` for our seeded workload and
> the closed form as a check — and we sweep placement to show the sensitivity honestly.

---

## Check your understanding
1. Why is the naive "DMR = L" wrong? What does PIM actually ship per bag, and why?
2. Compute `k` for `L = 8, B = 16` using `k = B·(1−(1−1/B)^L)`. Is PIM worth it there (DMR ≥ 1.5×)?
3. Explain the banking tax: why does *increasing* `B` *decrease* the data-movement win?

---

## References
- Classic occupancy / coupon-collector analysis (any probability text; e.g. Mitzenmacher & Upfal,
  *Probability and Computing*).
- S. Ghose et al., "Processing-in-Memory: A Workload-Driven Perspective," *IBM J. R&D*, 2019 (placement
  and bank-level concerns in real PIM).
- Grove: `../../pim-prereg.md` §3 (the `k(L,B)` definition + role), `../src/model.cpp`
  (`closed_form_k`, empirical sampling).

→ Previous: [04 — Byte-Accounting & the DMR](04-byte-accounting-and-dmr.md) · Next: [06 — The Crossover & Honest Science](06-the-crossover-and-honest-science.md)
