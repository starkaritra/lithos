# 02 — Near-Memory Computing

> **Goal:** understand the radical idea behind Arm C — moving computation *into* the memory — plainly
> and correctly. You'll learn what PIM is, the single condition under which it helps, the real systems
> that do it, and exactly what our model will and won't capture.

---

## 1. The problem: the data is in the wrong place

Chapter 01 left us with a stark fact: for memory-bound kernels, the cost *is* the long trip data makes
between the memory (where it lives) and the compute units (where the math happens). A normal machine —
CPU or GPU — has a fixed floor plan:

```
   ┌─────────────┐        thin, slow, expensive         ┌──────────────────┐
   │  COMPUTE    │◄════════════ the link ═══════════════►│   MEMORY (DRAM)  │
   │ (fast, big) │      every operand crosses here        │  (huge, slow)    │
   └─────────────┘                                        └──────────────────┘
```

You can make the compute faster, add more lanes, run more threads — none of it changes the fact that
**the data has to cross that link.** For a kernel that mostly moves data, the link is the whole story.

---

## 2. The idea: bring the compute to the data

**Processing-In-Memory (PIM)**, also called **near-memory computing**, changes the floor plan. Instead
of shipping data to a distant processor, it puts **small compute units next to (or inside) the memory
banks**, so simple operations happen *where the data already lives* — and only the (small) *result*
crosses the link.

```
   ┌──────────────────────────────────────────────┐
   │  MEMORY, now with tiny compute units inside    │
   │  ┌───────┐┌───────┐┌───────┐┌───────┐          │
   │  │bank+  ││bank+  ││bank+  ││bank+  │  ...      │   the data barely moves;
   │  │ +/×/Σ ││ +/×/Σ ││ +/×/Σ ││ +/×/Σ │          │   only RESULTS cross the link
   │  └───────┘└───────┘└───────┘└───────┘          │
   └───────────────────────┬────────────────────────┘
                           │ only small results
                           ▼
                     ┌─────────────┐
                     │    host      │
                     └─────────────┘
```

**Analogy.** You have 10,000 receipts in a warehouse and you want the total. The GPU approach trucks
all 10,000 receipts to your office and adds them there (10,000 trips' worth of data). The PIM approach
puts a clerk with a calculator *in the warehouse*, who adds them on site and phones you a single number.
Same answer; almost none of the hauling.

---

## 3. The one condition: PIM only helps when you aggregate

Here is the crucial, honest constraint — the thing chapter 05 will make precise:

> **PIM helps only when a kernel boils a LOT of input down to a LITTLE output.** If the program needs
> all the raw data anyway (a pure copy, or a gather with no summation), the clerk-in-the-warehouse has
> nothing to reduce — everything still has to be sent, and PIM's advantage collapses.

- **Aggregation kernels win:** sum, max, count, dot-product, embedding-bag pooling, sparse
  matrix-vector — output ≪ input.
- **Pure movement kernels don't:** copy, transpose, a gather with no reduction — output ≈ input.

This condition is not a footnote — it *is* the result of Arm C. A dishonest project would pick an
aggregating kernel, show a big win, and declare victory. An honest one (chapter 06) measures the whole
spectrum and shows *exactly where the win appears and where it disappears.*

---

## 4. In our model: what "compute in a bank" means

Our PIM model (`../include/pim/model.hpp`, `../src/model.cpp`) is deliberately minimal:
- Memory is split into **`B` banks**; each holds a slice of the data and has a tiny compute unit that
  can do **add / multiply-accumulate / reduce** on data *resident in that bank*.
- A bank can pre-aggregate its local data and emit **one partial result**, which is what crosses the
  link — instead of all the raw rows it holds.
- The host collects the per-bank partials and finishes the job.

A tiny **PIM ISA** captures this (chapter 03): `pim_load` (bank-local), `pim_add` / `pim_mac`,
`pim_reduce` (fold a bank's slice), `host_collect`, `halt`. That's the whole vocabulary — enough to
express aggregation near memory, nothing more.

---

## 5. On real hardware

PIM is not hypothetical — it is a live research area and a shipping commercial reality:
- **UPMEM** ships DPUs (DRAM Processing Units) — general-purpose cores next to DRAM banks.
- **Samsung HBM-PIM (Aquabolt-XL)** and **SK Hynix AiM** put compute in high-bandwidth memory, targeting
  exactly the memory-bound AI kernels we care about.
- The academic line (Mutlu, Ghose, and others) spans near-DRAM and in-DRAM (e.g. Ambit) computing.

Grove's contribution is **not** "we invented PIM" (we didn't) — it is *an open, minimal, honest,
cycle-approximate model that measures the data-movement win on a real workload and shows where it holds*
— the same honest-reproduction spirit as Arm A.

> **Reality check.** Real PIM has hard constraints our model omits: banks have limited compute, limited
> precision, and communication between banks is expensive; DRAM has timing (refresh, row buffers) we do
> not model. We deliberately keep only what the *data-movement* question needs — banks, bank-local
> aggregation, and one shared off-chip bandwidth cap (chapter 03) — so the result rests on countable
> bytes, not on timing fidelity we can't justify.

---

## Check your understanding
1. In one sentence, how does PIM change the machine's floor plan, and what still crosses the link?
2. State the single condition under which PIM helps. Give one kernel that satisfies it and one that
   doesn't.
3. Why is "pick an aggregating kernel and show a big win" a *dishonest* way to evaluate PIM?

---

## References
- S. Ghose, A. Boroumand, J. S. Kim, J. Gómez-Luna, O. Mutlu, "Processing-in-Memory: A Workload-Driven
  Perspective," *IBM Journal of R&D*, 2019 (excellent survey + the memory-wall motivation).
- V. Seshadri et al., "Ambit: In-Memory Accelerator for Bulk Bitwise Operations," *MICRO*, 2017.
- UPMEM PIM; Samsung HBM-PIM (Aquabolt-XL); SK Hynix AiM — real near-memory systems.
- Grove: `../../decisions.md` (D-011 the parked PIM idea, D-015/D-017 the arc + scope).

→ Previous: [01 — Recap: the Memory Wall](01-recap-the-memory-wall.md) · Next: [03 — The Bank Model & a Fair Fight](03-the-bank-model-and-fairness.md)
