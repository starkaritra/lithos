# 03 — The Bank Model & a Fair Fight

> **Goal:** understand how we model memory as **banks** with local compute, and — just as important —
> how we make the GPU-vs-PIM comparison **fair** so the result can't be rigged. The fairness spine (one
> shared bandwidth cap + a coalesced baseline) is what separates an honest measurement from a demo.

---

## 1. The problem: a comparison is only as good as its baseline

It is trivially easy to "prove" PIM wins by cheating: give PIM a fast link and the GPU a slow one, or
compare PIM against a *badly written* GPU kernel that re-reads data. Any such win is meaningless. So
before modeling PIM at all, we must pin down a **fair fight** — the single most important design
decision in this arm (it echoes the spike's "don't rig the baseline" lesson, `../../spike-prereg.md`).

Two commitments make it fair:
1. **The baseline is the strongest reasonable GPU** — it moves each needed byte across the link *exactly
   once*, perfectly coalesced. This is the theoretical floor; we are not beating a strawman.
2. **One shared bandwidth cap** limits *both* machines. Neither side gets a faster link.

---

## 2. The bank model

Real memory is physically divided into **banks** that can be accessed in parallel. We model that
directly (`../include/pim/model.hpp`):

```
   global memory, B banks:
   ┌────────┬────────┬────────┬─── ... ───┬────────┐
   │ bank 0 │ bank 1 │ bank 2 │           │ bank B-1│   each bank:
   │  data  │  data  │  data  │           │  data   │    - holds a slice of the address space
   │  +PCU  │  +PCU  │  +PCU  │           │  +PCU   │    - has a tiny compute unit (PCU): +, ×, Σ
   └────────┴────────┴────────┴─── ... ───┴────────┘    - can emit ONE partial result over the link
```

**Placement** decides which bank holds row/element `i` — the function `place(i)` in
`../src/model.cpp`:
- **round-robin** (default, neutral): `place(i) = i mod B` — consecutive rows spread across banks.
- **clustered** (locality-aware alternative): contiguous chunks per bank.

Placement matters because it determines *how many banks a given set of accesses touches* — the crux of
chapter 05. We sweep both so no one can accuse us of a placement that flatters PIM (chapter 06).

---

## 3. The fairness spine: one shared bandwidth cap

Both machines are limited by the **same** off-chip link — a single `cap` (bytes/cycle). This is the one
number we add to close a gap Arm A had (its mini-GPU modeled coalescing but not a cross-warp *bandwidth
ceiling*; see `../../simt/docs/09`). Crucially, we add **only** this — not a full DRAM timing model —
because the data-movement result must rest on **countable bytes**, not on timing fidelity we can't
justify (a deliberate scope choice; see decision D-017 and chapter 06's honesty discussion).

```
   BASELINE (GPU)                          PIM
   reads every needed byte across          reads its data IN-BANK (off the link),
   the shared link, once, coalesced        sends only per-bank partials across the
   → aggregates on-chip after arrival      same shared link
                     └──────── same cap (bytes/cycle) for both ────────┘
```

PIM gets **no** free memory advantage: it still pays, over the link, for every byte that genuinely must
cross it — the indices it's told to fetch, the partials each bank emits, and the final output. Those
"hidden" costs are charged in full (chapter 04) precisely so the comparison stays honest.

---

## 4. The PIM ISA (minimal, like Arm A's)

The bank compute units speak a tiny instruction set — just enough to express aggregation near memory:

| Instruction | Meaning |
|-------------|---------|
| `pim_load` | read a value from the local bank (off the link) |
| `pim_add` / `pim_mac` | add / multiply-accumulate, in-bank |
| `pim_reduce` | fold this bank's local slice into one partial |
| `host_collect` | gather the `B` partials at the host and finish |
| `halt` | stop |

Notice what's *absent*: no bank-to-bank communication, no general control flow, no floating-point
menagerie. That minimalism is honest — real PIM units are similarly limited — and it keeps the model
legible, exactly as Arm A's 10-op ISA did.

---

## 5. In our model: parameters you can turn

All pre-registered (`../../pim-prereg.md` §3; defaults from DLRM literature):

| Symbol | Meaning | Default |
|---|---|---|
| `B` | number of banks | 16 |
| `d` | embedding dimension (elements per row) | 64 |
| `L` | rows per bag (pooling factor) | 40 |
| `b`, `idx_b` | bytes per element / per index | 4, 4 |
| `nb` | number of bags (batch) | 1024 |
| `R_tab` | rows in the embedding table | 4,000,000 |
| `cap` | shared off-chip link bytes/cycle | 32 |

The next two chapters use these to *count bytes* (chapter 04) and to expose why the win is smaller than
it first looks (chapter 05).

---

## 6. On real hardware

- Real DRAM really is banked (and grouped into ranks/channels); bank-level parallelism is a genuine
  hardware property PIM exploits.
- Real systems must also handle **placement** — how data is laid out across banks/channels — and it's a
  real tuning knob (and a limitation: you can't always co-locate the data an operation needs).
- The **shared bandwidth cap** is a faithful abstraction of the fundamental constraint (off-chip pins
  and their bandwidth are a hard physical limit); we just don't model the DRAM protocol beneath it.

> **Reality check.** We model banks + local aggregation + one bandwidth cap. We do **not** model DRAM
> refresh, row-buffer hits/misses, bank conflicts in timing, or inter-bank networks. Those affect
> *cycles*, not the *byte counts* our primary metric rests on — so omitting them keeps the result
> honest rather than hiding behind unmodeled timing.

---

## Check your understanding
1. Name the two commitments that make the GPU-vs-PIM comparison fair, and why each matters.
2. What does `place(i)` decide, and why will it turn out to be central (chapter 05)?
3. Why do we add a single shared bandwidth cap but deliberately *not* a full DRAM timing model?

---

## References
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., §2 (DRAM
  organization: banks, ranks, channels).
- S. Ghose et al., "Processing-in-Memory: A Workload-Driven Perspective," *IBM J. R&D*, 2019.
- Lithos: `../../pim-prereg.md` §2–§3 (the fair baseline + the model), `../../decisions.md` D-017.

→ Previous: [02 — Near-Memory Computing](02-near-memory-computing.md) · Next: [04 — Byte-Accounting & the DMR](04-byte-accounting-and-dmr.md)
