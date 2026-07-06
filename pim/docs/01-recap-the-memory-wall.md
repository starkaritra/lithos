# 01 — Recap: the Memory Wall

> **Goal:** re-establish, crisply, the one fact that justifies this entire arm — that data movement,
> not arithmetic, is the bottleneck — and see why the GPU's strategy of *tolerating* it has a hard
> ceiling. If you did Arm A, this is a fast recap with the key numbers; if you didn't, it's the
> motivation you need.

---

## 1. The problem: arithmetic is cheap, moving data is not

For decades, the rate at which a chip can *compute* grew far faster than the rate at which memory can
*feed it data*. The gap is now enormous, and it has a name: the **memory wall** (Wulf & McKee, 1995).

```
   doing the math ....... ~1 cycle for an add
   getting the data ..... HUNDREDS of cycles for an off-chip DRAM access

   energy is even more lopsided (order-of-magnitude):
     one 32-bit ADD ................ ~1 unit
     read 32 bits from on-chip SRAM  ~10 units
     read 32 bits from off-chip DRAM ~1000+ units   ⚠ the wall
```

So for any high-throughput machine the real question is **not** "how fast is the arithmetic?" — that's
already easy — but **"how do you keep the arithmetic units busy despite slow, expensive memory?"**

---

## 2. Arm A measured the wall — here are the numbers

In Arm A you didn't take this on faith; you *measured* it on your own mini-GPU:

- **Latency face:** an arithmetic-intensity sweep gave `cycles = 451 + K` (K = arithmetic ops per
  memory access). The 451-cycle floor is pure memory — **one access ≈ 225 arithmetic ops**. Unless a
  kernel does hundreds of ops per value it loads, it's *waiting on memory*, not computing.
- **Bandwidth face:** a scattered gather kernel ran **6 instructions in ~900 cycles** — ~99% of the
  time was pure data movement.

Those two numbers *are* the wall, made concrete. (See `../../simt/docs/09-memory-wall-measured.md`.)

---

## 3. The GPU's strategy: *tolerate* the wall (and its ceiling)

Everything Arm A built is a way to *tolerate* slow memory, not remove it:

- **Latency hiding (occupancy):** run thousands of threads so that while some wait on memory, others
  compute. This hides *latency* — but it does nothing about the total *volume* of data that must move.
- **Coalescing:** make neighboring threads touch neighboring memory so accesses fuse into few
  transactions. This reduces *wasted* bandwidth — but you still move every useful byte.

```
   tolerate the wall (GPU):                the ceiling:
   ┌──────────┐  long, costly  ┌───────┐   for a kernel with no reuse (e.g. a one-shot gather),
   │ 1000s of │◄══════════════►│ DRAM  │   there is nothing to hide behind and nothing to
   │ threads  │  the data STILL │       │   coalesce away — the data volume itself is the cost.
   └──────────┘  makes the trip └───────┘   Hiding + coalescing cannot beat that.
```

**The ceiling, stated plainly:** if your program's whole job is to move a lot of data and compute
almost nothing (embedding lookups, sparse operations, graph traversal), then no amount of threads or
coalescing helps — the machine is a very expensive memory-copy engine. Arm A measured exactly this: the
gather kernel spent its 900 cycles moving data, and more ALUs would have sat just as idle.

---

## 4. The only way past the ceiling

If the fundamental cost is *the data making the long trip between memory and compute*, there are only
two moves:

1. **Move the data less** — caching, reuse, coalescing. That's the GPU's approach. It *tolerates* the
   wall but cannot beat a kernel that has no reuse.
2. **Move the compute to the data** — put small arithmetic units *inside or beside* the memory, so the
   data barely moves. This is **near-memory / Processing-In-Memory (PIM)**, and it is what Arm C
   builds and measures.

That second move is the subject of the rest of this course. The question we'll answer — honestly, with
counted bytes — is: *how much data movement does PIM actually save, and for which workloads?*

> **Reality check.** "Tolerate vs remove" is the right intuition, but it's not absolute: real systems
> combine both (a PIM system still has a normal memory hierarchy). Arm C isolates the *data-movement*
> question specifically, because that is the part PIM uniquely changes.

---

## Check your understanding
1. State the memory wall in one sentence, and give the two Arm-A numbers that quantify its latency and
   bandwidth faces.
2. Why can't latency hiding + coalescing beat a one-shot scattered gather?
3. What are the only two strategies against the wall, and which one does a GPU use?

---

## References
- Wm. A. Wulf & S. A. McKee, "Hitting the Memory Wall: Implications of the Obvious," *ACM SIGARCH
  Computer Architecture News*, 1995.
- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model," *CACM*,
  2009 (compute-bound vs bandwidth-bound).
- M. Horowitz, "Computing's Energy Problem (and what we can do about it)," *ISSCC*, 2014 (the energy
  numbers).
- Grove: `../../simt/docs/09-memory-wall-measured.md` (the Arm-A measurements recapped here).

→ Next: [02 — Near-Memory Computing](02-near-memory-computing.md)
