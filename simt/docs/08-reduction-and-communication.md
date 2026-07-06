# 08 — Reduction & Cross-Thread Communication

> **Goal:** understand how thousands of independent threads **combine** their results — the
> "reduction" pattern (sum, max, count…) — which forces threads to *communicate*. You'll
> meet the tree reduction, warp-synchronous execution, and why reduction inherently
> diverges. You'll derive our `mem[0] = 496` result.

---

## 1. The problem: independent threads, one answer

So far every thread has been gloriously independent: in vector-add, thread `i` writes `C[i]`
and never looks at another thread's work. That's the easy, ideal GPU case.

But a huge class of problems needs the threads to **combine** their values into one: the sum
of an array, the maximum, a dot product, the number of elements matching a condition. This is
a **reduction**: many inputs → one output. And it forces the thing independent threads never
had to do — **communicate** with each other.

```
   independent (easy):                 reduction (needs communication):
   t0 → C[0]   t1 → C[1]  ...           t0 t1 t2 t3 t4 t5 t6 t7
        (no thread talks to another)         \  /   \  /   \  /   \  /
                                               +      +      +      +
                                                \    /        \    /
                                                  +             +
                                                    \          /
                                                        +   =  one answer
```

---

## 2. Intuition: the tournament bracket

How do 32 players produce one champion fastest? Not by having one referee play all 31 matches
in series (that's the slow, sequential sum). You run a **tournament**: 16 matches in parallel
(round 1), then 8, then 4, 2, 1 — **5 rounds** for 32 players. Each round halves the field.

That's a **tree reduction**, and the number of rounds is `log₂(n)`: for 32 elements, 5 steps
instead of 31. Turning a sequential `O(n)` chain into a parallel `O(log n)` tree is the whole
reason reduction is done this way on a GPU.

```
   step (stride):   16        8       4      2     1
   active lanes:    0..15  →  0..7  → 0..3 → 0..1 → 0     ← each step halves the workers
   mem[tid] +=      m[+16]    m[+8]   m[+4]  m[+2]  m[+1]
```

At each step, the surviving lanes add in a partner from `stride` away, and the stride halves.
After `log₂(n)` steps, `mem[0]` holds the total.

---

## 3. The mechanism: communication needs synchronization

For lane 0 to add lane 16's value, lane 16 must have *finished writing* it. Reduction
therefore needs a **barrier** between steps: everyone finishes step *k*'s writes before
anyone starts step *k+1*'s reads. Otherwise you get a race — reading a value that hasn't been
updated yet.

Where do threads exchange values? Two levels matter:

- **Within one warp (32 lanes):** the lanes run in **lockstep** (chapter 02) — one program
  counter, one instruction at a time. So step *k*'s store instruction fully completes for all
  active lanes before step *k+1*'s load instruction begins. Lockstep *is* an implicit barrier.
  This is called **warp-synchronous** programming.
- **Across warps (a whole block):** different warps run asynchronously (that's how latency is
  hidden, chapter 03!). So a multi-warp reduction must use **shared memory** + an explicit
  barrier (`__syncthreads()` on real GPUs) to synchronize. (Our v1 has one warp, so we get
  the barrier for free; multi-warp shared-memory reduction is a future slice.)

> **The tension is beautiful:** the very asynchrony that hides memory latency (good) is what
> makes cross-warp communication need explicit barriers (a cost). Independence buys throughput;
> communication costs synchronization.

---

## 4. Reduction inherently diverges

Look at the active-lane counts again: 16 → 8 → 4 → 2 → 1. At **every** step, some lanes are
switched off (they have no partner to add). That means every reduction step is a **divergent
branch** (chapter 05): `if (tid < stride) { ... }`. By the last step, 31 of 32 lanes sit idle
while lane 0 does the final add.

This is a real, well-known inefficiency of naive GPU reduction — the hardware is
increasingly under-utilized as the tree narrows. Optimized GPU reductions fight it with tricks
(sequential addressing, processing multiple elements per thread, warp-shuffle intrinsics).
Our simulator lets you *see* it: the reduction reports **5 divergences**, one per step.

---

## 5. In our mini-GPU: the reduction kernel

See `kernels/reduction.sasm` (and `reduction_kernel()` in `tests/test_simt.cpp`). Each step is
the same guarded block, with the stride hard-coded (unrolled — exactly how CUDA reduction
tutorials present it):

```
mov  r10, 16          # stride
slt  r1, r0, r10      # predicate: participate if tid < stride
bra  r1, j16, j16     # non-participants skip straight to the join
iadd r2, r0, r10      # partner index = tid + stride
ld   r3, r0           # mem[tid]
ld   r4, r2           # mem[tid + stride]
iadd r5, r3, r4
st   r0, r5           # mem[tid] += mem[tid + stride]
jmp  j16
j16:                  # (repeat for strides 8, 4, 2, 1)
```

No explicit barrier appears — and that's the lesson: within one warp, the lockstep model
already orders step *k*'s `st` before step *k+1*'s `ld`. The `bra`/`jmp` guard is what makes
only `tid < stride` lanes participate (and what causes the divergence).

---

## 6. Measure it yourself — and derive 496

Seed `mem[i] = i` for `i = 0..31` (the CLI does this via its `A[i]=i` seeding) and run:

```
build\simt.exe kernels\reduction.sasm 32
→ reduced sum : mem[0] = 496   divergences: 5   mem ops: 15
```

**Why 496?** It's `0 + 1 + 2 + … + 31 = 31·32/2 = 496`. Trace the tree to confirm the
mechanism (with `mem[i] = i`):

```
  start:            mem[i] = i
  after stride 16:  mem[i] = i + (i+16) = 2i+16      (i < 16)
  after stride 8:   mem[i] = 4i + 48                 (i < 8)
  after stride 4:   mem[i] = 8i + 112                (i < 4)
  after stride 2:   mem[i] = 16i + 240               (i < 2)
  after stride 1:   mem[0] = 240 + 256 = 496         ✓
```

**Why 5 divergences and 15 memory ops?** 5 steps, each a divergent `if` (→ 5 divergences) and
each doing 2 loads + 1 store (→ 15 memory ops). The test `test_reduction` asserts
`mem[0] == 496` *and* `divergent_branches == 5` — correctness and the divergence cost, pinned.

---

## 7. On real GPUs

- The canonical reference is **Mark Harris's "Optimizing Parallel Reduction in CUDA"** — it
  walks through seven successively faster versions, each removing a specific inefficiency
  (divergent branching, bank conflicts, idle threads, loop overhead). Our kernel is the naive
  starting point of that sequence; the optimizations are excellent future slices.
- Modern GPUs also offer **warp-shuffle** instructions (`__shfl_down_sync`) that let lanes
  exchange registers *directly*, with no shared memory and no barrier — the hardware's answer
  to "communication is expensive."
- Multi-warp / multi-block reductions use **shared memory** + `__syncthreads()`, then a final
  cross-block pass (or atomics). Reduction is a gateway to the whole GPU memory hierarchy.

> **Reality check.** Our reduction is a single-warp, global-memory, unrolled tree — enough to
> teach cross-thread communication, warp-synchronous execution, and reduction-divergence. It
> omits shared memory, bank conflicts, and warp shuffles (all future slices, once we add
> thread blocks). The `log₂(n)` tree and the divergence cost are faithfully shown.

---

## Check your understanding
1. Why is a tree reduction `O(log n)` steps instead of `O(n)`, and why does that suit a GPU?
2. Our kernel has no explicit barrier — why is it still correct within one warp, and why
   would a *multi-warp* reduction need one?
3. Explain why the reduction reports exactly 5 divergences for 32 elements, and what that says
   about warp utilization in the last step.

---

## References
- M. Harris, "Optimizing Parallel Reduction in CUDA," NVIDIA Developer technical report.
- NVIDIA, *CUDA C++ Programming Guide* — "Shared Memory," "Synchronization Functions," and
  "Warp Shuffle Functions."
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., §4.4.
- Lithos code: `kernels/reduction.sasm`, `test_reduction` in `tests/test_simt.cpp`.

→ Previous: [07 — The ISA & Assembler](07-isa-and-assembler.md) · Next: [09 — The Memory Wall, Measured](09-memory-wall-measured.md)
