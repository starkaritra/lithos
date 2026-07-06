# Grove — Architecture

This document motivates each design before it specifies it (if a term isn't defined, it's defined at
first use). Grove has **two built arms** — a mini-GPU (Arm A) and a near-memory/PIM model (Arm C) —
plus a **concluded spike** whose negative result reshaped the project. Read `README.md` for the story
and `decisions.md` for the decision log; this file is the technical design.

> **History note.** Grove began as an **EDGE dataflow accelerator** aimed at XGBoost tree-ensemble
> inference. A pre-registered cost-model spike returned **NO-GO** (decisions D-008…D-014): a competent
> branchless CPU already neutralizes tree branching, so the win wasn't there. The project then pivoted
> (D-015) to the **A→C arc** documented below. The EDGE design is retired; this document specifies what
> was actually built.

---

## 0. The one idea both arms orbit: the memory wall
Arithmetic is cheap and fast; moving data to/from memory is slow and energy-expensive, and the gap has
grown for decades (Wulf & McKee 1995). A modern chip can do ~100+ arithmetic ops in the time it takes to
fetch one number from off-chip DRAM. So the real question for any high-throughput machine is not "how
fast is the arithmetic?" but **"how do you keep the arithmetic units fed despite slow memory?"**

Grove's two arms are two answers, measured:
- **Arm A (mini-GPU)** — the mainstream answer: *tolerate* the wall with massive parallelism (hide
  latency behind many threads; minimize transactions via coalescing). It then *measures* that
  memory-bound kernels stall anyway.
- **Arm C (PIM)** — the radical answer: *don't move the data* — put compute in the memory banks and ship
  only results. It *measures* how much data movement that actually saves, and where it stops helping.

---

## 1. Arm A — the mini-GPU (SIMT)

### 1.1 The problem it models
A GPU is a **throughput** machine: it spends its silicon on a huge array of simple arithmetic lanes
rather than on making one thread fast. To program that array without hand-packing vectors, NVIDIA's
model is **SIMT** (Single-Instruction, Multiple-Thread): you write per-thread code; the hardware runs
threads in lockstep groups (**warps**) on SIMD lanes. Arm A is a from-scratch, cycle-accurate SIMT core
built so the throughput-machine phenomena are **emergent and measurable**, not asserted.

### 1.2 The execution model (`simt/include/simt/core.hpp`, `src/core.cpp`)
- **Warp** = `WARP_SIZE = 32` lanes sharing **one program counter**, executing one instruction per
  cycle in lockstep. A per-lane **active mask** tracks which lanes are live (partial warps; divergence).
- **Single-issue, round-robin warp scheduler.** One warp issues per cycle; a warp that issued a
  long-latency memory op is "busy" (`ready_at` in the future), so the scheduler fills those cycles with
  *other* ready warps → **latency hiding** falls out of the schedule, not a special case.
- **Registers** are per-lane (`NREGS = 16` int32), so the same instruction computes different results
  per lane (that is SIMT).

### 1.3 The memory model
- **Word-addressed global memory.** A memory instruction's cost = `mem_latency + (transactions − 1) ×
  mem_txn_penalty`, where **transactions = the number of distinct memory *segments*** (`segment_words`,
  default 8) touched by the warp's active lanes. Contiguous access → few transactions (**coalesced**);
  scattered → up to 32 (**uncoalesced**). This makes coalescing a measurable effect.

### 1.4 The ISA (`simt/include/simt/isa.hpp`) — minimal by design (10 ops)
`mov` · `tid` (global thread id) · `iadd` · `imul` · `slt` (set-less-than → predicate) · `ld` · `st` ·
`jmp` · `bra` (predicated branch → divergence, with an else + join target) · `halt`. A two-pass
assembler with labels (`src/assembler.cpp`). Branch **divergence** is implemented with a per-warp
**reconvergence stack** (Fung et al.): when a warp's lanes disagree, the paths run serially under masks
and reconverge at the join — so divergence cost is emergent and measurable.

### 1.5 What it measures (all reproducible; see `simt/docs/` and `simt/analysis/`)
- **Latency hiding / occupancy** — 16× the warps costs ~7% more cycles (memory latency overlapped).
- **Coalescing** — stride 1→8 = 4→32 transactions.
- **Divergence** — a split warp runs both paths serially (measured vs a uniform warp).
- **Reduction** — a `log₂(n)` tree; warp-synchronous; every step diverges.
- **The memory wall** — one memory access ≈ 225 arithmetic ops; a scattered gather = 6 instructions but
  ~900 cycles. This is the forcing function that motivates Arm C.

### 1.6 Deliberately omitted (parking lot)
Branch prediction, out-of-order execution, large caches (CPU answers to a different question); thread
blocks + shared memory, warp-shuffle, a bandwidth-capped SIMT model, tiled matmul, RTL. These are
optional extensions, not needed to teach the core throughput-machine ideas.

---

## 2. Arm C — near-memory / Processing-In-Memory (PIM)

### 2.1 The bet
Arm A *measured* that memory-bound kernels are limited by **data movement**, not compute. The only way
to beat that (rather than tolerate it) is to **move the compute to the data**: put small compute units
in the memory banks so aggregation happens locally and only the (small) result crosses the off-chip
link. This is **PIM** (prior art: UPMEM, Samsung HBM-PIM; the Mutlu/Ghose research line). Grove's
contribution is an **open, minimal, cycle-approximate model + a fair, measured data-movement result**.

### 2.2 The model (`pim/include/pim/model.hpp`, `src/model.cpp`)
- Global memory split into **`B` banks**; row/element `i` lives in bank `place(i)` (round-robin default;
  clustered alternative). Each bank has a tiny PIM compute unit (add / multiply-accumulate / reduce).
- **One shared off-chip bandwidth cap** (bytes/cycle) limits **both** the GPU baseline and PIM — the
  fairness spine (you cannot rig the comparison by giving one side more bandwidth).
- **Minimal PIM ISA (~6 ops):** `pim_load`, `pim_add`, `pim_mac`, `pim_reduce`, `host_collect`, `halt`.

### 2.3 The measurement: byte-accounting, not cycles
The primary metric is **DMR = off-chip link bytes (baseline) ÷ off-chip link bytes (PIM)** — bytes are
**counted, not modelled** (assumption-free). For each kernel, *all* link traffic is charged to *both*
machines: input **indices**, **operands/partials**, and **output** (so PIM's win can't hide unaccounted
traffic).
- **Headline kernel — embedding-bag sum-pooling** (recsys/DLRM): gather `L` scattered rows from a huge
  table, sum to one vector. Baseline moves all `L` rows; PIM pre-sums co-banked rows and ships one
  partial **per distinct bank touched**.
- **The crux — the banking factor `k(L,B) = B·(1 − (1 − 1/B)^L)`:** the number of distinct banks a bag's
  `L` rows touch. Naive intuition says PIM sends 1 result (DMR = `L`); banking forces `k` partials, so
  the honest **DMR ≈ L/k**, far below the naive line.

### 2.4 What it measures (see `pim/docs/` and `pim/out/conclusion.md`)
Canonical (d=64, L=40, B=16): **DMR = 2.53×** — a *gray-zone* result. The win crosses the 3× bar only
for high pooling (L≳64) and vanishes (<1.5×) for low pooling (L≲16). Verdict (D-019):
**CONDITIONAL-GO** — the data-movement win is real but **banking-limited**; it pays off for high-pooling
recommendation features, not low-pooling ones. The reduction-ratio **crossover** (where PIM stops
helping) is the deliverable that proves the result is not a tautology.

### 2.5 Deliberately omitted (parking lot)
A general PIM compiler/language; DRAM-faithful timing (refresh, row buffers); RTL; elaborate energy
modelling; multi-level (near-cache + near-DRAM) PIM. The result rests only on **countable link bytes**,
not on timing fidelity.

---

## 3. Shared engineering conventions
- **The gem5 pattern (D-016):** each arm is a **headless C++ cycle/byte model** (MSVC + CMake + Ninja,
  `/W4`) with a **Python analysis layer** (numpy + matplotlib) that sweeps and plots. Clean engine/render
  split; the core is unit-tested and dependency-free.
- **Honesty tags** on every quantity: `[verified]` / `[believed]` / `[modelled]` / `[modelled-exact]`.
- **Pre-registration** for any decision that could be gamed: hypotheses, primary metric, and decision
  rule are fixed *before* data (`spike-prereg.md`, `pim-prereg.md`), and a **NO-GO is a valid outcome**.
- **Every load-bearing claim is measured and reproducible**, one command per arm.

---

## 4. Status & what could come next
Both arms are built, measured, and concluded (see `handoff.md`). Optional extensions: firm up the Arm C
verdict against a real DLRM per-feature pooling distribution vs the crossover `L*`; add tiled matmul or a
bandwidth-capped model to Arm A; add an SpMV kernel to Arm C's sweep. None are blocking — the A→C arc is
a complete, honest result.
