# 02 — The SIMT Execution Model

> **Goal:** understand how thousands of "threads" on a GPU actually run. You'll learn what
> a **warp** is, what **lockstep** execution means, and why the GPU's model (SIMT) feels
> like ordinary threading but behaves like vector hardware underneath.

---

## 1. The problem: how do you *program* a throughput machine?

Chapter 01 said a GPU wins by applying one instruction to many data elements at once (SIMD).
But pure SIMD is painful to program: you must manually pack data into fixed-width vectors
(`add these 8 floats to those 8 floats`) and hand-manage the leftovers. Nobody wants to
write a physics simulation that way.

NVIDIA's insight (introduced with the 2006 Tesla / G80 architecture) was to hide the vector
machine behind a **thread** abstraction. You write code as if for *one* data element — "I am
thread `i`; I handle element `i`" — and launch thousands of these threads. The hardware then
quietly runs them in groups, in lockstep, on its SIMD lanes. This is **SIMT**:
**Single-Instruction, Multiple-Thread.**

```
   What YOU write (one thread's view):        What the HARDWARE runs (a warp):
   ┌──────────────────────────────┐           32 threads, ONE instruction, together
   │  i = my_thread_id             │           ┌──┬──┬──┬──┬── ... ──┬──┐
   │  c[i] = a[i] + b[i]           │    ─────▶ │t0│t1│t2│t3│   ...   │31│  ← all execute
   └──────────────────────────────┘           └──┴──┴──┴──┴── ... ──┴──┘     "c=a+b"
      simple, scalar, familiar                 in the SAME cycle
```

You get the programmability of threads and the efficiency of vectors. That trade is why
CUDA and GPU computing took over.

---

## 2. The hierarchy: threads → warps → (blocks → grid)

GPU execution is organized in a strict hierarchy. Our mini-GPU implements the two levels
that matter most for understanding the hardware; the outer two are described for context.

```
   GRID  (the whole kernel launch: all threads)
     └── BLOCK  (a group that can share fast on-chip memory & synchronize)   [context]
           └── WARP  (32 threads that execute in LOCKSTEP)          ← the key unit
                 └── THREAD / LANE  (one data element's worth of work)
```

- **Thread / lane.** One instance of your kernel. Has its own registers. In our simulator,
  a lane owns `NREGS` (16) private integer registers — see `include/simt/core.hpp`.
- **Warp.** A fixed bundle of **32 threads** (real NVIDIA warp size; `WARP_SIZE` in our
  code). *The warp, not the thread, is the true unit of scheduling and execution.* All 32
  lanes of a warp execute the **same instruction** in the **same cycle**.
- **Block** *(context, not in v1).* A group of warps that can share a fast "shared memory"
  scratchpad and synchronize with barriers. Needed for kernels like tiled matmul.
- **Grid** *(context).* All the blocks of one kernel launch.

> **Why 32?** It's a hardware design point: wide enough to amortize the cost of fetching and
> decoding one instruction across many lanes, narrow enough that divergence (chapter 05)
> isn't catastrophic. AMD historically used 64 ("wavefront"); NVIDIA uses 32.

---

## 3. Lockstep and the active mask

"Lockstep" means: the warp has **one program counter** for all 32 lanes. Every lane executes
the same instruction at the same time; they differ only in the *data* in their registers
(notably their thread id, so lane `i` works on element `i`).

But what if you launch 40 threads? That's 1 full warp (32) + 1 partial warp (8). The partial
warp still occupies all 32 lanes of hardware — 8 do useful work, 24 sit idle. To track which
lanes are "real," each warp carries an **active mask**: one bit per lane.

```
   Launch 40 threads → 2 warps:

   warp 0 (lanes 0..31):  active mask = 11111111 11111111 11111111 11111111  (all real)
   warp 1 (lanes 0..31):  active mask = 11111111 00000000 00000000 00000000  (8 real, 24 idle)
                                        ^^^^^^^^ threads 32..39
```

Inactive lanes still "run" the instruction physically, but their results are thrown away.
This is your first glimpse of a deep truth: **on a SIMT machine, work you don't use still
costs you.** The active mask is also the exact mechanism behind branch divergence
(chapter 05) — an `if` simply switches some mask bits off for a while.

---

## 4. In our mini-GPU: how SIMT is modeled

Open `include/simt/core.hpp` and `src/core.cpp`. The mapping is direct and readable:

```cpp
constexpr int WARP_SIZE = 32;   // lanes per warp — the real NVIDIA number
constexpr int NREGS     = 16;   // private integer registers per lane

struct Warp {
    int id;                                   // which warp (for computing global tid)
    int pc;                                   // ONE program counter for all 32 lanes
    std::array<bool, WARP_SIZE> active;       // the active mask
    std::array<std::array<int32_t, NREGS>, WARP_SIZE> regs;  // regs[lane][reg]
};
```

Three things to notice, each a faithful echo of real hardware:

1. **One `pc` per warp, not per lane.** That *is* lockstep. There is no way for lane 5 to
   be at a different instruction than lane 6 within a warp.
2. **`regs[lane][reg]`** — every lane has its own private registers, so the same instruction
   (`iadd r4, r1, r0`) computes a *different* result per lane because `r0` (the thread id)
   differs. This is how "one instruction, many data" works.
3. **The `active` mask** gates writes. Look at how `execute()` applies an instruction:

```cpp
case Op::IADD:
    for (int l = 0; l < WARP_SIZE; ++l)
        if (w.active[l])                       // inactive lanes do nothing observable
            w.regs[l][in.rd] = w.regs[l][in.ra] + w.regs[l][in.rb];
    break;
```

The loop over lanes is the simulator *pretending* to be 32 parallel ALUs. On silicon these
happen truly simultaneously; in software we iterate, but the *semantics* — same instruction,
per-lane data, masked writes — are identical.

The thread-id instruction is what breaks the symmetry between lanes:

```cpp
case Op::TID:
    for (int l = 0; l < WARP_SIZE; ++l)
        if (w.active[l]) w.regs[l][in.rd] = w.id * WARP_SIZE + l;   // GLOBAL thread id
    break;
```

`w.id * WARP_SIZE + l` is the global thread id — warp 0 lane 5 → tid 5; warp 1 lane 5 →
tid 37. This is the SIMT contract: *"you are thread `tid`; go do element `tid`."*

---

## 5. Measure it yourself

The `vector_add` kernel is the canonical first SIMT program: thread `i` computes
`C[i] = A[i] + B[i]`.

```
build\simt.exe kernels\vector_add.sasm 32
```

Expected:
```
threads       : 32 (1 warp(s) x 32 lanes)
result        : PASS (C[i] == 3*i)
warp-instrs   : 12
```

Note the last line: **12 warp-instructions** for 32 threads' worth of work. A scalar machine
would execute the 12-instruction program 32 times = 384 instruction-executions. The warp did
it in 12 — one instruction fetched/decoded, 32 results produced. *That ratio is the whole
point of SIMT.*

The test `test_tid_global` (in `tests/test_simt.cpp`) proves the id mapping: it has every
thread store its id to `mem[tid]` and checks `mem[i] == i` for 64 threads (2 warps) — i.e.
warp 1's lanes correctly produce ids 32..63.

---

## 6. On real GPUs

- Real hardware groups a warp's lanes onto **SIMD execution units** and shares one
  fetch/decode. Modern NVIDIA GPUs also split a warp's 32 lanes across a few cycles
  internally, but the programmer-visible model is lockstep-per-warp.
- Since NVIDIA Volta (2017), each thread has its own program counter for *independent thread
  scheduling* — a refinement that eases some divergence cases. The classic mental model
  (one PC per warp) is still the right place to start, and is what our simulator implements.
- **SPMD vs SIMT.** You write **SPMD** code (Single Program, Multiple Data — every thread
  runs the same program). The hardware executes it as **SIMT** (bundling threads into
  lockstep warps). SPMD is the programming model; SIMT is the execution model.

> **Reality check.** Our simulator has one warp size, no thread blocks, no shared memory,
> and no independent-thread-scheduling. Those are deliberate omissions to keep the core
> legible. We'll add blocks/shared memory when we build the tiled-matmul kernel.

---

## Check your understanding
1. If you launch 100 threads, how many warps run, and what does the active mask of the last
   warp look like?
2. Why does the same instruction `iadd r4, r1, r0` produce a different value in each lane?
3. A kernel executes 12 warp-instructions for a 32-thread launch. How many
   instruction-executions would an equivalent scalar loop perform, and what does the
   difference tell you?

---

## References
- E. Lindholm, J. Nickolls, S. Oberman, J. Montrym, "NVIDIA Tesla: A Unified Graphics and
  Computing Architecture," *IEEE Micro*, 2008 (introduces the SIMT model).
- J. Nickolls & W. J. Dally, "The GPU Computing Era," *IEEE Micro*, 2010.
- NVIDIA, *CUDA C++ Programming Guide* — "Thread Hierarchy" and "SIMT Architecture."
- NVIDIA, *Volta V100 Whitepaper*, 2017 (independent thread scheduling).
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., §4.4.

→ Previous: [01 — Why GPUs Exist](01-why-gpus.md) · Next: [03 — Latency Hiding & Occupancy](03-latency-hiding-occupancy.md)
