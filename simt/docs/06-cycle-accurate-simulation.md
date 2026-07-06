# 06 — Cycle-Accurate Simulation

> **Goal:** understand what "cycle-accurate" means, where our simulator sits in the spectrum
> of simulation techniques, and — the centerpiece — **derive the 681-cycle vector-add result
> by hand**, cycle by cycle. If you can reproduce that number with pen and paper, you
> understand the machine.

---

## 1. Why simulate at all?

You cannot learn (or design) a processor by building silicon for every idea — it costs
millions of dollars and months per iteration. So computer architects use **simulators**:
software models of hardware that let you run programs and measure behavior. The entire field
of architecture research runs on them (gem5, GPGPU-Sim, Accel-Sim). Building one is how you
*truly* learn architecture, because a simulator forces you to make every mechanism explicit —
there's nowhere to hand-wave.

---

## 2. The spectrum of simulators (know where you stand)

Not all simulators model the same thing. From least to most detailed:

```
   FUNCTIONAL          CYCLE-APPROXIMATE / -ACCURATE        RTL / GATE-LEVEL
   "what is the        "how LONG does it take, cycle        "what does the actual
    correct answer?"    by cycle, under a timing model?"     circuit do, per wire?"
   ─────────────       ──────────────────────────────       ──────────────────────
   fast; no timing     models the pipeline, scheduler,       models registers & logic
   (e.g. an emulator)  memory latency, hazards               (Verilog + Verilator)
   ↑ our LD/ST results ↑ OUR SIMULATOR LIVES HERE            ↑ Grove's optional Arm-A
                                                               stretch goal
```

- **Functional simulation** answers *"what is the right result?"* — it executes instructions
  correctly but says nothing about time. Our `execute()` doing the actual adds/loads is the
  functional part.
- **Cycle-accurate (here, cycle-*approximate*) simulation** answers *"how many cycles does it
  take?"* by modeling a timing story: how instructions issue, when memory returns, how the
  scheduler fills stalls. This is where the *architecture* lives, and where our simulator
  sits.
- **RTL/gate-level** models the literal circuit. Maximally accurate, slowest, needed for
  building real chips. (Verilator-based RTL is a possible far-future Grove stretch.)

> **Honesty about our label.** We say "cycle-accurate" in the teaching sense: a deterministic
> per-cycle timing model. It is not *validated* against real silicon (that would make it
> "cycle-accurate" in the strict industrial sense). It is a **cycle-approximate analytical
> model** — precise and reproducible, but a teaching abstraction. Chapters flag where it
> simplifies.

---

## 3. Our timing model in one page

The model (in `src/core.cpp`, `Core::run`) rests on a few explicit rules:

| Rule | Value / behavior | Chapter |
|------|------------------|---------|
| Warp size | 32 lanes, lockstep | 02 |
| Issue width | **1 warp-instruction per cycle** (single-issue) | 03 |
| ALU / MOV / TID / branch op latency | 1 cycle | — |
| Memory op latency | `mem_latency + (transactions − 1) × mem_txn_penalty` | 03, 04 |
| Defaults | `mem_latency = 200`, `mem_txn_penalty = 8`, `segment_words = 8` | — |
| Scheduling | round-robin over **ready** warps; clock jumps only when *nobody* is ready (a real stall) | 03 |
| Completion | `stats.cycles` = the cycle the last instruction finishes | — |

Every performance story in this course is a consequence of these rules — not a special case.
That's the mark of a good model: rich behavior from few, honest assumptions.

---

## 4. The centerpiece: deriving 681 cycles by hand

Let's prove the number `build\simt.exe kernels\vector_add.sasm 32` prints. The program is 12
instructions; with 32 threads it's **one warp**, so there's no other warp to hide latency —
every memory op fully stalls. First, the per-instruction latencies:

```
  idx  instr           kind    latency
   0   mov  r1, 0       ALU     1
   1   mov  r2, 32      ALU     1
   2   mov  r3, 64      ALU     1
   3   tid  r0          ALU     1
   4   iadd r4, r1, r0  ALU     1
   5   iadd r5, r2, r0  ALU     1
   6   iadd r6, r3, r0  ALU     1
   7   ld   r7, r4      MEM     A[0..31] → segments {0,1,2,3} = 4 txns → 200 + 3×8 = 224
   8   ld   r8, r5      MEM     B[32..63] → segments {4,5,6,7} = 4 txns → 224
   9   iadd r9, r7, r8  ALU     1
  10   st   r6, r9      MEM     C[64..95] → segments {8,9,10,11} = 4 txns → 224
  11   halt             —       1
```

Now walk the scheduler. `cycle` is the clock; `ready_at` is when the warp may next issue.
With one warp, the loop issues one instruction, advances `cycle` by 1, and — when the warp is
mid-memory-op and *nothing else is ready* — **jumps** the clock to the warp's `ready_at`.

```
  cycle  action                              warp ready_at        stats.cycles
   0     issue #0 mov   (ALU,1)              → 1                   1
   1     issue #1 mov                         → 2                   2
   2     issue #2 mov                         → 3                   3
   3     issue #3 tid                         → 4                   4
   4     issue #4 iadd                        → 5                   5
   5     issue #5 iadd                        → 6                   6
   6     issue #6 iadd                        → 7                   7
   7     issue #7 ld    (MEM,224)             → 7+224 = 231         231
   8     nobody ready → JUMP clock to 231
  231    issue #8 ld    (MEM,224)             → 231+224 = 455       455
  232    nobody ready → JUMP clock to 455
  455    issue #9 iadd                        → 456                 456
  456    issue #10 st   (MEM,224)             → 456+224 = 680       680
  457    nobody ready → JUMP clock to 680
  680    issue #11 halt (1)                   → 681                 681
        all warps halted → stop
```

**Total = 681 cycles.** Notice the anatomy: 8 one-cycle ALU issues, then three memory ops
that each cost ~224 cycles *fully exposed* because a lone warp has nothing to overlap them
with. `3 × 224 = 672`, plus the ~9 cycles of ALU/issue work ≈ 681. **The memory ops dominate
by ~70×** — you are staring at the memory wall in a single number.

The test `test_vector_add_one_warp` asserts exactly `s.cycles == 681`, `s.mem_ops == 3`, and
`s.mem_transactions == 12`. If you change the timing model, that test *should* change — it's a
guardrail on the model's determinism, not a magic constant.

> **Contrast with chapter 03.** There, adding a *second* warp let those 224-cycle memory
> stalls overlap, so 2 warps cost ~= 1 warp. Here, with one warp, the stalls are fully
> exposed. Same model, opposite outcome — the difference *is* latency hiding.

---

## 5. Reproducibility (a result you can't reproduce isn't evidence)

The model is fully deterministic: no randomness in the single-warp path, and the multi-warp
scheduler is fixed round-robin. Given the same kernel, thread count, and `CoreConfig`, you
get the same cycle count every run. This is a deliberate discipline carried over from the
project's measurement phase (the spike logged versions, seeds, and hashes for exactly this
reason — see `../../spike-prereg.md` §8). When you extend the model, keep it deterministic so
every number stays a *reproducible* claim.

---

## 6. On real tools

- **gem5** — the standard academic/industry simulator; a C++ core scripted by Python. Our
  C++-core + Python-analysis split (decision D-016) deliberately mirrors it.
- **GPGPU-Sim** (Bakhoda et al., 2009) and **Accel-Sim** (Khairy et al., 2020) — the standard
  cycle-level GPU simulators; they model warp scheduling, divergence, coalescing, and the
  memory system in far greater detail than we do, but with the *same conceptual pieces* you're
  learning here.
- Real cycle-accurate simulators are validated against hardware and run millions–billions of
  cycles; ours runs thousands and targets *understanding*, not silicon sign-off.

---

## Check your understanding
1. Put functional, cycle-accurate, and RTL simulation in order of detail, and say what
   question each answers.
2. In the 681-cycle trace, which cycles are "real stalls," and what makes the clock jump?
3. Of the 681 cycles, roughly how many are memory vs everything else — and what does that
   ratio illustrate?

---

## References
- N. Binkert et al., "The gem5 Simulator," *ACM SIGARCH Computer Architecture News*, 2011.
- A. Bakhoda, G. Yuan, W. Fung, H. Wong, T. Aamodt, "Analyzing CUDA Workloads Using a Detailed
  GPU Simulator," *ISPASS*, 2009 (GPGPU-Sim).
- M. Khairy, Z. Shen, T. M. Aamodt, T. G. Rogers, "Accel-Sim: An Extensible Simulation
  Framework for Validated GPU Modeling," *ISCA*, 2020.
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., Appendix on
  simulation methodology.

→ Previous: [05 — Branch Divergence](05-branch-divergence.md) · Next: [07 — The ISA & Assembler](07-isa-and-assembler.md)
