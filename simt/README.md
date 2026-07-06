# Grove — Arm A: a mini-GPU (SIMT) you can measure

A from-scratch, cycle-accurate **SIMT** (Single-Instruction, Multiple-Thread) core —
the execution model real GPUs use. You write a tiny kernel, it compiles to a small
custom ISA, and a cycle-accurate simulator runs it while measuring the effects that
actually decide GPU performance. Building this is how you *learn GPU architecture*:
the phenomena aren't described, they're **emergent from the model**.

This is **Arm A** of the project (see `../handoff.md`, `../decisions.md` D-015/D-016).
Arm A exposes the **memory wall**; **Arm C** (later) builds near-memory/PIM to solve it.

## What it models (and why it matters on real GPUs)
- **Warps** — `WARP_SIZE = 32` lanes execute one instruction in lockstep (like NVIDIA warps).
- **Latency hiding / occupancy** — a single-issue, round-robin warp scheduler: while one
  warp stalls on a long memory op, another issues. So more resident warps *hide* memory
  latency. (Proven in `tests`: 2 warps cost ~1× the cycles of 1 warp, not 2×.)
- **Memory coalescing** — a memory instruction costs **one transaction per distinct
  memory segment** touched by the active lanes. Contiguous access → few transactions;
  strided/scattered → up to 32. (Proven: contiguous = 4 txns, strided = 32 txns.)
- **Divergence** — the per-lane active mask is in place; predicated branches (which
  serialize a warp) arrive in the next slice.

## The ISA (v1, deliberately minimal — D-015/OQ-6)
`mov rd,imm` · `tid rd` · `iadd rd,ra,rb` · `imul rd,ra,rb` · `ld rd,ra` (rd=mem[ra]) ·
`st ra,rb` (mem[ra]=rb) · `halt`. Word-addressed memory; 16 int32 registers per lane.

## Build & run ($0 — VS 2022 Build Tools: MSVC + bundled CMake + Ninja)
From a shell where `vcvars64.bat` has been sourced (see the repo's build note):
```
cmake -G Ninja -B build
cmake --build build
build\test_simt.exe                 # unit + cycle-accuracy + microarch tests
build\simt.exe kernels\vector_add.sasm 32
```
Expected: `all mini-GPU tests passed`, and vector-add reports `PASS`, `cycles: 681`,
`mem txns: 12`.

## Layout (clean engine/render split, like gem5)
```
include/simt/   isa.hpp  memory.hpp  core.hpp  assembler.hpp   (public headers)
src/            isa.cpp  assembler.cpp  core.cpp  main.cpp      (engine + CLI)
tests/          test_simt.cpp                                  (headless, $0 harness)
kernels/        vector_add.sasm                                (example kernels)
analysis/       (Python plots — the analysis layer, later)
```
