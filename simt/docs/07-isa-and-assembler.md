# 07 — The ISA & Assembler

> **Goal:** understand how a kernel becomes instructions a machine can run — the
> **Instruction Set Architecture (ISA)** — why ours is deliberately tiny, and how it relates
> to real GPU instruction sets (PTX and SASS).

---

## 1. What an ISA is, and why it's the most important contract in a computer

An **ISA** is the contract between software and hardware: the exact set of instructions the
machine understands, what each does, how they're encoded, and what state (registers, memory)
they operate on. It is the *boundary* that lets compilers and chips evolve independently — a
program compiled to an ISA runs on any hardware implementing it.

Designing an ISA means answering: What operations are primitive? How many registers? How is
memory addressed? How is control flow expressed? These choices ripple through the whole
system. Our project makes them from scratch, which is the best way to feel *why* real ISAs
look the way they do.

---

## 2. Our ISA (v1): minimal on purpose

We chose the smallest instruction set that can express real parallel kernels. Every
instruction operates **per active lane** (chapter 02).

| Instruction | Meaning | Notes |
|-------------|---------|-------|
| `mov rd, imm` | `rd = imm` | load an immediate constant |
| `tid rd` | `rd = global thread id` | `warp_id × 32 + lane` — breaks lane symmetry |
| `iadd rd, ra, rb` | `rd = ra + rb` | integer add (also used for addressing) |
| `imul rd, ra, rb` | `rd = ra × rb` | integer multiply (strides, matmul) |
| `slt rd, ra, rb` | `rd = (ra < rb) ? 1 : 0` | set-less-than → per-lane **predicate** |
| `ld rd, ra` | `rd = mem[ra]` | word-addressed load (a memory op) |
| `st ra, rb` | `mem[ra] = rb` | word-addressed store (a memory op) |
| `jmp label` | `pc = label` | unconditional jump (all active lanes) |
| `bra rp, else, join` | predicated branch | lanes with `rp≠0` → then; `rp=0` → `else`; reconverge at `join` (chapter 05) |
| `halt` | stop this warp | — |

State: **16 private int32 registers per lane** (`r0..r15`, `NREGS`), and a flat,
**word-addressed** global memory of int32 words. Control flow uses **labels** (a lone
`name:` line) resolved by the assembler. That's it — no floats, no stack, no function
calls — because these ten ops are enough to teach warps, latency hiding, coalescing, and
divergence, and leaving the rest out keeps the whole machine readable.

> **Design principle in action (KISS + YAGNI).** We add instructions only when a kernel we
> actually want to run needs them. `imul` exists because strided/matmul addressing needs it;
> `slt`/`bra`/`jmp` arrived with the divergence slice (chapter 05) because that's when
> control flow first mattered. We are not speculatively building a general CPU.

Why **word-addressed** (address = word index, not byte index)? It removes byte/word
conversion noise from every example so the interesting arithmetic (addresses, strides,
coalescing segments) stays front-and-center. Real ISAs are byte-addressed; that's a
refinement, not a concept.

---

## 3. From kernel text to instructions: the assembler

An **assembler** translates human-readable assembly text into the instruction structs the
simulator executes. Ours lives in `src/assembler.cpp` (`simt::assemble`). It is intentionally
simple: one instruction per line, `#` or `;` begins a comment, operands separated by spaces
or commas, registers written `rN`.

```
   TEXT (kernels/vector_add.sasm)        →   assemble()   →   std::vector<Instr>
   ┌────────────────────────────┐                            ┌──────────────────────┐
   │ tid  r0                     │                            │ {op=TID, rd=0}        │
   │ iadd r4, r1, r0             │                            │ {op=IADD, rd=4,       │
   │ ld   r7, r4                 │                            │  ra=1, rb=0}          │
   │ ...                         │                            │ {op=LD, rd=7, ra=4}   │
   └────────────────────────────┘                            │ ...                   │
                                                              └──────────────────────┘
```

The assembler also **validates**: unknown mnemonics, out-of-range registers, and
too-few-operands all raise errors rather than silently miscompiling (tested in
`test_assembler_errors`). A tool that fails loudly is a tool you can trust — a small but real
engineering-quality point.

`Instr` itself (`include/simt/isa.hpp`) is a plain struct — opcode plus operand fields:

```cpp
struct Instr {
    Op op;                 // which operation
    int rd, ra, rb;        // destination + source register indices
    int32_t imm;           // immediate (for mov)
};
```

Real hardware encodes instructions as **bit-packed binary words** (e.g. 32 or 64 bits with
fields for opcode/registers/immediate). We keep a decoded struct because the goal is
*understanding the semantics and timing*, not byte-level encoding. Adding a binary encoder is
a clean, optional exercise.

---

## 4. Worked example: how vector-add maps to the ISA

`C[i] = A[i] + B[i]`, with A at word base 0, B at 32, C at 64 (see
`kernels/vector_add.sasm`):

```
  mov  r1, 0        # r1 = baseA
  mov  r2, 32       # r2 = baseB
  mov  r3, 64       # r3 = baseC
  tid  r0           # r0 = i   (this lane's element)
  iadd r4, r1, r0   # r4 = baseA + i   = &A[i]
  iadd r5, r2, r0   # r5 = baseB + i   = &B[i]
  iadd r6, r3, r0   # r6 = baseC + i   = &C[i]
  ld   r7, r4       # r7 = A[i]
  ld   r8, r5       # r8 = B[i]
  iadd r9, r7, r8   # r9 = A[i] + B[i]
  st   r6, r9       # C[i] = r9
  halt
```

Every lane runs *these same 12 instructions*, but because `tid` gave each lane a different
`r0`, each computes a different address and element. That is SIMT (chapter 02) expressed in
the ISA: **uniform code, per-lane data.** The addresses `baseA + i` for consecutive lanes are
consecutive words — which is exactly why this kernel is well-**coalesced** (chapter 04).

---

## 5. On real GPUs: PTX and SASS

NVIDIA GPUs actually expose **two** instruction sets, which is worth knowing:

- **PTX** (Parallel Thread Execution) — a *virtual* ISA / intermediate representation. CUDA
  compiles to PTX, which is stable across GPU generations. It has virtual registers and
  higher-level ops. Think of it as portable GPU assembly.
- **SASS** (Streaming ASSembler) — the *real* machine code for a specific GPU architecture.
  PTX is compiled (by `ptxas`) down to SASS, which has real register limits, scheduling, and
  the encoding the silicon runs.

Our tiny ISA is closest in spirit to a stripped-down PTX: symbolic, register-based,
architecture-independent, meant for humans and a simulator. The two-level PTX/SASS split
exists for portability — the same reason ISAs exist at all (§1).

> **Reality check.** Real GPU ISAs have hundreds of instructions (floating point, transcend-
> entals, atomics, tensor-core ops, predication, barriers, memory-space qualifiers). Ours has
> seven. The seven are enough to *teach the architecture*; the rest are breadth, not new
> concepts.

---

## Check your understanding
1. What is an ISA, and why does it let hardware and compilers evolve independently?
2. In the vector-add listing, which single instruction makes each lane work on a different
   element, and how?
3. What's the difference between PTX and SASS, and which is our ISA more like?

---

## References
- NVIDIA, *Parallel Thread Execution ISA* (the PTX specification).
- NVIDIA, *CUDA Binary Utilities* (documents SASS and `cuobjdump`/`nvdisasm`).
- D. Patterson & J. Hennessy, *Computer Organization and Design* (the RISC-V edition is an
  excellent, gentle ISA-design companion).
- Grove code: `include/simt/isa.hpp`, `src/assembler.cpp`, `kernels/vector_add.sasm`.

→ Previous: [06 — Cycle-Accurate Simulation](06-cycle-accurate-simulation.md) · Back to [index](README.md)
