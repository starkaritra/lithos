# Glossary

Every term used in this course, defined once in plain words. Ordered roughly from
foundational to specific.

**Latency** — the time (often in cycles) for *one* operation to finish, start to end. CPUs
optimize for low latency.

**Throughput** — the amount of work completed per unit time (operations/second). GPUs
optimize for high throughput. (See [ch. 01](01-why-gpus.md).)

**Cycle** — one tick of the processor clock; the basic unit of time in these docs.

**Memory wall** — the large and growing gap between how fast processors compute and how fast
memory can supply data; processors spend much of their time *waiting* for memory. The central
problem GPUs (and Lithos's Arm C) are built to fight. (Wulf & McKee, 1995.)

**DRAM** — the large, slow, off-chip main memory. An access costs hundreds of cycles.

**SRAM** — small, fast, on-chip memory (registers, caches, shared memory). ~10× cheaper to
access than DRAM.

**Flynn's taxonomy** — a classification of machines by instruction/data streams: SISD, SIMD,
MISD, MIMD. (See [ch. 01](01-why-gpus.md).)

**SIMD** (Single Instruction, Multiple Data) — one instruction operates on many data elements
simultaneously (e.g. CPU vector units like AVX).

**SIMT** (Single Instruction, Multiple Thread) — NVIDIA's model: you write per-thread code;
the hardware runs threads in lockstep groups (warps) on SIMD lanes. Programmability of
threads, efficiency of vectors. (See [ch. 02](02-simt-execution-model.md).)

**SPMD** (Single Program, Multiple Data) — the *programming* model: every thread runs the same
program on different data. Executed as SIMT by the hardware.

**Thread / lane** — one instance of the kernel, handling one data element; has its own
registers. A "lane" is the hardware slot a thread runs in.

**Warp** — a fixed bundle of **32 threads** that execute the same instruction in the same
cycle (lockstep). The true unit of scheduling. (AMD calls its 64-thread version a
"wavefront.")

**Lockstep** — all lanes of a warp share **one program counter** and execute the same
instruction together, differing only in their register data.

**Active mask** — one bit per lane marking which lanes are "real" / doing useful work.
Partial warps and branch divergence turn bits off. Work done by inactive lanes is wasted.

**Block** — a group of warps that can share on-chip "shared memory" and synchronize. (Context;
not in our v1.)

**Grid** — all threads/blocks of one kernel launch.

**Kernel** — the program each thread runs (e.g. `vector_add`).

**Occupancy** — resident warps ÷ maximum possible resident warps. Higher occupancy provides
more independent work to hide latency. (See [ch. 03](03-latency-hiding-occupancy.md).)

**Latency hiding** — keeping the ALUs busy by switching to other ready warps while one warp
waits on memory, so slow-memory latency is overlapped with useful work.

**Little's Law** — `concurrency = latency × throughput`; to hide a latency `L` at issue rate
1/cycle you need ~`L` operations in flight. The theoretical basis for why GPUs run so many
threads. (Little, 1961.)

**Memory transaction** — one chunk-sized memory access (a segment / cache line). You pay per
transaction, not per byte.

**Segment / cache line** — the fixed-size chunk memory is delivered in (our model: 8 words =
32 bytes). Accessing 1 word or a whole segment costs about the same.

**Coalescing** — neighboring threads accessing neighboring memory so their accesses fuse into
few transactions. The #1 practical GPU optimization. Uncoalesced (scattered) access can cost
8–32× more. (See [ch. 04](04-memory-coalescing.md).)

**Roofline model** — a visual model classifying a kernel as compute-bound or bandwidth-bound.
Uncoalesced access pushes you toward the bandwidth roof (the memory wall). (Williams et al.,
2009.)

**Arithmetic intensity** — compute operations performed per byte (or per access) of data
moved. Low intensity → memory-bound (left of the roofline knee); high intensity → compute-bound.
Measured directly in [ch. 09](09-memory-wall-measured.md).

**Compute-bound / bandwidth-bound** — a kernel is compute-bound if arithmetic throughput limits
it, bandwidth-bound (memory-bound) if data movement limits it. Most memory-heavy kernels
(gathers, sparse ops) are bandwidth-bound — the case PIM (Arm C) targets.

**Branch divergence** — when lanes in a warp take different branches of an `if`, the warp must
execute both paths *serially* (masking off the other lanes each time), wasting throughput.
The reason GPUs suit regular workloads and struggle with irregular/branchy ones. (See
[ch. 05](05-branch-divergence.md).)

**Reconvergence** — the point after a divergent branch where masked-off lanes rejoin and the
warp continues in lockstep again; tracked with a reconvergence stack.

**Predicate** — a per-lane true/false bit (e.g. from a comparison) used to mask execution,
including for branches.

**ISA** (Instruction Set Architecture) — the contract of instructions, encodings, registers,
and memory model the hardware exposes to software. (See [ch. 07](07-isa-and-assembler.md).)

**Assembler** — the tool that turns assembly text into machine instructions.

**PTX** — NVIDIA's *virtual*, portable GPU instruction set / IR (CUDA compiles to it).

**SASS** — NVIDIA's *real* per-architecture GPU machine code (PTX compiles down to it).

**Functional simulation** — models *correct results* but not timing.

**Cycle-accurate simulation** — models *how many cycles* execution takes under a timing model.
(Ours is more precisely "cycle-approximate": deterministic but not silicon-validated.)

**RTL** (Register-Transfer Level) — a hardware description (e.g. Verilog) of the actual
circuit; the most detailed, slowest simulation level.

**gem5 / GPGPU-Sim / Accel-Sim** — standard research architecture simulators; gem5's C++
core + Python scripting is the pattern our project mirrors (decision D-016).

**PIM / near-memory computing** — Processing-In-Memory: moving compute *into or beside* memory
to beat the memory wall. The subject of Lithos's **Arm C** (the "solve" to Arm A's "expose").

**Reduction** — combining many values into one (sum, max, count…). Done as a **tree** in
`log₂(n)` steps to expose parallelism; requires cross-thread communication. (See
[ch. 08](08-reduction-and-communication.md).)

**Warp-synchronous** — relying on a warp's 32 lanes running in lockstep, so instructions are
implicitly ordered/synchronized *within* a warp (no explicit barrier needed).

**Barrier / `__syncthreads()`** — an explicit point where all threads in a block wait for one
another; needed for correct cross-*warp* communication (warps otherwise run asynchronously).

**Shared memory** — a small, fast, on-chip scratchpad shared by a block's warps; the usual
medium for cross-thread communication in reductions and tiled algorithms. (A future slice.)

→ Back to [index](README.md)
