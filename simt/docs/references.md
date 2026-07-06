# References

A consolidated, real bibliography for the whole course. Grouped by theme. These are genuine,
citable sources; where an exact page or DOI matters, consult the work directly (venues and
years are given so you can find them).

## Foundational textbooks
- **J. L. Hennessy & D. A. Patterson**, *Computer Architecture: A Quantitative Approach*, 6th
  ed., Morgan Kaufmann, 2017. The standard graduate architecture text; Ch. 2 (memory
  hierarchy) and Ch. 4 (data-level parallelism, vector/GPU) underpin most of this course.
- **D. A. Patterson & J. L. Hennessy**, *Computer Organization and Design* (RISC-V ed.),
  Morgan Kaufmann. Gentler companion; excellent for ISA and pipelining fundamentals.

## The GPU / SIMT model
- **E. Lindholm, J. Nickolls, S. Oberman, J. Montrym**, "NVIDIA Tesla: A Unified Graphics and
  Computing Architecture," *IEEE Micro*, 28(2), 2008. Introduces the SIMT execution model.
- **J. Nickolls & W. J. Dally**, "The GPU Computing Era," *IEEE Micro*, 30(2), 2010.
- **NVIDIA**, *CUDA C++ Programming Guide*. Primary source for the thread hierarchy, SIMT
  architecture, memory model, and performance guidelines.
- **NVIDIA**, *Volta V100 Architecture Whitepaper*, 2017. Independent thread scheduling
  (per-thread PCs) — the modern refinement to the classic one-PC-per-warp model.

## The memory wall, latency hiding, coalescing
- **Wm. A. Wulf & S. A. McKee**, "Hitting the Memory Wall: Implications of the Obvious," *ACM
  SIGARCH Computer Architecture News*, 23(1), 1995. Coins "memory wall."
- **J. D. C. Little**, "A Proof for the Queuing Formula L = λW," *Operations Research*, 9(3),
  1961. Little's Law — the basis for latency hiding via concurrency.
- **V. Volkov**, "Understanding Latency Hiding on GPUs," Ph.D. thesis, UC Berkeley, 2016; and
  "Better Performance at Lower Occupancy," GTC 2010. The definitive treatment of occupancy vs
  instruction-level parallelism for hiding latency.
- **S. Williams, A. Waterman, D. Patterson**, "Roofline: An Insightful Visual Performance
  Model for Multicore Architectures," *Communications of the ACM*, 52(4), 2009.
- **NVIDIA**, *CUDA C++ Best Practices Guide* — "Coalesced Access to Global Memory."
- **M. Harris**, "How to Access Global Memory Efficiently in CUDA C/C++ Kernels," NVIDIA
  Developer Blog.

## Control flow / divergence
- **W. W. L. Fung, I. Sham, G. Yuan, T. M. Aamodt**, "Dynamic Warp Formation and Scheduling
  for Efficient GPU Control Flow," *MICRO-40*, 2007. Foundational divergence/reconvergence
  work.

## Flynn's taxonomy
- **M. J. Flynn**, "Some Computer Organizations and Their Effectiveness," *IEEE Transactions
  on Computers*, C-21(9), 1972.

## Simulation methodology & tools
- **N. Binkert et al.**, "The gem5 Simulator," *ACM SIGARCH Computer Architecture News*,
  39(2), 2011. (Our C++-core + Python-analysis split mirrors gem5 — decision D-016.)
- **A. Bakhoda, G. Yuan, W. W. L. Fung, H. Wong, T. M. Aamodt**, "Analyzing CUDA Workloads
  Using a Detailed GPU Simulator," *ISPASS*, 2009. GPGPU-Sim.
- **M. Khairy, Z. Shen, T. M. Aamodt, T. G. Rogers**, "Accel-Sim: An Extensible Simulation
  Framework for Validated GPU Modeling," *ISCA*, 2020.

## Instruction sets
- **NVIDIA**, *Parallel Thread Execution ISA* (PTX specification).
- **NVIDIA**, *CUDA Binary Utilities* (SASS, `cuobjdump`, `nvdisasm`).

## Processing-in-Memory (for Arm C, ahead)
- **S. Ghose, A. Boroumand, J. S. Kim, J. Gómez-Luna, O. Mutlu**, "Processing-in-Memory: A
  Workload-Driven Perspective," *IBM Journal of R&D*, 2019. A strong survey of PIM and the
  memory-wall motivation.
- **UPMEM** PIM architecture; **Samsung HBM-PIM** — real commercial near-memory systems.

## Lithos project documents (internal)
- `../../decisions.md` — architecture decision records (esp. D-006, D-014 the spike NO-GO,
  D-015 the A→C pivot, D-016 the C++/Python stack).
- `../../handoff.md` — project state and roadmap.
- `../../spike/out/conclusion.md` — the measured tree-inference NO-GO that motivates the
  divergence discussion in ch. 05.

→ Back to [index](README.md)
