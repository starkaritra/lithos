# References (Arm C)

A consolidated, real bibliography for the PIM / near-memory arm. Grouped by theme. Venues and years are
given so you can find each work; consult the original for exact page numbers.

## The memory wall & performance modelling
- **Wm. A. Wulf & S. A. McKee**, "Hitting the Memory Wall: Implications of the Obvious," *ACM SIGARCH
  Computer Architecture News*, 23(1), 1995. Coins "memory wall."
- **S. Williams, A. Waterman, D. Patterson**, "Roofline: An Insightful Visual Performance Model for
  Multicore Architectures," *Communications of the ACM*, 52(4), 2009. Compute-bound vs bandwidth-bound.
- **M. Horowitz**, "Computing's Energy Problem (and what we can do about it)," *ISSCC*, 2014. The
  data-movement energy numbers (~640 pJ per 32-bit DRAM access ⇒ ~160 pJ/byte).
- **J. L. Hennessy & D. A. Patterson**, *Computer Architecture: A Quantitative Approach*, 6th ed.,
  Morgan Kaufmann, 2017. §2 memory hierarchy / DRAM organization (banks, ranks, channels).

## Processing-in-Memory / near-memory computing
- **S. Ghose, A. Boroumand, J. S. Kim, J. Gómez-Luna, O. Mutlu**, "Processing-in-Memory: A
  Workload-Driven Perspective," *IBM Journal of R&D*, 63(6), 2019. Strong survey + memory-wall
  motivation; placement and bank-level concerns.
- **V. Seshadri et al.**, "Ambit: In-Memory Accelerator for Bulk Bitwise Operations Using Commodity DRAM
  Technology," *MICRO*, 2017. In-DRAM computation.
- **J. Gómez-Luna et al.**, "Benchmarking a New Paradigm: Experimental Analysis and Characterization of a
  Real Processing-in-Memory System" (UPMEM), 2021/2022. A real general-purpose PIM system.
- **Samsung HBM-PIM (Aquabolt-XL)** and **SK Hynix AiM** — commercial near-memory systems targeting
  memory-bound AI kernels (vendor whitepapers / ISSCC & HotChips presentations).

## The recommendation (DLRM) workload
- **M. Naumov et al.**, "Deep Learning Recommendation Model for Personalization and Recommendation
  Systems," arXiv:1906.00091, 2019. The embedding-bag sum-pooling workload; why it is memory-bound.
- **U. Gupta et al.**, "The Architectural Implications of Facebook's DNN-based Personalized
  Recommendation," *HPCA*, 2020. Embedding tables as the memory-bound bottleneck; pooling factors.

## Probability (the banking factor)
- **M. Mitzenmacher & E. Upfal**, *Probability and Computing*, 2nd ed., Cambridge, 2017. Occupancy /
  balls-in-bins / coupon-collector analysis behind `k(L,B) = B·(1−(1−1/B)^L)`.

## Scientific method (why the result is trustworthy)
- **B. A. Nosek et al.**, "The preregistration revolution," *PNAS*, 115(11), 2018.
- **J. R. Platt**, "Strong Inference," *Science*, 146(3642), 1964. Designing experiments to *exclude*
  rival hypotheses.
- **A. Gelman & E. Loken**, "The garden of forking paths," 2013. Why analyses must be fixed before data.

## Grove project documents (internal)
- `../../pim-prereg.md` — the Arm C pre-registration (hypotheses, DMR metric, byte accounting, sweep,
  GO/NO-GO rule).
- `../out/conclusion.md` — the CONDITIONAL-GO verdict, rival scorecard, and calibrated confidence.
- `../../decisions.md` — D-011 (parked PIM idea), D-015 (the A→C pivot), D-017 (Arm C scope), D-018
  (pre-registration), D-019 (the outcome).
- `../../simt/docs/09-memory-wall-measured.md` — the Arm-A measurement that motivates this arm.

→ Back to [index](README.md)
