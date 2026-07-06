// mini-GPU CLI: assemble a kernel, run it on the SIMT core, print cycle-accurate stats.
// Usage:
//   simt <kernel.sasm> [n_threads]
// Memory is seeded as A[i]=i at word base 0 and B[i]=2i at base 32 (so the vector_add
// kernel, which writes C at base 64, produces C[i]=3i). For other kernels the seeding is
// harmless; the stats and a small output dump are printed so you can see what happened.
#include "simt/assembler.hpp"
#include "simt/core.hpp"
#include "simt/memory.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <kernel.sasm> [n_threads]\n";
        return 2;
    }
    const std::string path = argv[1];
    const int n = argc >= 3 ? std::stoi(argv[2]) : 32;

    std::ifstream f(path);
    if (!f) {
        std::cerr << "cannot open kernel: " << path << "\n";
        return 2;
    }
    std::stringstream ss;
    ss << f.rdbuf();

    std::vector<simt::Instr> prog;
    try {
        prog = simt::assemble(ss.str());
    } catch (const std::exception& e) {
        std::cerr << "assembler error: " << e.what() << "\n";
        return 1;
    }

    simt::GlobalMemory mem(256);
    for (int i = 0; i < n; ++i) {
        mem.store(0 + i, i);       // A[i] = i
        mem.store(32 + i, 2 * i);  // B[i] = 2i
    }

    simt::Core core(prog, mem);
    core.run(n);

    const auto& s = core.stats();
    const int n_warps = (n + simt::WARP_SIZE - 1) / simt::WARP_SIZE;
    std::cout << "kernel        : " << path << "\n"
              << "threads       : " << n << " (" << n_warps << " warp(s) x "
              << simt::WARP_SIZE << " lanes)\n"
              << "cycles        : " << s.cycles << "\n"
              << "warp-instrs   : " << s.warp_instructions << "\n"
              << "mem ops       : " << s.mem_ops << "\n"
              << "mem txns      : " << s.mem_transactions << "  (coalescing: fewer is better)\n"
              << "divergences   : " << s.divergent_branches << "  (warps that split on a branch)\n";

    // Kernel-specific readout so you can SEE the result.
    if (path.find("vector_add") != std::string::npos) {
        bool ok = true;
        for (int i = 0; i < n; ++i)
            if (mem.load(64 + i) != 3 * i) ok = false;
        std::cout << "result        : " << (ok ? "PASS (C[i] == 3*i)" : "FAIL") << "\n";
    } else if (path.find("divergence") != std::string::npos) {
        std::cout << "output C[0..7]: ";
        for (int i = 0; i < 8 && i < n; ++i) std::cout << mem.load(i) << " ";
        std::cout << "\n(lanes < 16 took 'then' = 100+i; the rest took 'else' = 200+i)\n";
    }
    return 0;
}
