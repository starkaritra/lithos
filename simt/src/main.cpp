// mini-GPU CLI: assemble a kernel, run vector-add on the SIMT core, verify the
// result, and print the cycle-accurate stats. Usage:
//   simt <kernel.sasm> [n_threads]
// The vector-add scenario uses the memory layout baked into the kernel:
//   A at word base 0, B at base 32, C at base 64 (so n<=32 for the v1 spine).
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

    // Layout: A[0..n), B[32..32+n), C[64..64+n). Seed A[i]=i, B[i]=2i => C[i]=3i.
    const std::size_t baseA = 0, baseB = 32, baseC = 64;
    simt::GlobalMemory mem(baseC + static_cast<std::size_t>(n) + 8);
    for (int i = 0; i < n; ++i) {
        mem.store(baseA + i, i);
        mem.store(baseB + i, 2 * i);
    }

    simt::Core core(prog, mem);
    core.run(n);

    bool ok = true;
    for (int i = 0; i < n; ++i)
        if (mem.load(baseC + i) != 3 * i) ok = false;

    const auto& s = core.stats();
    std::cout << "kernel        : " << path << "\n"
              << "threads       : " << n << " (" << (n + simt::WARP_SIZE - 1) / simt::WARP_SIZE
              << " warp(s) x " << simt::WARP_SIZE << " lanes)\n"
              << "result        : " << (ok ? "PASS (C[i] == 3*i)" : "FAIL") << "\n"
              << "cycles        : " << s.cycles << "\n"
              << "warp-instrs   : " << s.warp_instructions << "\n"
              << "mem ops       : " << s.mem_ops << "\n"
              << "mem txns      : " << s.mem_transactions
              << "  (coalescing: fewer is better)\n";
    return ok ? 0 : 1;
}
