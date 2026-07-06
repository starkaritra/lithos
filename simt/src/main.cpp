// mini-GPU CLI: assemble a kernel, run it on the SIMT core, print cycle-accurate stats.
//
// Usage:
//   simt <kernel.sasm> [n_threads] [flags]
// Flags (for the Python analysis layer to sweep the timing model):
//   --json                 emit one machine-readable JSON line instead of the report
//   --mem-latency N        global-memory round-trip latency (cycles)   [default 200]
//   --mem-txn-penalty N    extra cycles per uncoalesced transaction     [default 8]
//   --segment-words N      coalescing granularity in words              [default 8]
//   --mem-words N          size of global memory in words               [default 256]
//
// Memory is seeded A[i]=i at base 0 and B[i]=2i at base 32 so the vector_add kernel
// (writing C at base 64) yields C[i]=3i. Harmless for other kernels.
#include "simt/assembler.hpp"
#include "simt/core.hpp"
#include "simt/memory.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Minimal flag parser: "--name value". Returns fallback if absent.
long get_flag(const std::vector<std::string>& args, const std::string& name, long fallback) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == name) return std::stol(args[i + 1]);
    return fallback;
}
bool has_flag(const std::vector<std::string>& args, const std::string& name) {
    for (const auto& a : args) if (a == name) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        std::cerr << "usage: simt <kernel.sasm> [n_threads] [--json] [--mem-latency N] ...\n";
        return 2;
    }

    // First two non-flag tokens are the kernel path and the thread count. Skip both a
    // "--flag" token and the value token that follows a value-flag, so a flag value is
    // never mistaken for a positional.
    static const std::vector<std::string> value_flags = {
        "--mem-latency", "--mem-txn-penalty", "--segment-words", "--mem-words"};
    std::string path;
    int n = 32;
    int positional = 0;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a.rfind("--", 0) == 0) {
            // If this is a value-flag, also skip its value.
            for (const auto& vf : value_flags)
                if (a == vf) { ++i; break; }
            continue;
        }
        if (positional == 0) path = a;
        else if (positional == 1) n = std::stoi(a);
        ++positional;
    }

    const bool json = has_flag(args, "--json");
    simt::CoreConfig cfg;
    cfg.mem_latency     = static_cast<int>(get_flag(args, "--mem-latency", cfg.mem_latency));
    cfg.mem_txn_penalty = static_cast<int>(get_flag(args, "--mem-txn-penalty", cfg.mem_txn_penalty));
    cfg.segment_words   = static_cast<int>(get_flag(args, "--segment-words", cfg.segment_words));
    const long mem_words = get_flag(args, "--mem-words", 256);

    std::ifstream f(path);
    if (!f) { std::cerr << "cannot open kernel: " << path << "\n"; return 2; }
    std::stringstream ss;
    ss << f.rdbuf();

    std::vector<simt::Instr> prog;
    try {
        prog = simt::assemble(ss.str());
    } catch (const std::exception& e) {
        std::cerr << "assembler error: " << e.what() << "\n";
        return 1;
    }

    simt::GlobalMemory mem(static_cast<std::size_t>(mem_words));
    for (int i = 0; i < n && (64 + i) < mem_words; ++i) {
        mem.store(0 + i, i);
        if (32 + i < mem_words) mem.store(32 + i, 2 * i);
    }

    simt::Core core(prog, mem, cfg);
    core.run(n);
    const auto& s = core.stats();
    const int n_warps = (n + simt::WARP_SIZE - 1) / simt::WARP_SIZE;

    if (json) {
        std::cout << "{"
                  << "\"n_threads\":" << n
                  << ",\"n_warps\":" << n_warps
                  << ",\"cycles\":" << s.cycles
                  << ",\"warp_instructions\":" << s.warp_instructions
                  << ",\"mem_ops\":" << s.mem_ops
                  << ",\"mem_transactions\":" << s.mem_transactions
                  << ",\"divergent_branches\":" << s.divergent_branches
                  << ",\"mem_latency\":" << cfg.mem_latency
                  << ",\"segment_words\":" << cfg.segment_words
                  << "}\n";
        return 0;
    }

    std::cout << "kernel        : " << path << "\n"
              << "threads       : " << n << " (" << n_warps << " warp(s) x "
              << simt::WARP_SIZE << " lanes)\n"
              << "cycles        : " << s.cycles << "\n"
              << "warp-instrs   : " << s.warp_instructions << "\n"
              << "mem ops       : " << s.mem_ops << "\n"
              << "mem txns      : " << s.mem_transactions << "  (coalescing: fewer is better)\n"
              << "divergences   : " << s.divergent_branches << "  (warps that split on a branch)\n";
    if (path.find("vector_add") != std::string::npos) {
        bool ok = true;
        for (int i = 0; i < n; ++i)
            if (mem.load(64 + i) != 3 * i) ok = false;
        std::cout << "result        : " << (ok ? "PASS (C[i] == 3*i)" : "FAIL") << "\n";
    } else if (path.find("reduction") != std::string::npos) {
        std::cout << "reduced sum   : mem[0] = " << mem.load(0)
                  << "   (sum of A[0..n) = " << (static_cast<long>(n) * (n - 1)) / 2 << ")\n";
    }
    return 0;
}
