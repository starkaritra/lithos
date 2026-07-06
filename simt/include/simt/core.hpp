#pragma once
// The SIMT core: warps of WARP_SIZE lanes execute one instruction in lockstep.
// A single-issue, round-robin warp scheduler drives a cycle-accurate model whose
// whole point is to make GPU microarchitecture effects MEASURABLE:
//   - latency hiding: while one warp stalls on a long memory op, another issues,
//     so many resident warps overlap memory latency (occupancy).
//   - coalescing: a memory instruction costs 1 transaction per distinct memory
//     segment touched by the warp's active lanes -> contiguous access is cheap,
//     scattered access is expensive.
// Divergence (predicated branches) plugs into the active-mask in a later slice.
#include "simt/isa.hpp"
#include "simt/memory.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace simt {

constexpr int WARP_SIZE = 32;   // lanes per warp (as on real NVIDIA GPUs)
constexpr int NREGS = 16;       // integer registers per lane

struct CoreConfig {
    int mem_latency = 200;      // cycles for a global-memory round trip
    int mem_txn_penalty = 8;    // extra cycles per additional (uncoalesced) transaction
    int segment_words = 8;      // coalescing granularity (8 words = 32 B segment)
};

struct SimStats {
    std::uint64_t cycles = 0;               // completion cycle of the last instruction
    std::uint64_t warp_instructions = 0;    // warp-level instructions issued
    std::uint64_t mem_transactions = 0;     // total memory transactions (coalescing metric)
    std::uint64_t mem_ops = 0;              // memory instructions issued (warp-level)
};

struct Warp {
    int id = 0;
    int pc = 0;
    bool halted = false;
    std::uint64_t ready_at = 0;   // earliest cycle this warp may issue again
    std::array<bool, WARP_SIZE> active{};
    std::array<std::array<std::int32_t, NREGS>, WARP_SIZE> regs{};
};

class Core {
public:
    Core(std::vector<Instr> program, GlobalMemory& mem, CoreConfig cfg = {});

    // Launch n_threads, partitioned into ceil(n_threads/WARP_SIZE) warps, and run
    // to completion. Lane l of warp w has global thread id w*WARP_SIZE + l.
    void run(int n_threads);

    const SimStats& stats() const { return stats_; }

private:
    // Execute one instruction of warp w at the given cycle; returns the issue
    // latency (1 for ALU, memory-dependent for LD/ST).
    std::uint64_t execute(Warp& w);
    std::uint64_t memory_access(Warp& w, bool is_store);

    std::vector<Instr> prog_;
    GlobalMemory& mem_;
    CoreConfig cfg_;
    std::vector<Warp> warps_;
    SimStats stats_;
};

}  // namespace simt
