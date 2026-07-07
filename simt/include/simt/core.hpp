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
#include "simt/trace.hpp"

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
    std::uint64_t divergent_branches = 0;   // branches where a warp's lanes disagreed
};

// A reconvergence-stack frame (Fung et al., MICRO 2007). When a warp diverges, the
// paths are pushed as frames and executed one at a time; a frame is popped when its
// pc reaches its reconvergence point (rpc), rejoining the lanes below it.
struct Frame {
    std::array<bool, WARP_SIZE> mask{};   // which lanes are active on this path
    int pc = 0;                           // next instruction for this path
    int rpc = -1;                         // reconvergence pc (-1 = none, base frame)
};

struct Warp {
    int id = 0;
    int pc = 0;                   // live (top-of-stack) program counter
    int rpc = -1;                 // live frame's reconvergence pc
    bool halted = false;
    std::uint64_t ready_at = 0;   // earliest cycle this warp may issue again
    std::array<bool, WARP_SIZE> active{};   // live (top-of-stack) active mask
    std::vector<Frame> stack;               // frames beneath the live one (divergence)
    // Registers are per-lane and PERSIST across divergence (a masked-off lane keeps
    // its state), so they live on the warp, not per-frame.
    std::array<std::array<std::int32_t, NREGS>, WARP_SIZE> regs{};
};

class Core {
public:
    Core(std::vector<Instr> program, GlobalMemory& mem, CoreConfig cfg = {});

    // Launch n_threads, partitioned into ceil(n_threads/WARP_SIZE) warps, and run
    // to completion. Lane l of warp w has global thread id w*WARP_SIZE + l.
    void run(int n_threads);

    const SimStats& stats() const { return stats_; }

    // Optional tracing (WASM playground animation). Off by default — enabling it does not
    // change the simulation or its cycle counts, only records what happened.
    void enable_trace() { tracing_ = true; }
    const std::vector<TraceEvent>& trace() const { return trace_; }

private:
    // Execute one instruction of warp w at the given cycle; returns the issue
    // latency (1 for ALU, memory-dependent for LD/ST).
    std::uint64_t execute(Warp& w);
    std::uint64_t memory_access(Warp& w, bool is_store);
    void branch(Warp& w);   // BRA: uniform redirect or divergent split via the stack

    std::vector<Instr> prog_;
    GlobalMemory& mem_;
    CoreConfig cfg_;
    std::vector<Warp> warps_;
    SimStats stats_;
    // Tracing state (only touched when tracing_ is true; zero effect otherwise).
    bool tracing_ = false;
    std::vector<TraceEvent> trace_;
    int last_txns_ = 0;          // memory transactions of the most recent memory op
    bool last_diverged_ = false; // whether the most recent BRA diverged
    bool pending_stall_ = false; // a clock-jump happened before the next issue
    // Per-lane capture for the trace's visualization fields (filled only when tracing_).
    std::array<std::int32_t, WARP_SIZE> last_lane_addr_{};
    std::array<std::int32_t, WARP_SIZE> last_lane_val_{};
};

}  // namespace simt
