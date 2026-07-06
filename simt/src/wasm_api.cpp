// WebAssembly entry point for the mini-GPU playground. Compiles (via Emscripten) the REAL
// simt_core engine to WASM so the browser runs the actual cycle-accurate simulator — no JS
// reimplementation, single source of truth with the native build + its tests.
//
// Exposes one C-linkage function `run_kernel` that assembles a kernel, runs it with tracing
// on, and returns a JSON string {stats, trace, error?} the frontend animates.
#include "simt/assembler.hpp"
#include "simt/core.hpp"
#include "simt/memory.hpp"

#include <algorithm>
#include <sstream>
#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace {

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

// Held static so the returned char* stays valid until the next call (cwrap 'string').
std::string g_result;

}  // namespace

extern "C" EMSCRIPTEN_KEEPALIVE
const char* run_kernel(const char* src, int n_threads, int mem_latency,
                       int segment_words, int mem_txn_penalty) {
    std::ostringstream out;
    std::vector<simt::Instr> prog;
    try {
        prog = simt::assemble(src ? std::string(src) : std::string());
    } catch (const std::exception& e) {
        out << "{\"error\":\"" << json_escape(e.what()) << "\"}";
        g_result = out.str();
        return g_result.c_str();
    }

    if (n_threads < 1) n_threads = 1;
    if (n_threads > 256) n_threads = 256;  // playground guardrail
    simt::CoreConfig cfg;
    if (mem_latency > 0) cfg.mem_latency = mem_latency;
    if (segment_words > 0) cfg.segment_words = segment_words;
    if (mem_txn_penalty >= 0) cfg.mem_txn_penalty = mem_txn_penalty;

    // Generous, fixed memory for the playground; seed A[i]=i, B[i]=2i (so vector_add works).
    const std::size_t words = 8192;
    simt::GlobalMemory mem(words);
    for (int i = 0; i < n_threads && (64 + i) < static_cast<int>(words); ++i) {
        mem.store(static_cast<std::size_t>(i), i);
        mem.store(static_cast<std::size_t>(32 + i), 2 * i);
    }

    simt::Core core(prog, mem, cfg);
    core.enable_trace();
    try {
        core.run(n_threads);
    } catch (const std::exception& e) {
        out << "{\"error\":\"runtime: " << json_escape(e.what())
            << " (check addresses stay in range)\"}";
        g_result = out.str();
        return g_result.c_str();
    }

    const auto& s = core.stats();
    const int n_warps = (n_threads + simt::WARP_SIZE - 1) / simt::WARP_SIZE;
    out << "{\"stats\":{"
        << "\"cycles\":" << s.cycles
        << ",\"warp_instructions\":" << s.warp_instructions
        << ",\"mem_ops\":" << s.mem_ops
        << ",\"mem_transactions\":" << s.mem_transactions
        << ",\"divergent_branches\":" << s.divergent_branches
        << ",\"n_threads\":" << n_threads
        << ",\"n_warps\":" << n_warps
        << ",\"warp_size\":" << simt::WARP_SIZE << "},\"trace\":[";
    const auto& tr = core.trace();
    for (std::size_t i = 0; i < tr.size(); ++i) {
        const auto& e = tr[i];
        if (i) out << ",";
        out << "{\"cycle\":" << e.cycle
            << ",\"warp\":" << e.warp_id
            << ",\"pc\":" << e.pc
            << ",\"op\":\"" << e.op << "\""
            << ",\"mask\":" << e.active_mask
            << ",\"latency\":" << e.latency
            << ",\"txns\":" << e.mem_txns
            << ",\"diverged\":" << (e.diverged ? "true" : "false")
            << ",\"stall\":" << (e.after_stall ? "true" : "false") << "}";
    }
    out << "]}";
    g_result = out.str();
    return g_result.c_str();
}
