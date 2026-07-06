#pragma once
// Grove Arm C — near-memory / PIM data-movement model (pim-prereg.md §3).
//
// This is a byte-accounting model, NOT silicon and not cycle-accurate: for a given
// kernel + config it COUNTS the off-chip link bytes moved by (a) a competent coalesced
// GPU baseline and (b) a PIM machine that aggregates in the memory banks. The primary
// result is DMR = baseline_link_bytes / pim_link_bytes (assumption-free: bytes are
// counted, not modelled). See pim-prereg.md for the full pre-registration.
#include <cstdint>
#include <string>

namespace pim {

// All parameters are pre-registered (pim-prereg.md §3 defaults, §6 sweep).
struct Params {
    std::string kernel = "embedding_bag";  // "reduction" | "embedding_bag"
    long long b = 4;         // bytes per data element (fp32)
    long long idx_b = 4;     // bytes per index
    int B = 16;              // number of banks
    int d = 64;              // embedding dimension (elements per row)
    int L = 40;              // rows per bag (pooling factor)
    long long nb = 1024;     // number of bags (batch)
    long long R_tab = 4000000;    // rows in the embedding table
    long long N = 16000000;       // array length (reduction kernel)
    int cap = 32;            // off-chip link bytes/cycle (shared by both machines)
    double e = 160.0;        // data-movement energy per off-chip byte (pJ)
    std::string placement = "round_robin";  // "round_robin" | "clustered"
    unsigned seed = 42;
};

// The full auditable byte breakdown for one config (pim-prereg.md §3/G1).
struct Accounting {
    // Baseline (coalesced GPU floor): indices + all operands + output.
    long long base_idx = 0;
    long long base_operand = 0;
    long long base_output = 0;
    long long baseline_bytes = 0;
    // PIM: indices + k pre-summed partials + output.
    long long pim_idx = 0;
    long long pim_partial = 0;
    long long pim_output = 0;
    long long pim_bytes = 0;
    // Banking factor (distinct banks a bag touches -> partials PIM must ship).
    double k_empirical = 0.0;   // measured from seeded sampled indices
    double k_closed = 0.0;      // closed-form expectation B*(1-(1-1/B)^L)
    // Primary + guardrail quantities.
    double dmr = 0.0;                    // baseline_bytes / pim_bytes (the OEC)
    double pim_overhead_fraction = 0.0;  // (index + output) / pim_bytes (hidden traffic)
    long long baseline_cycles = 0;       // ceil(bytes / cap)  [secondary]
    long long pim_cycles = 0;
    double baseline_energy_pj = 0.0;     // bytes * e           [secondary]
    double pim_energy_pj = 0.0;
    double energy_dmr = 0.0;             // = dmr (e cancels); reported for honesty
};

// Bank placement: which bank holds table row / element i (pim-prereg.md §3).
int place(long long i, int B, const std::string& placement, long long domain_rows);

// Closed-form expected distinct banks touched by L uniform-random rows (coupon-collector
// occupancy): B*(1 - (1-1/B)^L).
double closed_form_k(int L, int B);

// Compute the full byte accounting for the given config.
Accounting account(const Params& p);

}  // namespace pim
