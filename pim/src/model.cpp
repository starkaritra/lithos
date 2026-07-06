#include "pim/model.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

namespace pim {

int place(long long i, int B, const std::string& placement, long long domain_rows) {
    if (placement == "clustered") {
        // Contiguous chunks: rows [0..chunk) in bank 0, etc. With uniform-random indices
        // this spreads across banks much like round-robin; it differs only when accesses
        // are locality-correlated (swept for honesty, T9).
        const long long chunk = (domain_rows + B - 1) / B;
        int bank = static_cast<int>(i / std::max<long long>(chunk, 1));
        return std::min(bank, B - 1);
    }
    // round-robin (default, neutral placement).
    return static_cast<int>(i % B);
}

double closed_form_k(int L, int B) {
    if (B <= 0) return 0.0;
    return B * (1.0 - std::pow(1.0 - 1.0 / B, L));
}

namespace {

long long ceil_div(long long a, long long c) { return (a + c - 1) / c; }

Accounting finalize(Accounting a, const Params& p) {
    a.baseline_bytes = a.base_idx + a.base_operand + a.base_output;
    a.pim_bytes = a.pim_idx + a.pim_partial + a.pim_output;
    a.dmr = a.pim_bytes > 0 ? static_cast<double>(a.baseline_bytes) / a.pim_bytes : 0.0;
    a.pim_overhead_fraction =
        a.pim_bytes > 0
            ? static_cast<double>(a.pim_idx + a.pim_output) / a.pim_bytes
            : 0.0;
    a.baseline_cycles = ceil_div(a.baseline_bytes, std::max(p.cap, 1));
    a.pim_cycles = ceil_div(a.pim_bytes, std::max(p.cap, 1));
    a.baseline_energy_pj = static_cast<double>(a.baseline_bytes) * p.e;
    a.pim_energy_pj = static_cast<double>(a.pim_bytes) * p.e;
    a.energy_dmr = a.pim_energy_pj > 0 ? a.baseline_energy_pj / a.pim_energy_pj : 0.0;
    return a;
}

// Kernel B — array reduction (pim-prereg.md §3.1). Extreme-aggregation endpoint.
Accounting account_reduction(const Params& p) {
    Accounting a;
    a.base_idx = 0;                       // contiguous stream, no indices
    a.base_operand = p.N * p.b;           // read every element across the link once
    a.base_output = p.b;                  // write the scalar
    a.pim_idx = 0;
    a.pim_partial = static_cast<long long>(p.B) * p.b;  // one partial per bank
    a.pim_output = p.b;                   // host folds B partials -> 1 scalar
    a.k_empirical = p.B;                  // every bank participates
    a.k_closed = p.B;
    return finalize(a, p);
}

// Kernel A — embedding-bag sum-pooling (pim-prereg.md §3.2). The HEADLINE decision kernel.
Accounting account_embedding(const Params& p) {
    Accounting a;
    // Empirical banking factor: sample nb bags of L uniform-random rows (seeded) and
    // count distinct banks touched per bag -> total partials PIM must ship.
    std::mt19937_64 rng(p.seed);
    std::uniform_int_distribution<long long> pick(0, p.R_tab - 1);
    long long total_k = 0;
    std::unordered_set<int> banks;
    for (long long bag = 0; bag < p.nb; ++bag) {
        banks.clear();
        for (int l = 0; l < p.L; ++l)
            banks.insert(place(pick(rng), p.B, p.placement, p.R_tab));
        total_k += static_cast<long long>(banks.size());
    }
    a.k_empirical = static_cast<double>(total_k) / static_cast<double>(p.nb);
    a.k_closed = closed_form_k(p.L, p.B);

    const long long dxb = static_cast<long long>(p.d) * p.b;  // bytes per row-vector
    // Indices + output are charged to BOTH machines (G1 — no undercounting PIM).
    a.base_idx = p.nb * p.L * p.idx_b;
    a.base_operand = p.nb * static_cast<long long>(p.L) * dxb;  // gather all L rows
    a.base_output = p.nb * dxb;
    a.pim_idx = p.nb * p.L * p.idx_b;
    a.pim_partial = total_k * dxb;         // each touched bank ships ONE d-vector partial
    a.pim_output = p.nb * dxb;
    return finalize(a, p);
}

}  // namespace

Accounting account(const Params& p) {
    if (p.kernel == "reduction") return account_reduction(p);
    return account_embedding(p);
}

}  // namespace pim
