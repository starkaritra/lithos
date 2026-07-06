// Behavioural tests for the Arm C PIM byte-accounting model (no framework, $0).
// These validate the accounting is CORRECT and the anti-tautology properties hold —
// never a pass/fail on the DMR threshold itself (that is experimentAS's GO/NO-GO call).
#include "pim/model.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {
int g_failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "FAIL: " << #cond << "  (" << __FILE__ << ":"    \
                      << __LINE__ << ")\n";                               \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

bool approx(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// closed-form banking factor: L=1 -> exactly 1 bank; large L -> saturates at B.
void test_closed_form_k() {
    CHECK(approx(pim::closed_form_k(1, 16), 1.0, 1e-9));
    CHECK(pim::closed_form_k(2, 16) > 1.0 && pim::closed_form_k(2, 16) < 2.0);
    CHECK(pim::closed_form_k(100000, 16) > 15.99);  // saturates at B
    CHECK(pim::closed_form_k(40, 16) < 16.0);       // never exceeds B
}

// placement: round-robin cycles mod B; clustered groups contiguous rows.
void test_placement() {
    CHECK(pim::place(0, 16, "round_robin", 1000) == 0);
    CHECK(pim::place(17, 16, "round_robin", 1000) == 1);
    CHECK(pim::place(0, 16, "clustered", 1600) == 0);
    CHECK(pim::place(1599, 16, "clustered", 1600) == 15);  // last chunk -> last bank
}

// Reduction (extreme aggregation): DMR ~= N/B, and the plumbing is correct.
void test_reduction_dmr() {
    pim::Params p;
    p.kernel = "reduction";
    p.N = 16000000; p.B = 16; p.b = 4;
    auto r = pim::account(p);
    CHECK(r.base_operand == p.N * p.b);
    CHECK(r.pim_partial == static_cast<long long>(p.B) * p.b);
    // DMR = (N*b + b) / (B*b + b) ~= N/B = 1,000,000
    CHECK(approx(r.dmr, double(p.N * p.b + p.b) / double(p.B * p.b + p.b), 1.0));
    CHECK(r.dmr > 900000.0);  // huge, as expected for the tautological corner
}

// Empirical banking factor should track the closed-form expectation (seeded, nb large).
void test_k_empirical_matches_closed() {
    pim::Params p;
    p.kernel = "embedding_bag";
    p.L = 40; p.B = 16; p.nb = 4096;
    auto r = pim::account(p);
    CHECK(approx(r.k_empirical, r.k_closed, 0.5));  // within half a bank
}

// Pure gather (L=1): every bag touches exactly one bank -> PIM sends a full row per bag,
// so DMR -> ~1. This is the CROSSOVER endpoint that proves it is not a tautology.
void test_embedding_gather_crossover() {
    pim::Params p;
    p.kernel = "embedding_bag";
    p.L = 1; p.B = 16; p.d = 64; p.nb = 1024;
    auto r = pim::account(p);
    CHECK(approx(r.k_empirical, 1.0, 1e-9));
    CHECK(r.dmr < 1.05);  // PIM barely helps for pure gather
}

// The honest DMR must be BELOW the naive tautological line (DMR_naive = L): banking (k)
// and hidden index/output traffic pull it down. This is the core anti-tautology property.
void test_dmr_below_naive() {
    pim::Params p;
    p.kernel = "embedding_bag";
    p.L = 40; p.B = 16; p.d = 64; p.nb = 1024;
    auto r = pim::account(p);
    CHECK(r.dmr < static_cast<double>(p.L));   // honest < naive L
    CHECK(r.dmr > 1.0);                        // but PIM still helps at high pooling
    // approx L/k for large d (indices/output are small next to d*b operands)
    CHECK(r.dmr < static_cast<double>(p.L) / (r.k_empirical - 0.5));
}

// DMR is monotincreasing in L (more pooling -> more co-banked rows collapse).
void test_dmr_monotonic_in_L() {
    auto dmr_at = [](int L) {
        pim::Params p; p.kernel = "embedding_bag"; p.L = L; p.B = 16; p.nb = 1024;
        return pim::account(p).dmr;
    };
    CHECK(dmr_at(4) < dmr_at(16));
    CHECK(dmr_at(16) < dmr_at(64));
}

// Hidden traffic (index + output) is counted and reported for PIM (G1).
void test_overhead_counted() {
    pim::Params p;
    p.kernel = "embedding_bag";
    p.L = 40; p.B = 16; p.d = 16;  // small d -> indices/output are a larger share
    auto r = pim::account(p);
    CHECK(r.pim_idx > 0);
    CHECK(r.pim_output > 0);
    CHECK(r.pim_overhead_fraction > 0.0 && r.pim_overhead_fraction < 1.0);
}

}  // namespace

int main() {
    test_closed_form_k();
    test_placement();
    test_reduction_dmr();
    test_k_empirical_matches_closed();
    test_embedding_gather_crossover();
    test_dmr_below_naive();
    test_dmr_monotonic_in_L();
    test_overhead_counted();

    if (g_failures == 0) {
        std::cout << "all PIM model tests passed\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
}
