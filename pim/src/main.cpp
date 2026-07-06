// Arm C PIM byte-accounting CLI. Given a kernel + config, prints the off-chip link-byte
// accounting for the coalesced GPU baseline vs the PIM machine, as a JSON line the Python
// analysis layer collects. Usage (all flags optional; defaults = pim-prereg.md §3):
//   pim [--kernel embedding_bag|reduction] [--L n] [--B n] [--d n] [--idx_b n]
//       [--nb n] [--R_tab n] [--N n] [--cap n] [--e pj] [--placement round_robin|clustered]
//       [--seed n] [--json]
#include "pim/model.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

std::string sflag(const std::vector<std::string>& a, const std::string& name,
                  const std::string& fallback) {
    for (std::size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == name) return a[i + 1];
    return fallback;
}
long long lflag(const std::vector<std::string>& a, const std::string& name, long long fb) {
    for (std::size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == name) return std::stoll(a[i + 1]);
    return fb;
}
double dflag(const std::vector<std::string>& a, const std::string& name, double fb) {
    for (std::size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == name) return std::stod(a[i + 1]);
    return fb;
}
bool has(const std::vector<std::string>& a, const std::string& name) {
    for (const auto& s : a) if (s == name) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    pim::Params p;
    p.kernel = sflag(a, "--kernel", p.kernel);
    p.b = lflag(a, "--b", p.b);
    p.idx_b = lflag(a, "--idx_b", p.idx_b);
    p.B = static_cast<int>(lflag(a, "--B", p.B));
    p.d = static_cast<int>(lflag(a, "--d", p.d));
    p.L = static_cast<int>(lflag(a, "--L", p.L));
    p.nb = lflag(a, "--nb", p.nb);
    p.R_tab = lflag(a, "--R_tab", p.R_tab);
    p.N = lflag(a, "--N", p.N);
    p.cap = static_cast<int>(lflag(a, "--cap", p.cap));
    p.e = dflag(a, "--e", p.e);
    p.placement = sflag(a, "--placement", p.placement);
    p.seed = static_cast<unsigned>(lflag(a, "--seed", p.seed));

    const pim::Accounting r = pim::account(p);

    if (has(a, "--json")) {
        std::cout << "{"
                  << "\"kernel\":\"" << p.kernel << "\""
                  << ",\"L\":" << p.L << ",\"B\":" << p.B << ",\"d\":" << p.d
                  << ",\"idx_b\":" << p.idx_b << ",\"nb\":" << p.nb
                  << ",\"placement\":\"" << p.placement << "\""
                  << ",\"cap\":" << p.cap << ",\"e\":" << p.e
                  << ",\"baseline_bytes\":" << r.baseline_bytes
                  << ",\"pim_bytes\":" << r.pim_bytes
                  << ",\"base_idx\":" << r.base_idx
                  << ",\"base_operand\":" << r.base_operand
                  << ",\"base_output\":" << r.base_output
                  << ",\"pim_idx\":" << r.pim_idx
                  << ",\"pim_partial\":" << r.pim_partial
                  << ",\"pim_output\":" << r.pim_output
                  << ",\"k_empirical\":" << r.k_empirical
                  << ",\"k_closed\":" << r.k_closed
                  << ",\"dmr\":" << r.dmr
                  << ",\"pim_overhead_fraction\":" << r.pim_overhead_fraction
                  << ",\"baseline_cycles\":" << r.baseline_cycles
                  << ",\"pim_cycles\":" << r.pim_cycles
                  << ",\"baseline_energy_pj\":" << r.baseline_energy_pj
                  << ",\"pim_energy_pj\":" << r.pim_energy_pj
                  << ",\"energy_dmr\":" << r.energy_dmr
                  << "}\n";
        return 0;
    }

    std::cout << "kernel        : " << p.kernel << "\n"
              << "config        : L=" << p.L << " B=" << p.B << " d=" << p.d
              << " nb=" << p.nb << " placement=" << p.placement << "\n"
              << "baseline bytes: " << r.baseline_bytes << "\n"
              << "PIM bytes     : " << r.pim_bytes << "\n"
              << "k (empirical) : " << r.k_empirical << "  (closed-form " << r.k_closed << ")\n"
              << "DMR           : " << r.dmr << "x  (baseline / PIM off-chip bytes)\n"
              << "PIM overhead  : " << r.pim_overhead_fraction
              << "  (index+output share of PIM link traffic)\n";
    return 0;
}
