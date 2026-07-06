// Behavioural + cycle-accuracy tests for the mini-GPU (no external framework, $0).
// These assert functional correctness AND the exact deterministic cycle model, plus
// the two microarchitecture phenomena v1 is built to demonstrate: memory coalescing
// and latency hiding via multiple resident warps.
#include "simt/assembler.hpp"
#include "simt/core.hpp"
#include "simt/memory.hpp"

#include <cstdint>
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

const char* kVectorAdd = R"(
mov  r1, 0
mov  r2, 32
mov  r3, 64
tid  r0
iadd r4, r1, r0
iadd r5, r2, r0
iadd r6, r3, r0
ld   r7, r4
ld   r8, r5
iadd r9, r7, r8
st   r6, r9
halt
)";

// One warp of vector-add: correct result, exact stats (deterministic model).
void test_vector_add_one_warp() {
    auto prog = simt::assemble(kVectorAdd);
    CHECK(prog.size() == 12);

    simt::GlobalMemory mem(128);
    for (int i = 0; i < 32; ++i) {
        mem.store(0 + i, i);       // A[i] = i
        mem.store(32 + i, 2 * i);  // B[i] = 2i
    }
    simt::Core core(prog, mem);
    core.run(32);

    for (int i = 0; i < 32; ++i) CHECK(mem.load(64 + i) == 3 * i);

    const auto& s = core.stats();
    CHECK(s.warp_instructions == 12);
    CHECK(s.mem_ops == 3);
    // 3 memory ops, each 32 contiguous words over 8-word segments -> 4 segments each.
    CHECK(s.mem_transactions == 12);
    // Exact hand-derived cycle count for the single-warp schedule (see decisions/tests).
    CHECK(s.cycles == 681);
}

// TID must yield the global thread id: warp w, lane l -> w*32 + l.
void test_tid_global() {
    // Store each lane's tid to mem[tid], then verify.
    auto prog = simt::assemble("tid r0\nst r0, r0\nhalt\n");
    simt::GlobalMemory mem(64);
    simt::Core core(prog, mem);
    core.run(64);
    for (int i = 0; i < 64; ++i) CHECK(mem.load(i) == i);
}

// Coalescing: contiguous access touches few segments; strided access touches many.
void test_coalescing() {
    // Contiguous: addr = tid  -> 32 lanes over words 0..31 -> 4 segments.
    auto contig = simt::assemble("tid r0\nld r1, r0\nhalt\n");
    simt::GlobalMemory m1(64);
    simt::Core c1(contig, m1);
    c1.run(32);
    CHECK(c1.stats().mem_transactions == 4);

    // Strided: addr = tid*8 -> every lane in its own segment -> 32 transactions.
    auto strided = simt::assemble("tid r0\nmov r2, 8\nimul r3, r0, r2\nld r1, r3\nhalt\n");
    simt::GlobalMemory m2(300);
    simt::Core c2(strided, m2);
    c2.run(32);
    CHECK(c2.stats().mem_transactions == 32);

    // The uncoalesced access must cost more cycles (bandwidth penalty).
    CHECK(c2.stats().cycles > c1.stats().cycles);
}

// Latency hiding: a 2nd resident warp issues while the 1st stalls on memory, so
// doubling the work costs almost nothing -> occupancy hides memory latency.
void test_latency_hiding() {
    auto prog = simt::assemble("tid r0\nld r1, r0\nld r2, r0\nhalt\n");

    simt::GlobalMemory m1(128);
    simt::Core c1(prog, m1);
    c1.run(32);  // 1 warp

    simt::GlobalMemory m2(128);
    simt::Core c2(prog, m2);
    c2.run(64);  // 2 warps, double the work

    const std::uint64_t one = c1.stats().cycles;
    const std::uint64_t two = c2.stats().cycles;
    // If latency were NOT hidden, two warps would cost ~2x. Strong hiding => ~1x.
    CHECK(two < one + one / 5);  // < 1.2x single-warp cycles
}

// Assembler rejects malformed input rather than silently miscompiling.
void test_assembler_errors() {
    bool threw = false;
    try {
        simt::assemble("bogus r0\n");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        simt::assemble("iadd r0, r1\n");  // too few operands
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        simt::assemble("jmp nowhere\n");  // undefined label
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

// A structured if/else kernel: if (tid < threshold) C[tid]=100+tid else C[tid]=200+tid.
std::string divergence_kernel(int threshold) {
    return
        "mov  r5, " + std::to_string(threshold) + "\n"
        "tid  r0\n"
        "slt  r1, r0, r5\n"     // predicate = (tid < threshold)
        "bra  r1, else, join\n"
        "mov  r2, 100\n"        // then-block
        "iadd r3, r2, r0\n"
        "st   r0, r3\n"
        "jmp  done\n"
        "else:\n"
        "mov  r2, 200\n"        // else-block
        "iadd r3, r2, r0\n"
        "st   r0, r3\n"
        "join:\n"
        "done:\n"
        "halt\n";
}

// Branch divergence: a warp that disagrees on an if runs BOTH paths serially; a warp
// that agrees runs only one. Both must be functionally correct.
void test_divergence() {
    // Divergent: threshold 16 splits the 32-lane warp (lanes 0..15 vs 16..31).
    auto div = simt::assemble(divergence_kernel(16));
    simt::GlobalMemory md(64);
    simt::Core cd(div, md);
    cd.run(32);
    for (int i = 0; i < 32; ++i)
        CHECK(md.load(i) == (i < 16 ? 100 + i : 200 + i));  // masked exec is correct
    CHECK(cd.stats().divergent_branches == 1);

    // Uniform: threshold 64 -> every lane takes the then-path, warp never splits.
    auto uni = simt::assemble(divergence_kernel(64));
    simt::GlobalMemory mu(64);
    simt::Core cu(uni, mu);
    cu.run(32);
    for (int i = 0; i < 32; ++i) CHECK(mu.load(i) == 100 + i);
    CHECK(cu.stats().divergent_branches == 0);

    // The divergent warp executed the then AND else blocks in series -> strictly more
    // instructions and cycles than the uniform warp. That gap IS the divergence cost.
    CHECK(cd.stats().warp_instructions > cu.stats().warp_instructions);
    CHECK(cd.stats().cycles > cu.stats().cycles);
}

}  // namespace

int main() {
    test_vector_add_one_warp();
    test_tid_global();
    test_coalescing();
    test_latency_hiding();
    test_assembler_errors();
    test_divergence();

    if (g_failures == 0) {
        std::cout << "all mini-GPU tests passed\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
}
