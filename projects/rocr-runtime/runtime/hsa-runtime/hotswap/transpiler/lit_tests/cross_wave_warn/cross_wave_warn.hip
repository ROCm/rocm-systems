// ARCH: gfx1250
//
// Fixture for the cross-wave translation warning
// (lit_tests/cross_wave_warn).
//
// The raiser's Phase-1.4 cross-wave gate in raiser.cpp must warn when
// a kernel that manipulates EXEC is translated from one wave size to
// another (e.g. gfx1250 authored for wave32, raised for a gfx942
// wave64 target), because `emitLaneActiveBit` then relies on
// modulo-replication, which is not provably correct in general. See
// hotswap/docs/wave-size-translation.md §§5–7 for the cross-wave
// translation policy.
//
// Today the policy is warn-only — the raiser prints a `transpiler:
// WARNING: cross-wave translation ...` diagnostic and continues — so
// this test asserts *only* that the warning is emitted and that the
// raise still succeeds with `exit=0`. Downstream CI can escalate by
// piping stderr through `grep -q 'cross-wave translation'` if a
// strict policy is desired.
//
// We keep the kernel minimal: one `v_cmpx` divergent region that
// writes EXEC, plus an EXEC restore. `v_cmpx` is on the SPE
// allow-list (so the Phase-1.5 gate does *not* fire) and is the
// canonical EXEC-writer the cross-wave scan looks for.

#include <hip/hip_runtime.h>

extern "C" __global__ void cross_wave_writer_kernel(int *out) {
  int lane = (int)threadIdx.x;
  int *addr = out + lane;
  int val = 0xAA;
  asm volatile(
      // Divergent region: bounds-check vs. 64 ≥ W_s (W_s-1 for gfx1250
      // = 31). Using a constant ≥ W_s keeps this a structurally
      // lane-position-INDEPENDENT comparison (the classifier in
      // `c5_predicate_chain_classifier.cpp` accepts bounds checks
      // whose constant exceeds W_s-1 because they cannot partition
      // lanes by their in-wave position). The v_cmpx still writes
      // EXEC — that is what the Phase 1.4 cross-wave gate is pinned
      // to — it just no longer looks like a scan-stage / half-wave
      // predicate that would miscompile under modulo-replication.
      // See hotswap/docs/modrep-predicate-chain.md §5 narrow-O1 for
      // the distinction between lane-position-scoped and bounds-
      // check predicates. (Pre-C5-classifier this fixture used
      // `v_cmpx_lt_u32_e64 lane, 8`; that was a genuine Class-5
      // latent-bug kernel that the C5 classifier correctly now
      // refuses. The fixture was updated rather than xfail'd so the
      // cross-wave warning path still has an IR-generating producer.)
      "v_cmpx_lt_u32_e64 %[lane], 64\n\t"
      "global_store_b32 %[addr], %[val], off\n\t"
      "s_wait_storecnt 0\n\t"
      "s_mov_b32 exec_lo, -1\n\t"
      :
      : [addr] "v"(addr), [lane] "v"(lane), [val] "v"(val)
      : "exec", "memory");
}
