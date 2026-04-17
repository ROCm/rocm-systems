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
// SPE_DESIGN.md ("Cross-wave translation policy").
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
      // Divergent region: lanes 0..7 write 0xAA.
      "v_cmpx_lt_u32_e64 %[lane], 8\n\t"
      "global_store_b32 %[addr], %[val], off\n\t"
      "s_wait_storecnt 0\n\t"
      "s_mov_b32 exec_lo, -1\n\t"
      :
      : [addr] "v"(addr), [lane] "v"(lane), [val] "v"(val)
      : "exec", "memory");
}
