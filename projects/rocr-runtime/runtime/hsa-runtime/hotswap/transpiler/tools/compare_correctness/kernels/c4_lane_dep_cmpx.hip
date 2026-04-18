// Probe for SPE_DESIGN.md §3 Class 4 — lane-position-dependent EXEC
// writes. Mirrors lit_tests/c4_lane_dep_cmpx (offline classifier pin);
// this file provides the runtime evidence that (a) native gfx942
// produces a specific output, (b) legacy half-wave-masking produces a
// DIFFERENT output (upper target half dead), and (c) the
// classifier-gated salmon path refuses the kernel instead of silently
// miscompiling.
//
// Kernel design. Each thread writes 1 to `out[tid]` iff it is active
// during the masked store. The v_cmpx compares `mbcnt_lo(-1, 0)`
// against the constant 16 with a >= predicate — semantically "enable
// lanes in the upper half of the wavefront." The two observable
// quantities are:
//
//   - On gfx942 native (wave64): v_mbcnt_lo clamps at 32 for lanes
//     32..63 (popcount of all below-lane exec_lo bits), so
//     v_cmpx_ge_u32 v10, 16 enables lanes 16..63. Output: 0 at
//     wave-relative positions 0..15, 1 at 16..63.
//   - On gfx1250 source (wave32): v_mbcnt_lo produces 0..31 cleanly,
//     v_cmpx_ge_u32 v10, 16 enables source lanes 16..31. Under
//     legacy half-wave masking, target lanes 32..63 never run, so
//     target lane 32..63 stay at sentinel (0). Output diverges from
//     native at exactly those positions.
//
// The classifier-gated salmon path aborts with
// "cross-wave-Class-4-lane-position-dependent", preventing the silent
// miscompile.

#include <hip/hip_runtime.h>

extern "C" __global__ void c4_lane_dep_cmpx(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int *addr = out + tid;
  int val = 1;
  asm volatile(
      // lane_id via v_mbcnt_lo. On wave32 source this is 0..31; on
      // wave64 native the upper 32 lanes clamp to 32 because
      // exec_lo only has 32 below-lane bits.
      "v_mbcnt_lo_u32_b32 v10, -1, 0\n\t"
      // Lane-position-dependent EXEC write. Classifier must refuse
      // under gfx1250 → gfx942.
      "v_cmpx_ge_u32_e64 v10, 16\n\t"
      "global_store_b32 %[addr], %[val], off\n\t"
      "s_wait_storecnt 0\n\t"
      "s_mov_b32 exec_lo, -1\n\t"
      :
      : [addr] "v"(addr), [val] "v"(val)
      : "v10", "exec", "memory");
}
