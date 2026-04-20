// ARCH: gfx1250
//
// Fixture for the Class-4 "lane-position-dependent EXEC writes"
// obstruction (lit_tests/c4_lane_dep_cmpx).
//
// Class 4 (hotswap/docs/wave-size-translation.md §6): a v_cmpx_* (or s_*_saveexec_b32 derived
// from mbcnt_hi) whose gating expression depends on the ABSOLUTE
// lane position within the wave. Under modulo-replication the EXEC
// mask is projected by `lane_id mod W_s`, but the compare operates
// on target-hardware lane_id directly, so the mask the backend
// materialises does not reproduce the source's intent.
//
// This fixture contrasts with lit_tests/cross_wave_warn, which
// intentionally uses a lane-position-INDEPENDENT v_cmpx (bounds
// check against a non-lane-derived value) so the kernel remains
// wave-size-oblivious and the (warn-only, pre-classifier) gate
// trips without the (c) refusal firing.
//
// The distinguishing feature is that here the v_cmpx's RHS is the
// lane-id itself (computed via v_mbcnt_lo into v10), making the
// compare fundamentally lane-position-dependent and unrewritable.

#include <hip/hip_runtime.h>

extern "C" __global__ void c4_lane_dep_cmpx_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int *addr = out + tid;
  int val = 1;
  asm volatile(
      // lane_id = popcount of bits below current lane in exec_lo.
      // On wave32 source this is 0..31; the v_cmpx below uses that
      // value as its LHS, so the compare IS lane-position-dependent.
      "v_mbcnt_lo_u32_b32 v10, -1, 0\n\t"
      // Lane-position-dependent EXEC write: enable lanes where
      // lane_id < 16. Classifier must recognise this as a C4 site
      // that has no rewrite in §4's table.
      "v_cmpx_lt_u32_e64 v10, 16\n\t"
      "global_store_b32 %[addr], %[val], off\n\t"
      "s_wait_storecnt 0\n\t"
      // Restore EXEC so any code after the asm block is unaffected.
      "s_mov_b32 exec_lo, -1\n\t"
      :
      : [addr] "v"(addr), [val] "v"(val)
      : "v10", "exec", "memory");
}
