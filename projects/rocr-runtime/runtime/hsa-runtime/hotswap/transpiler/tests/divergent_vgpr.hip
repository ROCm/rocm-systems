// Per-lane VGPR divergence verification (SPE step 4, part A).
//
// Complement to divergent_exec.hip. That test exercised divergent memory
// *stores* under divergent EXEC — the point of SPE's `emitUnderExec`
// wrapping. This one exercises divergent VGPR *values*: two v_cmpx-gated
// regions write different per-lane-selected constants to the SAME VGPR,
// and a single unmasked store at the end observes each lane's final
// value. The intended per-lane semantics are:
//
//   lane  0..15:  out[lane] = 0xAA  (region 1 wrote, region 2 skipped)
//   lane 16..31:  out[lane] = 0xBB  (region 1 skipped, region 2 wrote)
//   lane 32..63:  out[lane] = 0xCC  (pre-initialised default; both skipped)
//
// The expectation is that SPE already handles this correctly *by
// construction*: LLVM SIMT allocas are per-thread, SPE's `emitUnderExec`
// diamond produces a conditional store per active lane, and mem2reg
// inserts a phi joining the "active" and "inactive" paths with each
// lane's own predicate. So every lane carries its own VGPR history
// through the raised IR. This test formalises that reasoning — a pass
// means the model is verified end-to-end on real hardware.
//
// `"+v"(val)` tells the compiler to keep `val` in a VGPR across the asm
// block; `"exec"`, `"s4"`, `"s5"`, `"memory"` prevent reordering and
// ensure the inline assembly boundary is respected.

#include <hip/hip_runtime.h>

extern "C" __global__ void divergent_vgpr_kernel(int *out) {
  int lane = (int)threadIdx.x;
  int val = 0xCC;

  asm volatile(
      // Region 1: lanes 0..15 write 0xAA into `val`.
      "v_cmpx_lt_u32_e64 exec, %[lane], 16\n\t"
      "v_mov_b32 %[val], 0xAA\n\t"
      "s_mov_b64 exec, -1\n\t"
      // Region 2: lanes 16..31 write 0xBB into `val`.
      "v_cmpx_ge_u32_e64 exec, %[lane], 16\n\t"
      "v_cmp_lt_u32_e64 s[4:5], %[lane], 32\n\t"
      "s_and_b64 exec, exec, s[4:5]\n\t"
      "v_mov_b32 %[val], 0xBB\n\t"
      "s_mov_b64 exec, -1\n\t"
      : [val] "+v"(val)
      : [lane] "v"(lane)
      : "exec", "s4", "s5", "memory");

  out[lane] = val;
}
