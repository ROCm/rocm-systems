// ARCH: gfx1250
//
// Fixture for the V_CMPX / V_CMP→SGPR ballot lit test
// (lit_tests/v_cmpx_ballot).
//
// Compiled as a wave32 (gfx1250) kernel and lifted to wave64 (gfx942)
// by the companion .ll test. This triggers the cross-wave EXEC write
// path in `handle_valu_vcmp.cpp`, which MUST materialise the new EXEC
// mask through `amdgcn.ballot` + truncation rather than the pre-fix
// `sext i1 to i32`. See the MODREP markers in `handle_valu_vcmp.cpp`
// and `ModuloReplicationProjection::ballotI1ToWidth` in
// `wave_projection.cpp` for the contract.
//
// Two EXEC writers share the kernel so one fixture exercises both code
// paths:
//
//   1. `v_cmpx_lt_u32_e64` — writes EXEC directly (V_CMPX branch).
//   2. `v_cmp_lt_u32_e64 s4, ...` — writes a wave32 SGPR mask
//      (V_CMP → SGPR branch). On wave32 the sdst is a single SGPR
//      (32 bits), so we use s4 rather than s[4:5].
//
// The companion `s_mov_b32 exec_lo, -1` restores EXEC after region 1
// so region 2 runs with full EXEC; the raised IR should show both
// `amdgcn.ballot` calls clearly.
//
// Keeping `val` VGPR-live via `"+v"(val)` prevents the compiler from
// eliminating the divergent writes; the unmasked store at the end
// gives every lane a data path to the final value so the `v_mov_b32`s
// don't DCE.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_cmpx_ballot_kernel(int *out) {
  int lane = (int)threadIdx.x;
  int val = 0xCC;
  asm volatile(
      // Region 1: V_CMPX narrows EXEC, divergent write, EXEC restore.
      "v_cmpx_lt_u32_e64 %[lane], 16\n\t"
      "v_mov_b32 %[val], 0xAA\n\t"
      "s_mov_b32 exec_lo, -1\n\t"
      // Region 2: V_CMP writes an SGPR-destination wave mask.
      "v_cmp_lt_u32_e64 s4, %[lane], 8\n\t"
      : [val] "+v"(val)
      : [lane] "v"(lane)
      : "exec", "s4", "memory");
  out[lane] = val;
}
