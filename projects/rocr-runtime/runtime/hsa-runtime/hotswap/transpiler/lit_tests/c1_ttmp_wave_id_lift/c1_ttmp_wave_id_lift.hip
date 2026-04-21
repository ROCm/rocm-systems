// ARCH: gfx1250
//
// Fixture for the Class-1 "wave_id lift via s_bfe_u32 ttmp8 0x50019"
// rescue (lit_tests/c1_ttmp_wave_id_lift). Paired with the
// lit_tests/c1_lane_id_leak v_mbcnt_hi fixture; while
// c1_lane_id_leak pins an *unrewritable* Class-1 leak (refuses), this
// fixture pins the canonical `wave_id_in_workgroup` read pattern
// that has a *principled rescue* via the `S_BFE_U32` pattern-lift
// in `handle_sop2.cpp` (see `hotswap/docs/wave-size-translation.md`
// §5.6.2, §6 Class-1 rescue).
//
// The gfx1250 command-processor materialises `wave_id_in_workgroup`
// in ttmp8[29:25] for every launched wavefront. Every
// Tensile / rocBLAS matmul reads it via:
//
//     s_bfe_u32 sN, ttmp8, 0x50019   // bits [29:25] → wave_id
//
// Under same-wave translation (gfx1250 → gfx1250 / gfx1251) the
// per-lane value is uniform across a wave32 and the transpiler's
// generic BFE handler + SGPR alloca round-trip works. Under cross-
// widening (gfx1250 → gfx942 / gfx950) the per-lane value *must*
// differ within a target wave64 (lanes 0..31 see `wave_id = 2k`,
// lanes 32..63 see `wave_id = 2k+1`), and the backend's
// scalarisation of the SGPR-path read loses that distinction.
//
// The rescue is to detect the canonical `(ttmp8, 0x50019)` operand
// shape at raise time and emit the source-intended expression
// directly:
//
//     %tid   = call i32 @llvm.amdgcn.workitem.id.x()
//     %wave  = lshr i32 %tid, <log2(W_s)>         ; W_s = 32 → 5
//     %wid   = and  i32 %wave, 31
//
// The divergent-leaf `workitem.id.x` intrinsic ensures the backend
// keeps the value in a VGPR per lane, so downstream tile-offset
// arithmetic and memory address computation see the correct
// per-source-wave rank and matmul kernels stop checkerboarding.
//
// We use inline assembly so the HIP front-end does not elide the
// BFE as dead / uniform-folded — inline asm's "output in SGPR"
// constraint guarantees clang emits the `s_bfe_u32` shape verbatim.

#include <hip/hip_runtime.h>

extern "C" __global__ void c1_ttmp_wave_id_lift_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Canonical Tensile / rocBLAS wave_id_in_workgroup extraction:
  // offset=25, width=5, yielding a value in [0, max_waves_per_wg).
  int wave_id;
  asm volatile("s_bfe_u32 %[wid], ttmp8, 0x50019\n\t"
               : [wid] "=s"(wave_id));
  out[tid] = wave_id;
}
