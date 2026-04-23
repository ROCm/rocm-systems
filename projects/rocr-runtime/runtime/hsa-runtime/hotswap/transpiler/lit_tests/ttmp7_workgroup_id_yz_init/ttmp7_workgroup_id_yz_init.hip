// ARCH: gfx1250
//
// Fixture for the `ttmp7 = (workgroup_id_z << 16) | (workgroup_id_y & 0xFFFF)`
// raiser-entry init on gfx12+ (see `raiser.cpp`'s Phase 4 block comment).
//
// The AMDGPU backend's entry-function ABI for gfx12+ preloads
// `workgroup_id_y` into bits [15:0] of ttmp7 and `workgroup_id_z` into
// bits [31:16] (per `AMDGPULegalizerInfo::loadInputValue`:
//
//     WorkGroupIDY = ArgDescriptor::createRegister(TTMP7, 0xFFFFu);
//     WorkGroupIDZ = ArgDescriptor::createRegister(TTMP7, 0xFFFF0000u);
//
// ).  Triton-generated gfx1250 2D-grid matmul kernels (e.g.
// `matmul_kernel_16x16`, `matmul_kernel`) read the Y component via
//
//     s_and_b32 sDST, ttmp7, 0xffff     ; wg_id_y
//
// Without the raiser initialising ttmp7 to this layout, the lifted
// kernel reads an uninitialised SGPR and `workgroup_id_y` is
// effectively always zero; for a 2D grid every workgroup's Y column
// collapses onto the Y=0 column's output region and the rest of the
// output matrix stays at whatever the destination memory held at
// dispatch (verified empirically on `matmul_kernel_16x16` M=32:
// cols 0..15 = correct 32.0 for an all-1s×all-1s matmul, cols 16..31
// = poison-fill from the host's pre-launch memset).
//
// We use inline asm so the front-end doesn't elide the ttmp7 read
// as dead / uniform-folded — the inline asm's "output in SGPR"
// constraint guarantees clang emits the canonical `s_and_b32`
// shape verbatim.

#include <hip/hip_runtime.h>

extern "C" __global__ void ttmp7_wg_id_y_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Canonical Triton / matmul workgroup_id_y extraction:
  //   s_and_b32 sDST, ttmp7, 0xffff
  int wg_id_y;
  asm volatile("s_and_b32 %[wy], ttmp7, 0xffff\n\t"
               : [wy] "=s"(wg_id_y));
  out[tid] = wg_id_y;
}
