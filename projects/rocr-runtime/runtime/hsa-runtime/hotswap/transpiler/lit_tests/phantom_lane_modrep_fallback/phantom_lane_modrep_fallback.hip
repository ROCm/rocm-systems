// ARCH: gfx1250
//
// Lit fixture pinning the phantom-lane → MODREP fallback added to
// `raiser.cpp`'s projection selection.  The fixture gives the
// transpiler a kernel that (a) has `max_flat_workgroup_size = 32`
// (via `__launch_bounds__(32)`), which is BELOW the gfx942 target
// wavefront width of 64 and therefore triggers the phantom-lane
// regime the fallback guards against; (b) does NOT use WMMA, so
// the `handle_valu_vop3p.cpp` refusal gate added in the same
// commit does NOT fire — the fallback's CORRECT outcome for
// non-WMMA kernels is a successful lift under
// ModuloReplicationProjection.
//
// Paired with `wmma_phantom_lane_refuse/` (same launch_bounds,
// adds an inline-asm WMMA that DOES trigger the refusal gate).
// Together the two fixtures pin both arms of the fallback's
// post-MODREP decision tree:
//
//   * Non-WMMA kernel in phantom-lane regime: silently falls back
//     to MODREP and raises OK — this fixture.
//   * WMMA kernel in phantom-lane regime: MODREP-fallback then
//     refusal at the WMMA handler — `wmma_phantom_lane_refuse`.
//
// A regression that silently drops the phantom-lane fallback (and
// leaves WaveNative in charge of a sub-wave-width kernel) would
// re-introduce the HIP-error-700 "memory aperture violation" that
// surfaced via `compare_correctness`'s `matmul_fp16` /
// `matmul_fp16_16x16` Triton recipes.  Cross-checked against the
// end-to-end runtime suite by those recipes transitioning from
// `EXIT=2 (HIP 700 crash)` to `EXIT=2 (principled refusal)` after
// the full fix chain landed.
//
// The kernel itself is a trivial per-lane add.  The global_load
// and global_store carry the pointer-arithmetic shape that WOULD
// fault on a phantom lane if the fallback were missing (and
// `handle_flat.cpp`'s load-gating had NOT closed the ungated-load
// portion of the bug, which landed in the same fix chain).

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32) void
phantom_lane_modrep_fallback_kernel(const int *a, int *out) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  out[gid] = a[gid] + tid;
}
