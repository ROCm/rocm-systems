// ARCH: gfx1250
//
// Lit fixture pinning the WMMA → MFMA refusal gate added to
// `handle_valu_vop3p.cpp` in the same commit that introduced
// `WaveProjection::providesFullWaveExecInvariant()`.  The kernel's
// `__launch_bounds__(32)` drives `max_flat_workgroup_size` to 32
// in the gfx1250 HSACO's amdhsa metadata, triggering
// `raiser.cpp`'s phantom-lane → MODREP fallback on a gfx942
// (wave64) target.  Under MODREP, the `emitWMMAtoMFMA_F32_16x16x4`
// lowering's full-wave EXEC dependency (all 64 target lanes must
// participate in the MFMA collective, guaranteed only by
// WaveNativeProjection's `init_whole_wave`) collapses: running the
// lowering would produce silently-wrong numerics.  The gate
// refuses loudly instead.
//
// Paired with `phantom_lane_modrep_fallback/` (same launch_bounds,
// no WMMA — fallback silently succeeds under MODREP).  Together the
// two fixtures pin both arms of the post-fallback decision:
//
//   * Non-WMMA kernel in phantom-lane regime: silent MODREP accept.
//   * WMMA kernel in phantom-lane regime: MODREP-fallback then
//     refusal here.
//
// This fixture's CHECK lines name the specific opcode
// (`v_wmma_f32_16x16x4_f32`), the RaiseFailure format string
// (`unsupportedShape` / `VOP3P`), and the phantom-lane attribution
// so a regression that weakens the gate (e.g. drops the
// `providesFullWaveExecInvariant` check in
// `handle_valu_vop3p.cpp`'s K=4 f32 case) is caught with a clean
// attribution trail pointing at the specific handler site.
//
// See `compare_correctness`'s `matmul_fp16_16x16` Triton recipe
// for the end-to-end runtime equivalent: before the gate landed,
// salmon silently produced WRONG numerics for every shape (after
// the phantom-lane MODREP fallback closed the HIP-700 crash); post-
// gate, salmon refuses with EXIT=2 on every shape and the stderr
// carries the same attribution as this lit fixture.

#include <hip/hip_runtime.h>

typedef float v2f __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

extern "C" __global__ __launch_bounds__(32) void
wmma_phantom_lane_refuse_kernel(const v2f *a_ptr,
                                 const v2f *b_ptr,
                                 v8f *c_ptr) {
  v2f a = a_ptr[0];
  v2f b = b_ptr[0];
  v8f c = c_ptr[0];
  c = __builtin_amdgcn_wmma_f32_16x16x4_f32(
      /*a_neg=*/false, a,
      /*b_neg=*/false, b,
      /*c_mod=*/(short)0, c,
      /*a_reuse=*/false,
      /*b_reuse=*/false);
  c_ptr[0] = c;
}
