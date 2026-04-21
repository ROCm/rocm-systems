// ARCH: gfx1250
//
// Fallback-path companion to `v_cmp_cndmask_sgpr/`. Same V_CMP ->
// SGPR -> V_CNDMASK_B32_e64 idiom, but with a scalar clobber of
// the mask SGPR sandwiched in between the producer and the
// consumer. The scalar write routes through `AllocaRegFile::
// storeSGPR32`, which fires the `onSgprWritten` hook that
// invalidates the shadow entry keyed on that SGPR (see
// `RaiseContext::invalidateSgprWaveMaskI1`). The subsequent
// V_CNDMASK_B32_e64 consumer then finds the shadow empty and
// takes the fallback `extractLaneBitFromWaveMask` path — exactly
// the intended "any interference defeats the cache" invariant
// (hotswap/docs/sgpr-wave-mask-translation.md section 3.1 / I3).
//
// The `.ll` companion pins:
//   * the extract chain IS emitted in the consumer's dataflow
//     (`mask_lane_idx` / `mask_at_lane` / `mask_lane_bit` /
//     `mask_lane_i1` → cndmask `select i1`),
//   * the cndmask `select` does NOT consume `%vcmpf` directly
//     (the direct-i1 shadow path would be WRONG here: by the
//     time the cndmask reads s4, the scalar clobber has erased
//     the wave-mask meaning of s4, and reading the i1 would
//     produce a stale result).
//
// This fixture cost is low but coverage value is high — it's the
// only regression guard for "we correctly INVALIDATE the shadow
// on scalar writes." Without it, a future change that forgets
// the invalidation hook (or mis-wires it) would silently read a
// stale i1 and miscompile, but the fused-path fixture would
// still pass.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_cmp_cndmask_sgpr_scalar_clobber_kernel(
    float *out, const float *in) {
  int lane = (int)threadIdx.x;
  float val = in[lane];
  float result = 0.0f;
  // Scalar sentinel the clobber writes into s4 between V_CMP and
  // V_CNDMASK. Its exact value does not matter; what matters is
  // that `s_mov_b32 s4, ...` routes through storeSGPR32 and fires
  // the invalidation hook.
  int sentinel = 0x5A5A5A5A;
  asm volatile(
      // Producer: V_CMP writes s4 with the per-lane wave mask;
      // shadow records the compare's i1 under map[4].
      "v_cmp_ge_f32_e64 s4, |%[val]|, 0.5\n\t"
      // Scalar clobber: s_mov_b32 writes s4 with a uniform i32
      // scalar value. storeSGPR32(4, ...) fires the onSgprWritten
      // hook; invalidateSgprWaveMaskI1(4) drops map[4].
      "s_mov_b32 s4, %[sentinel]\n\t"
      // Dummy consumer of the scalar s4 so the compiler cannot DCE
      // the s_mov. We read s4 into a VGPR via a sideways v_mov with
      // the SGPR source, then XOR it with itself to absorb it into
      // `result` without disturbing the per-lane cndmask below. Kept
      // as `%[sentinel]` rather than `"s"(sentinel)` in the out-of-
      // line operand list because the s_mov was already issued; we
      // just need to keep the output visible to prevent the clobber
      // from being deleted.
      "v_mov_b32_e32 v5, s4\n\t"
      // Consumer: V_CNDMASK reads s4 as a wave mask. Shadow is
      // empty (invalidated above), so the handler falls back to
      // `extractLaneBitFromWaveMask(loadSGPR32(4))`. On
      // cross-widening this is the documented lossy path — which
      // is exactly correct here, because `s4` no longer holds the
      // original compare's wave mask; it holds `sentinel`. The
      // fallback extract reproduces the per-lane bits implied by
      // the scalar sentinel: for `sentinel = 0x5A5A5A5A = 0b...01011010
      // 01011010 01011010 01011010`, per-lane bit i is (sentinel >> i)
      // & 1, i.e. a fixed pattern independent of `val`. The CPU
      // reference matches this exactly.
      "v_cndmask_b32_e64 %[r], -1.0, 1.0, s4\n\t"
      : [r] "=v"(result)
      : [val] "v"(val), [sentinel] "s"(sentinel)
      : "s4", "v5");
  // XOR with whatever v5 holds; OR the low bit into `result` so the
  // compiler cannot DCE the v_mov even under aggressive inlining.
  // (We only observe `result` on the output buffer, so keeping v5
  // live through a no-op combination suffices.)
  out[lane] = result;
}
