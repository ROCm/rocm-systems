// ARCH: gfx1250
//
// Lift fixture for `v_div_scale_f32`'s literal-numerator encoding.
// Pins the three-arm operand decoder in `handle_valu.cpp`'s
// `V_DIV_SCALE_F32` branch: the hardware (src0, src1, src2) triple
// identifies which of (numer, denom) gets scaled by operand-identity
// equality, and the identity check has to work for BOTH register
// and inline-constant operands.  The AMDGPU fdiv expansion for
// `1.0 / x` emits two scale calls whose numerator lives in an
// inline `1.0` literal in src2:
//
//   v_div_scale_f32 vA, null,    x,   x,   1.0    ; (d, d, n) — scale denom
//   v_div_scale_f32 vB, vcc_lo,  1.0, x,   1.0    ; (n, d, n) — scale numer
//
// Before the handler knew about literal-operand equality, the
// register-only `isSrcReg(0) && isSrcReg(2)` gate fell through for
// both calls, decoded them as `selectNumerator = false`, and
// passed `(s0, s1, false)` to `@llvm.amdgcn.div.scale.f32` — the
// (d, d, false) call told the backend "compute d/d, scale denom",
// and the (1.0, v, false) call asked for scaled denom of a divide
// whose scale-numer arm never fired.  End-to-end effect: every
// translated `1.0 / sqrt(...)` collapsed through the downstream
// `v_div_fmas` / `v_div_fixup` chain to the constant `1.0`
// (observable as layer-norm's rstd reading `0x3f800000` bit-exact
// regardless of input; see the triage under
// `hotswap/docs/triage-*-softmax-matmul*.md` for the end-to-end
// reproducer).
//
// Post-fix: the lift emits a canonical `(numer, denom, flag)` pair
// sourced from (src2, src1) with `flag = false` for scale-denom
// and `flag = true` for scale-numer.  The CHECKs below pin both
// arms of the pair and negatively assert the pre-fix `(%x, %x, …)`
// shape.
//
// Kernel.  `1.0f / sqrtf(x)` is the smallest source that forces
// hipcc's AMDGPU backend to emit the full IEEE-conformant expansion
// with an inline `1.0` numerator.  The `volatile` load + `out[]`
// store keep the expression live so the expansion survives dead-
// code elimination.
#include <hip/hip_runtime.h>
extern "C" __global__ void v_div_scale_f32_literal_numer_kernel(
    float *out, const float *in) {
  int tid = (int)threadIdx.x;
  // `volatile` load prevents hipcc from constant-folding across
  // kernel launches; the result is a per-lane VGPR.
  float x = *(const volatile float *)(in + tid);
  // `__builtin_sqrtf` is lowered to `@llvm.sqrt.f32`, which the
  // AMDGPU backend lowers to `v_sqrt_f32` plus Newton fixup.
  // `1.0f / sqrt(x)` then triggers the IEEE-conformant fdiv
  // expansion — the exact instruction sequence we need to pin.
  out[tid] = 1.0f / __builtin_sqrtf(x);
}
