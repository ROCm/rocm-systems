// Packed f32 -> f16 round-toward-zero conversion.
//
// Target handler: V_CVT_PKRTZ_F16_F32 (handle_valu.cpp).  On gfx1250
// hipcc emits this as `v_cvt_pk_rtz_f16_f32`; on gfx942 the same
// `__builtin_amdgcn_cvt_pkrtz` builtin lowers to `v_cvt_pkrtz_f16_f32`.
// Both route to the same pseudo (SemOp::V_CVT_PKRTZ_F16_F32) after
// canonicalization, so lifting this kernel exercises the handler end
// to end.
//
// Output shape: 1:1 — one u32 per input pair.  The u32 is the
// bit-pattern of the v2f16 result (low half = from `a`, high half =
// from `b`).  Wave-size-agnostic.
//
// Round-toward-zero is mandatory here: f16 cannot represent most f32
// values exactly, and the CPU reference must mirror the RTZ rounding
// of the builtin to diff cleanly.

#include <hip/hip_runtime.h>

using v2f16 = __attribute__((ext_vector_type(2))) _Float16;

extern "C" __global__ void cvt_pkrtz(const float *a, const float *b,
                                     uint32_t *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  auto packed = __builtin_amdgcn_cvt_pkrtz(a[i], b[i]);
  c[i] = *reinterpret_cast<uint32_t *>(&packed);
}
