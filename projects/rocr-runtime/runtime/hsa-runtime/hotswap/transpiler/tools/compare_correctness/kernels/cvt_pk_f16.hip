// Packed f32 -> f16 round-to-nearest-even conversion.
//
// Target handler: V_CVT_PK_F16_F32 (handle_valu.cpp).  The gfx1250
// backend fuses a `{(_Float16)a, (_Float16)b}` pair into a single
// `v_cvt_pk_f16_f32` instruction; gfx942 does not have this opcode
// and lowers the same source to two scalar `v_cvt_f16_f32` plus a
// `v_pack_b32_f16`.  Both produce the same f16 bit pattern per lane,
// so the CPU reference applies to both engines.
//
// Output shape: 1:1 — one u32 (= bit-pattern of v2f16) per input
// pair.  Wave-size-agnostic.

#include <hip/hip_runtime.h>

using v2f16 = __attribute__((ext_vector_type(2))) _Float16;

extern "C" __global__ void cvt_pk_f16(const float *a, const float *b,
                                      uint32_t *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  v2f16 packed = {(_Float16)a[i], (_Float16)b[i]};
  c[i] = *reinterpret_cast<uint32_t *>(&packed);
}
