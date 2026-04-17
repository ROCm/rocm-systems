// bfloat16 -> float32 conversion via v_cvt_f32_bf16.
//
// Target handler: V_CVT_F32_BF16 (handle_valu.cpp).  The opcode exists
// on gfx950 and gfx1250 but NOT on gfx942, so we can only force it on
// the gfx1250 source compile; the gfx942 native compile has to use a
// bit-reinterpretation fallback.  Both paths produce identical floats
// for all finite bf16 inputs because bf16 -> f32 is exact — the high
// 16 bits of the f32 are the bf16 bit-pattern and the low 16 bits are
// zero.
//
// Output shape: 1:1.  Inputs are 16-bit bf16 values (staged as u16 in
// the input buffer for portability; the kernel reinterprets).

#include <hip/hip_runtime.h>

extern "C" __global__ void cvt_f32_bf16(const uint16_t *a, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t bits = static_cast<uint32_t>(a[i]);
  float r;
#if defined(__gfx1250__)
  // The opcode takes a 32-bit input and reads the low 16 bits as bf16.
  asm volatile("v_cvt_f32_bf16 %0, %1" : "=v"(r) : "v"(bits));
#else
  // Pure bit-level upcast: bf16 XXYY -> f32 XXYY_0000.  Exact for all
  // finite and non-NaN bf16 values; matches the instruction semantics.
  uint32_t upped = bits << 16;
  r = *reinterpret_cast<float *>(&upped);
#endif
  c[i] = r;
}
