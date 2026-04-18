// ARCH: gfx1250
//
// Negative-case fixture for the Class-2 ds_swizzle_b32 sub-category
// (lit_tests/c2_ds_swizzle_fft_reserved).
//
// Companion to c2_ds_swizzle_unsafe.hip. That fixture pins refusal
// of the RESERVED top-nibble envelope (top nibble in {0x9, 0xA,
// 0xB, 0xD, 0xF}). This fixture pins refusal of imms that are
// *within* a valid envelope (FFT_MODE here) but have reserved bits
// set, which AMDGPU SIDefines.h `Swizzle::EncBits` does not assign
// semantics to either.
//
// The immediate `offset:0xe020` is inside the FFT_MODE envelope
// (top nibble = 0xE) but bit 5 is set — and bits 5..11 are RESERVED
// per the FFT encoding (only bits 0..4 carry the FFT swizzle
// selector; everything else must be 0). Hardware behavior is
// undefined.
//
// Without the FFT/ROTATE reserved-bit validation in
// `dsSwizzleSafeForModRep`, this imm would have passed the safety
// gate (since `imm & FFT_ROTATE_MODE_MASK == FFT_MODE_ENC` matches
// the broad envelope check). The classifier must catch this with a
// loud refusal so a future widening that drops the reserved-bit
// validation is caught.

#include <hip/hip_runtime.h>

extern "C" __global__ void c2_ds_swizzle_fft_reserved_kernel(int *a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[tid];
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0xe020\n\t"
      "s_wait_dscnt 0\n\t"
      : [v] "+v"(v));
  a[tid] = v;
}
