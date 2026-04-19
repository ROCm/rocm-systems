// ARCH: gfx1250
//
// Lit fixture for `v_pk_add_f32` with an INLINE LITERAL source.
// SemOp::V_PK_ADD_F32 + the VOP3P handler in
// transpiler/handle_valu_vop3p.cpp already handle the
// register/register form; this fixture covers the previously-refused
// shape where one operand is a 32-bit literal that the hardware
// broadcasts to both packed lanes.
//
// Tensilelite swiglu (`_swiglu_bf16xbf16_16x128`) issues this exact
// pattern to add a packed bias constant.  Before this fixture's
// supporting handler change, raise_cli aborted the kernel with
// `transpiler: v_pk_add_f32: non-register source 1 (immediate in
// VOP3P not supported)` and reported `unsupportedOpcode [VALU]`.
//
// We use inline asm to pin the literal-source addressing form: HIP's
// natural `__builtin_amdgcn_pk_add_f32` lowering on gfx942 host
// compiles can't be retargeted to gfx1250 VOP3P literal-source
// encoding.  The literal `1.0` is a VOP3P inline constant; the
// hardware broadcasts it to both packed lanes.

#include <hip/hip_runtime.h>

extern "C" __global__ void
v_pk_add_f32_lit_kernel(unsigned int *out) {
  // Read the source pair as a 64-bit value via the `v` constraint,
  // which binds a VGPR pair (VReg_64).  We seed it from a non-
  // constant kernarg-derived `out` to defeat constant folding and
  // force the compiler to materialise a real packed `<2 x f32>`.
  unsigned long long packed = (unsigned long long)(uintptr_t)out;

  asm volatile(
      "v_pk_add_f32 %[d], %[s], 1.0\n\t"
      : [d] "=v"(packed)
      : [s] "v"(packed));

  out[threadIdx.x * 2 + 0] = (unsigned int)(packed & 0xFFFFFFFFu);
  out[threadIdx.x * 2 + 1] = (unsigned int)((packed >> 32) & 0xFFFFFFFFu);
}
