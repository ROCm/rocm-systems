// ARCH: gfx1250
//
// Lit fixture for the VOP3P mixed-precision FMA family:
//
//   v_fma_mix_f32        f16-narrow (all gfx targets, VOP3PInstructions.td:109)
//   v_fma_mix_f32_bf16   bf16-narrow (gfx9.5+/gfx1250, same .td row)
//
// Both share the same operand shape and modifier layout
// (op_sel / op_sel_hi bracketlists); only the narrow element type
// changes (f16 vs bf16).  handle_valu_vop3p.cpp dispatches both
// through the same parser + write-back shape, parameterised by
// `narrowTy` — this fixture pins that the BF16 variant goes through
// `fpext bfloat -> float` (named `mix_cvt_bf16`) and the F16 variant
// goes through `fpext half -> float` (named `mix_cvt`), and that both
// emit the canonical `llvm.fma.f32` for the triple.
//
// The BF16 shape is cross-target-universal — `fpext bfloat -> float`
// lowers on every AMDGPU target (it's a shift-left-16 + bitcast-to-
// float on gfx942 and a first-class conversion on gfx1250), so the
// test runs the lift under `--target-isa=gfx942` specifically to prove
// the BF16 narrow half does not require any cross-target refusal.
//
// We use inline asm to pin the literal modifier surface: the exact
// op_sel / op_sel_hi pair below (`op_sel:[0,1,0] op_sel_hi:[1,1,0]`)
// matches the shape the kerneldex scope_discovery `_attn_fwd` kernel
// emits — src0 reads the LO bf16 half, src1 reads the HI bf16 half,
// src2 is passed through as full f32.  Inputs come from the kernarg
// pointer to defeat const-fold and force real VGPR sources.

#include <hip/hip_runtime.h>

extern "C" __global__ void
v_fma_mix_f32_bf16_kernel(unsigned int *out) {
  unsigned int a = (unsigned int)(uintptr_t)out;
  unsigned int b = (unsigned int)(uintptr_t)(out + 1);
  unsigned int c = (unsigned int)(uintptr_t)(out + 2);
  unsigned int r_bf16, r_f16;

  asm volatile(
      "v_fma_mix_f32_bf16 %[d], %[s0], %[s1], %[s2] "
      "op_sel:[0,1,0] op_sel_hi:[1,1,0]\n\t"
      : [d] "=v"(r_bf16)
      : [s0] "v"(a), [s1] "v"(b), [s2] "v"(c));

  asm volatile(
      "v_fma_mix_f32 %[d], %[s0], %[s1], %[s2] "
      "op_sel:[0,1,0] op_sel_hi:[1,1,0]\n\t"
      : [d] "=v"(r_f16)
      : [s0] "v"(a), [s1] "v"(b), [s2] "v"(c));

  out[threadIdx.x * 2 + 0] = r_bf16;
  out[threadIdx.x * 2 + 1] = r_f16;
}
