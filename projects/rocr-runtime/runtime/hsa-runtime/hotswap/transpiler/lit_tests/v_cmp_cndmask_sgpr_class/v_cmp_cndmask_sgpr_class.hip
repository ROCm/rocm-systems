// ARCH: gfx1250
//
// Class-compare companion to `v_cmp_cndmask_sgpr/`. Identical
// shadow-path contract but the producer is `v_cmp_class_f32_e64`
// (VOPC, FP IEEE-class predicate) rather than the predicate-
// compare `v_cmp_ge_f32_e64`. The V_CMP_CLASS handler in
// `handle_valu_vcmp.cpp` takes the `if (m->isClass)` arm and
// emits `llvm.amdgcn.class.f32` whose result is a per-lane i1,
// the same shape as the `fcmp` output of the predicate-compare
// path — so the shadow recording is sound for both arms.
//
// Previously there was a `!m->isClass` gate around
// `ctx.recordSgprWaveMaskI1`, intended as conservative scoping.
// It left class-compare consumers under cross-widening falling
// through to the lossy `extractLaneBitFromWaveMask` extract — for
// no correctness reason. Gate removed; this fixture is the
// regression guard for "class compares also get the direct-i1
// shadow path."
//
// Mask 0x200 = bit 9 = +inf singleton, same choice as
// `v_cmp_class_f32.hip`. Keeps the fixture self-documenting: any
// rewrite of the class-mask semantics would show up as a
// different literal in the lifted IR.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_cmp_cndmask_sgpr_class_kernel(float *out,
                                                            const float *in) {
  int lane = (int)threadIdx.x;
  float val = in[lane];
  float result = 0.0f;
  asm volatile(
      // Producer: V_CMP_CLASS_F32_e64 writes a single wave32 SGPR
      // with the per-lane class predicate. The raiser emits
      // `call i1 @llvm.amdgcn.class.f32(float %val, i32 512)` as
      // `%vclass`, then routes `%vclass` through the same ballot /
      // trunc / SGPR store tail as predicate compares. With the
      // shadow map populated with `%vclass`, the cndmask consumer
      // below reads it directly.
      "v_cmp_class_f32_e64 s4, %[val], 0x200\n\t"
      // Consumer: V_CNDMASK_B32_e64 reads s4 as a per-lane mask.
      // Under the fused path this is `select i1 %vclass, ...` —
      // direct-i1 from the class intrinsic, no extract chain. The
      // `-1.0` / `1.0` immediates are the same distinctive literals
      // as the predicate-compare sibling so the CHECK line matches
      // the same i32 bit patterns.
      "v_cndmask_b32_e64 %[r], -1.0, 1.0, s4\n\t"
      : [r] "=v"(result)
      : [val] "v"(val)
      : "s4");
  out[lane] = result;
}
