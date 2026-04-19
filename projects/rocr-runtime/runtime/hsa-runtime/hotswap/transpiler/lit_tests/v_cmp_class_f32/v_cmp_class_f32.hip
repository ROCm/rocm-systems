// ARCH: gfx1250
//
// Lift fixture for VOPC v_cmp_class_f32_e64. This is NOT a predicate
// compare — src1 is an i32 mask of FP classes (signaling NaN, quiet
// NaN, -inf, -normal, -subnormal, -0, +0, +subnormal, +normal, +inf
// at bits 0..9), and the result lane bit is set iff src0's IEEE
// class matches any enabled bit in the mask. The lift target is
// `llvm.amdgcn.class.f32(float, i32) -> i1`, and the wave-mask
// write-back to the SGPR-pair destination must route through
// `llvm.amdgcn.ballot` + truncation (same discipline as the
// V_CMP/V_CMPX predicate compares — see lit_tests/v_cmpx_ballot
// for the wave32 -> wave64 cross-wave plumbing contract).
//
// The metadata is extracted by `parseVCmpPseudoName` in
// transpiler/opcode_map.cpp (which collapses the V_CMP_CLASS_F<bits>
// MC pseudos onto SemOp::V_CMP with `VCmpMeta::isClass = true`),
// and the special-case lift lives in the `if (m->isClass)` branch
// of transpiler/handle_valu_vcmp.cpp.
//
// Mask 0x200 = bit 9 = +inf (singleton class). Picking a singleton
// class makes the test self-documenting: any other mask spelling
// in the lifted IR would silently be wrong.
//
// Keeping `val` VGPR-live via `"+v"(val)` and the unmasked store at
// the end prevents the compiler from DCE'ing the SGPR write.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_cmp_class_f32_kernel(int *out, const float *in) {
  int lane = (int)threadIdx.x;
  float val = in[lane];
  int sink = 0;
  asm volatile(
      // Wave32 source: V_CMP_CLASS_F32_e64 sdst is a single SGPR (s4),
      // not an SGPR pair. The cross-wave (wave32 -> wave64) lift then
      // widens the ballot via `ballotI1ToWidth` — exactly the path
      // the v_cmpx_ballot fixture pins for the predicate-compare
      // siblings.
      "v_cmp_class_f32_e64 s4, %[val], 0x200\n\t"
      "s_mov_b32 %[sink], s4\n\t"
      : [sink] "=s"(sink)
      : [val] "v"(val)
      : "s4");
  out[lane] = sink;
}
