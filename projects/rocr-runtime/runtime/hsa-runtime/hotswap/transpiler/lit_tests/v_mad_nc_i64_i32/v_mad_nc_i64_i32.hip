// ARCH: gfx1250
//
// Lift fixture for VOP3 v_mad_nc_i64_i32 (gfx1250+ no-carry signed
// 64-bit multiply-add).  See `SemOp::V_MAD_NC_I64_I32` in
// transpiler/semop.hpp and the matching handler block in
// transpiler/handle_valu.cpp (`if (sop == SemOp::V_MAD_NC_I64_I32)`);
// the decoder entry lives in transpiler/opcode_map.cpp next to the
// legacy `V_MAD_U64_U32_e64` row.  Handler ↔ test back-reference is
// non-negotiable per the transpiler test policy.
//
// v_mad_nc_i64_i32 is gfx1250-only, defined in
// AMDGPU/VOP3Instructions.td at encoding 0x2fb as
//   defm V_MAD_NC_I64_I32 : VOP3Inst<"v_mad_nc_i64_i32",
//                                    VOP_I32_I32_I64_DPP>;
// inside a `SubtargetPredicate = isGFX1250Plus` block, with the
// DPP-capable VOP3Only_Realtriple_gfx1250<0x2fb> real-instance row
// further down the same file.  The `VOP_I32_I32_I64_DPP` profile
// (`VOPProfile<[i64, i32, i32, i64]>`) pins the operand layout:
//   dst  : VGPR pair, 64-bit signed result
//   src0 : VGPR, 32-bit signed factor A
//   src1 : VGPR, 32-bit signed factor B
//   src2 : VGPR pair, 64-bit signed accumulator C
// Semantically `D.i64 = sext(S0.i32) * sext(S1.i32) + S2.i64`, with
// no carry output (contrast `v_mad_i64_i32` on gfx9/gfx942, which
// additionally writes the overflow bit to VCC).  The AMDGPU backend's
// `SelectMad64_32(Signed=true)` pattern matcher (see
// AMDGPUISelDAGToDAG.cpp:1220 and AMDGPUInstructionSelector.cpp:606)
// lowers the canonical `add(mul(sext s0, sext s1), s2_i64)` IR that
// our handler emits back into `v_mad_nc_i64_i32` on gfx1250 targets
// and into `v_mad_i64_i32` (with the VCC carry allocated to a
// scratch SGPR and discarded) on gfx942 cross-lifts — so the lit
// fixture's `--target-isa=gfx942` RUN line exercises exactly the
// path `downcast_to_mxfp_{fp32,bf16}` needed before this handler
// existed (raise-time refusal with `v_mad_nc_i64_i32 [VOP3]`).
//
// The kernel body routes the two 32-bit factors and the 64-bit
// accumulator through VGPRs (the operand constraints force the
// inline assembler's operand allocator to pick VGPR registers,
// not SGPRs — `v_mad_nc_i64_i32` rejects scalar source operands by
// its encoding).  We deliberately avoid readfirstlane / writelane
// patterns here: the companion `corpus_softmax_fp32` kernel proves
// that the post-raise `writelane/readlane-post-raise-safety-net`
// will refuse if we form a `readlane → readfirstlane` chain, which
// would mask this fixture's intent (is the V_MAD_NC_I64_I32 decoder
// entry reachable? is the handler emitting the right IR?).

#include <hip/hip_runtime.h>

extern "C" __global__ void v_mad_nc_i64_i32_kernel(long long *out,
                                                   const int *a_ptr,
                                                   const int *b_ptr,
                                                   const long long *c_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int a = a_ptr[tid];
  int b = b_ptr[tid];
  long long c = c_ptr[tid];
  long long result;
  asm volatile("v_mad_nc_i64_i32 %[r], %[a], %[b], %[c]\n\t"
               : [r] "=v"(result)
               : [a] "v"(a), [b] "v"(b), [c] "v"(c));
  out[tid] = result;
}
