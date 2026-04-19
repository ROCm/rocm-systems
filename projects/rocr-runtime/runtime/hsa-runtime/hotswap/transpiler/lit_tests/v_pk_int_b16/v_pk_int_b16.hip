// ARCH: gfx1250
//
// Lift fixture for the VOP3P packed-pair `<2 x i16>` int family.
// Pins both opcodes the kerneldex corpus exercises today:
//
//   * v_pk_lshlrev_b16 — clshl_rev_16 SDAG: dst = src1 << (src0 & 15),
//     reversed-operand convention (shift count is src0, value is
//     src1), with the AMDGPU hardware clamp to the low 4 bits of the
//     shift count modelled explicitly as an `and i16 ..., 15` so the
//     LLVM `shl` semantics match the hardware semantics for every
//     legal input. Dominant corpus shape is the inline-literal shift
//     count (e.g. `0x60002` = `<lo=2, hi=6>`), which the AMDGPU
//     disassembler surfaces via `op.src(i)` returning a non-register
//     i32 value bitcast directly to `<2 x i16>` (NO broadcast — the
//     literal width matches the operand width).
//
//   * v_pk_add_u16 — lane-wise i16 add. Same operand decode, only
//     the IR opcode differs.
//
// Co-locating both opcodes in the same kernel pins the shared
// operand-decode + write-back shape (handler factor in
// `transpiler/handle_valu_vop3p.cpp` under the
// `case SemOp::V_PK_ADD_U16: case SemOp::V_PK_LSHLREV_B16:` block).
// The matching SemOps live in `transpiler/semop.hpp` under VOP3P; the
// opcode map entries live in `opcode_map.cpp` under the VOP3P section.
//
// Sibling V_PK_LSHRREV_B16 / V_PK_ASHRREV_I16 / V_PK_SUB_U16 /
// V_PK_MUL_LO_U16 / V_PK_MAX_{I,U}16 / V_PK_MIN_{I,U}16 share the
// identical handler shape (one extra inner-switch case + IR opcode
// dispatch); they are intentionally NOT exercised here because the
// kerneldex corpus has zero producers for them today.

#include <hip/hip_runtime.h>

extern "C" __global__ void v_pk_int_b16_kernel(unsigned *out,
                                                const unsigned *in) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned val = in[tid];
  unsigned shifted, summed;
  // Inline-literal shift count: 0x60002 = <lo=2, hi=6>. Matches the
  // exact corpus shape from the failing _upcast_from_mxfp kernel.
  asm volatile("v_pk_lshlrev_b16 %[shifted], 0x60002, %[val]\n\t"
               "v_pk_add_u16 %[summed], %[val], %[shifted]\n\t"
               : [shifted] "=v"(shifted), [summed] "=v"(summed)
               : [val] "v"(val));
  out[2 * tid + 0] = shifted;
  out[2 * tid + 1] = summed;
}
