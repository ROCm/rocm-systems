// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Operand-aware SIMD glue for the auto-generated execute_<mnemonic>
// kernels in execute_shared.h. Hand-maintained; lives separately from
// the generated header so the same code is not duplicated in
// simd_codegen.py (raw string) and execute_shared.h (emitted output).
//
// Layering: this header sees rocjitsu types (Wavefront, plus the Op /
// Inst template parameters) and bridges to the generic util SIMD
// primitives in util/simd.h. The generic util layer never depends on
// rocjitsu; only this direction is permitted.

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/probes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/select.h"

/// Probe macro emitted at the top of each SIMD-eligible execute_<mnemonic>
/// kernel by simd_codegen.py. Expands to the binary-VOP2 fast-path call and
/// an early `return` on success, keeping each generated body to a single
/// line. Variadic in the operator argument so functor lambdas (which contain
/// commas) pass through as one token sequence. Relies on the kernel's `inst`
/// and `wf` parameters being in scope.
#define ROCJITSU_TRY_SIMD_VOP2_BINARY(T, ...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_simd<T>(inst, wf, __VA_ARGS__))                  \
  return

/// VOP1 unary counterpart of ROCJITSU_TRY_SIMD_VOP2_BINARY. Variadic in the
/// operator argument so functor lambdas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOP1_UNARY(Tin, Tout, ...)                                               \
  if (::rocjitsu::amdgpu::try_execute_unary_vop1_simd<Tin, Tout>(inst, wf, __VA_ARGS__))           \
  return

/// Carry-VOP2 counterpart. The wrapper-owned writer merges DPP-suppressed
/// lanes and applies the target wave-width policy before the VCC commit.
#define ROCJITSU_TRY_SIMD_VOP2_CARRY_RESULT(WRITE_RESULT, READ_CARRY, ...)                         \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_carry_simd<READ_CARRY>(inst, wf, __VA_ARGS__,    \
                                                                         WRITE_RESULT))            \
  return

/// Literal FMA/MAD VOP2 counterpart. `KEXPR` is the inline-literal bits
/// broadcast to every lane before the call. Variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP2_TERNARY(T, KEXPR, ...)                                              \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_simd<T>(inst, wf, ::util::broadcast<T>(KEXPR),  \
                                                           __VA_ARGS__))                           \
  return

/// Dst-accumulate FMA/MAC VOP2 counterpart. The helper acquires vdst as a
/// read-write register view, so only these forms observe a vdst read.
#define ROCJITSU_TRY_SIMD_VOP2_TERNARY_ACC(T, KEXPR, ...)                                          \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_acc_simd<T>(                                    \
          inst, wf, ::util::broadcast<T>(KEXPR), __VA_ARGS__))                                     \
  return

#define ROCJITSU_TRY_SIMD_VOP2_FMA_F16_ADD_LITERAL(KEXPR)                                          \
  if (::rocjitsu::amdgpu::try_execute_fma_vop2_f16_simd<                                           \
          ::rocjitsu::amdgpu::F16Vop2FmaShape::AddLiteral>(inst, wf, KEXPR))                       \
  return

#define ROCJITSU_TRY_SIMD_VOP2_FMA_F16_MULTIPLY_LITERAL(KEXPR)                                     \
  if (::rocjitsu::amdgpu::try_execute_fma_vop2_f16_simd<                                           \
          ::rocjitsu::amdgpu::F16Vop2FmaShape::MultiplyLiteral>(inst, wf, KEXPR))                  \
  return

#define ROCJITSU_TRY_SIMD_VOP2_FMAC_F16()                                                          \
  if (::rocjitsu::amdgpu::try_execute_fma_vop2_f16_simd<                                           \
          ::rocjitsu::amdgpu::F16Vop2FmaShape::Accumulate>(inst, wf))                              \
  return

/// v_cndmask_b32 counterpart. Fixed op (VCC-driven select), so no type or
/// functor argument.
#define ROCJITSU_TRY_SIMD_VOP2_CNDMASK()                                                           \
  if (::rocjitsu::amdgpu::try_execute_cndmask_vop2_simd(inst, wf))                                 \
  return

/// v_cndmask_b32 VOP3 counterpart. Same shape, but the selector comes from the
/// wave-mask scalar `src2` instead of fixed VCC.
#define ROCJITSU_TRY_SIMD_VOP3_CNDMASK()                                                           \
  if (::rocjitsu::amdgpu::try_execute_cndmask_vop3_simd(inst, wf))                                 \
  return

/// 16-bit variant of v_cndmask_b32 VOP3 (RDNA3+). Same select + low-16 mask
/// on the result.
#define ROCJITSU_TRY_SIMD_VOP3_CNDMASK_B16()                                                       \
  if (::rocjitsu::amdgpu::try_execute_cndmask_b16_vop3_simd(inst, wf))                             \
  return

/// 64-bit-lane MODE-aware VOP2 FMA counterpart (v_fmac_f64).

/// 64-bit-lane VOP2 binary counterpart (v_add/mul/max_num/min_num_f64). Lane
/// type fixed to double, read/written through the split lo/hi VGPR-pair path
/// (vsrc1 as the second source). Variadic in the functor.

/// 64-bit-lane VOP1 unary counterpart. `T` is the 64-bit lane type (`double`
/// for the f64 math ops, `uint64_t` for v_mov_b64). Variadic in the functor so
/// its commas pass through as one token sequence.

/// Mixed-width cvt counterpart, f64 source -> 32-bit dst. `Tout` is the 32-bit
/// result lane type; the functor (`native<double> -> narrow32<Tout>`) is variadic
/// so its commas pass through as one token sequence.

/// Mixed-width cvt counterpart, 32-bit source -> f64 dst. `Tin` is the 32-bit
/// source lane type; the functor (`narrow32<Tin> -> native<double>`) is variadic.

/// VOP3 f64-source -> 32-bit-fp-dst cvt counterpart (src0 abs/neg + result
/// omod/clamp; for v_frexp_exp_i32_f64). Functor: native<double> -> narrow32<float>.

/// VOP3 v_cvt_f32_f16 counterpart. Generic form reads src0.l; TRUE16 form
/// selects src0 via op_sel[0]. Both write a full f32 dword.
#define ROCJITSU_TRY_SIMD_CVT_F32_F16_VOP3()                                                       \
  if (::rocjitsu::amdgpu::try_execute_cvt_f32_f16_vop3_simd<false>(inst, wf))                      \
  return

#define ROCJITSU_TRY_SIMD_CVT_F32_F16_VOP3_TRUE16()                                                \
  if (::rocjitsu::amdgpu::try_execute_cvt_f32_f16_vop3_simd<true>(inst, wf))                       \
  return

/// VOPC compare counterpart. `T` is the 32-bit lane read type; the comparison
/// functor (which may convert/narrow inside) is variadic so its commas pass
/// through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOPC(T, ...)                                                             \
  if (::rocjitsu::amdgpu::try_execute_vopc_simd<T>(inst, wf, __VA_ARGS__))                         \
  return

/// 64-bit-lane VOPC compare counterpart (f64/i64/u64). `T` is the 64-bit lane
/// read type; the comparison functor is variadic so its commas pass through.

/// Mixed-width v_cmp_class_f64 counterpart (64-bit value, 32-bit mask). No type
/// argument; the class functor `(native<uint64_t> bits, narrow32<uint32_t> mask)
/// -> mask` is variadic so its commas pass through as one token sequence.

/// VOP3 v_cmp_class_f16/f32 counterpart (32-bit value, abs/neg modifiers, src1
/// mask, SGPR-pair dst). `SM` is the per-op sign-bit mask (0x8000 / 0x80000000);
/// the class functor is variadic so its commas pass through.
#define ROCJITSU_TRY_SIMD_VOP3_CLASS_B32(SM, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_b32_simd<false>(                                  \
          inst, wf, SM, __VA_ARGS__, [&](uint64_t result) {                                        \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return

#define ROCJITSU_TRY_SIMD_VOP3_CLASS_B32_RESULT(WRITE_RESULT, SM, ...)                             \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_b32_simd<false>(inst, wf, SM, __VA_ARGS__,        \
                                                                 WRITE_RESULT))                    \
  return

/// VOP3 f16 class counterpart for true16 OPSEL source/mask halves.
#define ROCJITSU_TRY_SIMD_VOP3_CLASS_TRUE16_B32(SM, ...)                                           \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_b32_simd<true>(                                   \
          inst, wf, SM, __VA_ARGS__, [&](uint64_t result) {                                        \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return

#define ROCJITSU_TRY_SIMD_VOP3_CLASS_TRUE16_B32_RESULT(WRITE_RESULT, SM, ...)                      \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_b32_simd<true>(inst, wf, SM, __VA_ARGS__,         \
                                                                WRITE_RESULT))                     \
  return

/// VOP3 v_cmp_class_f64 counterpart (64-bit value). `SM` is the f64 sign-bit mask
/// (0x8000000000000000); the class functor is variadic.

/// VOP3 integer/bitwise binary counterpart (reads src0/src1, no modifiers).
/// `T` is the 32-bit integer lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_INT(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_simd<T>(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 binary counterpart for true16 source selectors with a full b32 result,
/// such as v_pack_b32_f16.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_SRC(T, ...)                                           \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_true16_src_simd<T>(inst, wf, __VA_ARGS__))       \
  return

/// VOP3 f16 binary counterpart: same packed path as the integer form, but bails
/// to the (modifier-applying) scalar body when any abs/neg/omod/clamp field is
/// set. `T` is the 32-bit packed lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_F16(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_f16_simd<false, T>(inst, wf, __VA_ARGS__))       \
  return

/// VOP3 f16 binary counterpart for true16 OPSEL source/destination halves.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_TRUE16_F16(T, ...)                                           \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_f16_simd<true, T>(inst, wf, __VA_ARGS__))        \
  return

/// VOP3 f32 binary counterpart (reads src0/src1, applies abs/neg/omod/clamp).
/// `T` is the 32-bit float lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_FP(T, ...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_fp_simd<T>(inst, wf, __VA_ARGS__))               \
  return

/// VOP3 f32 unary counterpart (reads src0, applies abs/neg/omod/clamp). `Tin`
/// and `Tout` are both float32_t; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(Tin, Tout, ...)                                            \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp_simd<Tin, Tout>(inst, wf, __VA_ARGS__))        \
  return

/// VOP3 raw-lane VOPC compare counterpart (32-bit lane, SGPR-pair dst). Despite
/// the _INT name, f16/f32 relations use it too: the glue applies no modifiers,
/// and a float functor applies the captured ABS/NEG fields and MODE input
/// flush itself via comparison::evaluate. `T` is the 32-bit raw lane read type;
/// variadic in the functor so its commas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_INT(T, ...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T>(                                       \
          inst, wf, __VA_ARGS__, [&](uint64_t result) {                                            \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_INT_RESULT(WRITE_RESULT, T, ...)                               \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T>(inst, wf, __VA_ARGS__, WRITE_RESULT))  \
  return

/// VOP3 raw-lane compare counterpart whose 16-bit sources select their half
/// with true16 OPSEL.
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_TRUE16_INT(T, ...)                                             \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T, true>(                                 \
          inst, wf, __VA_ARGS__, [&](uint64_t result) {                                            \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_TRUE16_INT_RESULT(WRITE_RESULT, T, ...)                        \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T, true>(inst, wf, __VA_ARGS__,           \
                                                                  WRITE_RESULT))                   \
  return

/// 64-bit-lane VOP3 raw-lane VOPC compare counterpart (i64/u64/f64, SGPR-pair
/// dst). As with the 32-bit macro, f64 callers use it too, with the functor
/// applying ABS/NEG and MODE input flush via comparison::evaluate. `T` is the
/// 64-bit raw lane read type; variadic in the functor.

/// VOP3 integer/bitwise ternary counterpart (reads src0/src1/src2, no
/// modifiers). `T` is the 32-bit integer lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT(T, ...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_simd<T>(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 ternary counterpart for true16 SRC0/SRC1 plus full-width SRC2/result.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_SRC01(T, ...)                                        \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_true16_src01_simd<T>(inst, wf, __VA_ARGS__))    \
  return

/// VOP3 ternary counterpart for true16 SRC0/SRC1/SRC2 plus selected-half dst.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16(T, ...)                                              \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_true16_simd<T>(inst, wf, __VA_ARGS__))          \
  return

/// VOP3 f32 ternary counterpart (per-source abs/neg, result omod/clamp).
/// Functor takes already-modified `native<float>` arguments; variadic.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 f16 ternary counterpart (raw uint32 lanes; widen f16->f32 each src,
/// abs/neg, op, omod/clamp, narrow). Variadic.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp16_simd<false>(inst, wf, __VA_ARGS__))        \
  return

/// VOP3 f16 ternary counterpart for true16 source and destination halves.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_TRUE16_FP16(...)                                            \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp16_simd<true>(inst, wf, __VA_ARGS__))         \
  return

#define ROCJITSU_TRY_SIMD_FMA_VOP3_FP16()                                                          \
  if (::rocjitsu::amdgpu::try_execute_fma_vop3_fp16_simd<false>(inst, wf))                         \
  return

#define ROCJITSU_TRY_SIMD_FMA_VOP3_TRUE16_FP16()                                                   \
  if (::rocjitsu::amdgpu::try_execute_fma_vop3_fp16_simd<true>(inst, wf))                          \
  return

/// VOP3 f64 ternary counterpart (64-bit RegisterAccess reads, per-source
/// abs/neg, omod/clamp).
/// Variadic.

/// VOP3 dst-accumulate FMA counterpart (f32). Per-isa class has no src2;
/// vdst is the accumulator. abs/neg apply to src0/src1 only.
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp_simd(inst, wf, __VA_ARGS__))                    \
  return

/// VOP3 dst-accumulate FMA counterpart (f16). Widen chain, vdst is the
/// (widened) accumulator.
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP16(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp16_simd<false>(inst, wf, __VA_ARGS__))           \
  return

/// VOP3 dst-accumulate f16 counterpart for true16 source/accumulator/dst halves.
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_TRUE16_FP16(...)                                               \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp16_simd<true>(inst, wf, __VA_ARGS__))            \
  return

#define ROCJITSU_TRY_SIMD_FMAC_VOP3_MODE_FP16()                                                    \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp16_mode_simd<false>(inst, wf))                   \
  return

#define ROCJITSU_TRY_SIMD_FMAC_VOP3_MODE_TRUE16_FP16()                                             \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp16_mode_simd<true>(inst, wf))                    \
  return

/// VOP3 dst-accumulate FMA counterpart (f64).

/// VOP3 ldexp counterpart (f32 src0 + int32 src1 exp). Variadic functor.
#define ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP32(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_ldexp_vop3_fp32_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 ldexp counterpart (f64 src0 + int32 src1 exp). Variadic functor.

/// VOP3 fused division scaling with explicit MODE controls and no OMOD/CLAMP.
#define ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP32()                                                     \
  if (::rocjitsu::amdgpu::try_execute_div_fmas_f32_simd(inst, wf))                                 \
  return

/// VOP3 f64 binary counterpart (64-bit RegisterAccess reads, per-source
/// abs/neg, omod/clamp on the result). Variadic in the functor so its commas pass through as one
/// token sequence.

/// VOP3 f64 unary counterpart (64-bit RegisterAccess read, src0 abs/neg,
/// omod/clamp on the result). Variadic in the functor.

/// VOP3 f16 unary counterpart (raw uint32 lanes; widen f16->f32, src0 abs/neg,
/// op, omod/clamp, narrow f32->f16). The functor takes already-widened-and-
/// modified `native<float>` and returns `native<float>`; variadic.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp16_simd<false>(inst, wf, __VA_ARGS__))          \
  return

/// VOP3 f16 unary counterpart for true16 source and destination halves.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_TRUE16_FP16(...)                                              \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp16_simd<true>(inst, wf, __VA_ARGS__))           \
  return

/// VOP3 64-bit reverse-shift counterpart (v_lshlrev_b64 / v_lshrrev_b64 /
/// v_ashrrev_i64). src0 = 32-bit shift, src1 = 64-bit value; the functor takes
/// `(native<uint64_t> value, native<uint64_t> shift)`. Variadic.
#define ROCJITSU_TRY_SIMD_SHIFT64_VOP3(...)                                                        \
  if (::rocjitsu::amdgpu::try_execute_shift64_vop3_simd(inst, wf, __VA_ARGS__))                    \
  return

/// VOP3 v_lshl_add_u64 counterpart. Fixed op ((src0 << (src1 & 63)) + src2).
#define ROCJITSU_TRY_SIMD_LSHL_ADD_U64()                                                           \
  if (::rocjitsu::amdgpu::try_execute_lshl_add_u64_simd(inst, wf))                                 \
  return

/// VOP3 wide 32x32->64 multiply-add counterpart (v_mad_u64_u32 / v_mad_i64_i32).
/// The functor takes `(narrow32<uint32_t> s0, narrow32<uint32_t> s1,
/// native<uint64_t> c)` and returns `SimdCarry<native<uint64_t>, mask>`;
/// variadic so its commas pass through.

/// VOP3 carry-OUT counterpart (no carry-in; carry-out to SGPR sdst). Lane type
/// fixed to uint32_t; variadic in the SimdCarry functor.
#define ROCJITSU_TRY_SIMD_VOP3_CO(...)                                                             \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_co_simd(inst, wf, __VA_ARGS__))                  \
  return
#define ROCJITSU_TRY_SIMD_VOP3_CO_RESULT(WRITE_RESULT, ...)                                        \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_co_result_simd(inst, wf, __VA_ARGS__,            \
                                                                 WRITE_RESULT))                    \
  return

/// VOP3 carry-IN counterpart (carry-in from SGPR src2, carry-out to SGPR sdst).
/// Lane type fixed to uint32_t; variadic in the SimdCarry functor.
#define ROCJITSU_TRY_SIMD_VOP3_CIN(...)                                                            \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_cin_simd(inst, wf, __VA_ARGS__))                 \
  return
#define ROCJITSU_TRY_SIMD_VOP3_CIN_RESULT(WRITE_RESULT, ...)                                       \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_cin_result_simd(inst, wf, __VA_ARGS__,           \
                                                                  WRITE_RESULT))                   \
  return

/// VOP3P fma_mix / mad_mix probes. The destination shape is the only thing
/// that differs across the six ops, so the probe is parameterised by
/// `FmaMixDst::{F32,F16_LO,F16_HI}` and the shared scalar formula `a*b+c`
/// (incl. clamp) lives in the glue. No functor argument.
#define ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F32()                                                      \
  if (::rocjitsu::amdgpu::try_execute_vop3p_fma_mix_simd<::rocjitsu::amdgpu::FmaMixDst::F32>(inst, \
                                                                                             wf))  \
  return

#define ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_LO()                                                   \
  if (::rocjitsu::amdgpu::try_execute_vop3p_fma_mix_simd<::rocjitsu::amdgpu::FmaMixDst::F16_LO>(   \
          inst, wf))                                                                               \
  return

#define ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_HI()                                                   \
  if (::rocjitsu::amdgpu::try_execute_vop3p_fma_mix_simd<::rocjitsu::amdgpu::FmaMixDst::F16_HI>(   \
          inst, wf))                                                                               \
  return

/// VOP3P packed-16 integer binary probe. Functor takes two u32 simd vectors
/// (each holding {low16, high16} packed) and returns the same shape with
/// the per-half op applied. The glue applies half selectors and retains
/// scalar execution for integer CLAMP.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_INT(...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_int_simd(inst, wf, __VA_ARGS__))             \
  return

/// VOP3P packed-16 integer ternary probe (3-source pk_mad family).
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_INT(...)                                                \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_int_simd(inst, wf, __VA_ARGS__))            \
  return

/// VOP3P packed-f32 binary probe (v_pk_add_f32 / v_pk_mul_f32). Functor takes
/// (a, b) as native<float> (neg-applied) and returns native<float>.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32(...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_f32_simd(inst, wf, __VA_ARGS__))             \
  return

/// Profile-aware form for ISA-local VOP3P bodies whose selector fields do not
/// use the canonical op_sel/op_sel_hi member names.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32_SELECTORS(OpSel, OpSelHi, ...)                       \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_f32_simd(                                    \
          inst, wf, static_cast<uint32_t>(OpSel), static_cast<uint32_t>(OpSelHi), __VA_ARGS__))    \
  return

/// VOP3P packed-f32 ternary probe (v_pk_fma_f32). Functor takes (a, b, c) as
/// native<float>; selectors choose each source half, including broadcasts.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32(...)                                                \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_f32_simd(inst, wf, __VA_ARGS__))            \
  return

/// Profile-aware form for ISA-local ternary VOP3P bodies whose selector fields
/// do not use the canonical op_sel/op_sel_hi/op_sel_hi_2 member names.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32_SELECTORS(OpSel, OpSelHi, OpSelHi2, ...)            \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_f32_simd(                                   \
          inst, wf, static_cast<uint32_t>(OpSel), static_cast<uint32_t>(OpSelHi),                  \
          static_cast<uint32_t>(OpSelHi2), __VA_ARGS__))                                           \
  return

/// VOP3P v_pk_mov_b32 probe. Functorless / fixed-op.
#define ROCJITSU_TRY_SIMD_VOP3P_MOV_B32()                                                          \
  if (::rocjitsu::amdgpu::try_execute_vop3p_mov_b32_simd(inst, wf))                                \
  return

/// VOP3P integer dot-product probe. Args: (ElemBits, Signed) — e.g.
/// (8, true) for v_dot4_i32_i8, (4, false) for v_dot8_u32_u4. Functorless.
#define ROCJITSU_TRY_SIMD_VOP3P_DOT_INT(...)                                                       \
  if (::rocjitsu::amdgpu::try_execute_vop3p_dot_int_simd<__VA_ARGS__>(inst, wf))                   \
  return

/// VOP3P v_dot2_f32_{f16,bf16} SIMD probe. Arg: the half-precision widening
/// format (F16 or BF16) as a ::rocjitsu::amdgpu::Vop3pDotHalfFormat enumerator.
#define ROCJITSU_TRY_SIMD_VOP3P_DOT_F16(Fmt)                                                       \
  if (::rocjitsu::amdgpu::try_execute_vop3p_dot_f16_simd<                                          \
          ::rocjitsu::amdgpu::Vop3pDotHalfFormat::Fmt>(inst, wf))                                  \
  return

/// VOP3P mixed-sign integer dot probe (v_dot4_i32_iu8 / v_dot8_i32_iu4). Arg:
/// ElemBits (8 or 4). Per-operand sign read at runtime from inst.neg.
#define ROCJITSU_TRY_SIMD_VOP3P_DOT_INT_MIXED(ElemBits)                                            \
  if (::rocjitsu::amdgpu::try_execute_vop3p_dot_int_mixed_simd<ElemBits>(inst, wf))                \
  return

/// VOP2/VOP3 dst-accumulate integer dot probe (the "c" forms). Args:
/// (ElemBits, Vop3) — e.g. (8, true) for v_dot4c_i32_i8_vop3.
#define ROCJITSU_TRY_SIMD_DOTC_INT(...)                                                            \
  if (::rocjitsu::amdgpu::try_execute_dotc_int_simd<__VA_ARGS__>(inst, wf))                        \
  return

/// VOP2/VOP3 dst-accumulate f16 dot probe (v_dot2c_f32_f16). Arg: Vop3 bool.
#define ROCJITSU_TRY_SIMD_DOTC_F16(...)                                                            \
  if (::rocjitsu::amdgpu::try_execute_dotc_f16_simd<__VA_ARGS__>(inst, wf))                        \
  return

/// Exact packed F16/BF16 arithmetic, selected by PackedFloatOp and Bf16.
#define ROCJITSU_TRY_SIMD_PACKED_FLOAT(Op, Bf16)                                                   \
  if (::rocjitsu::amdgpu::try_execute_packed_float_simd<::rocjitsu::amdgpu::PackedFloatOp::Op,     \
                                                        Bf16>(inst, wf))                           \
  return

/// Fused mixed-precision arithmetic with the selected destination format.
#define ROCJITSU_TRY_SIMD_FUSED_MIX(Dst)                                                           \
  if (::rocjitsu::amdgpu::try_execute_vop3p_fma_mix_simd<::rocjitsu::amdgpu::FmaMixDst::Dst,       \
                                                         true>(inst, wf))                          \
  return

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
