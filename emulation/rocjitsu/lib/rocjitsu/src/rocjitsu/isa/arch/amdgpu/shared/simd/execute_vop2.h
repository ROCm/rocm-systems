// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP2_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP2_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

/// VOP2 binary SIMD fast path. Returns true when the SIMD path executed
/// and the caller should skip its scalar per-lane loop; false on the
/// `force_scalar` override or when any operand reports `!simd_capable()`.
/// Constrained on `util::has_stdx_simd`; an unconstrained overload
/// below returns false unconditionally when the constraint cannot be
/// satisfied, so callers can write the probe without an `if constexpr`
/// guard.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  // Resolve operand views once. Read-view acquisition observes plugin-visible
  // VGPR reads; write-view acquisition is write-only. Operands that are not
  // contiguous VGPR storage carry their scalar broadcast/fallback behavior in
  // the view object.
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto r = bin_op(a, b);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback selected when `util::has_stdx_simd` is false.
/// Trivially inlined to `return false;` so the generated probe at the
/// call site costs nothing on toolchains without `<experimental/simd>`.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop2_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP2 carry SIMD fast path (v_add_co/sub_co/subrev_co/addc/subb/subbrev_u32).
/// The lane type is fixed to uint32_t. `carry_op` is invoked as
///   carry_op(native<uint32_t> src0, native<uint32_t> vsrc1, native<uint32_t> cin)
///     -> SimdCarry<native<uint32_t>, mask>
/// where `cin` carries the incoming VCC bit (0/1 per lane); ops without a
/// carry-in ignore it. The result is masked-stored to vdst and the carry mask
/// is returned through `write_result`, which owns the architectural VCC commit
/// after any DPP source-validity merge. Inactive-lane bits remain zero.
template <bool ReadCarry, typename Inst, typename CarryOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_carry_simd(Inst &inst, Wavefront &wf,
                                                             CarryOp carry_op,
                                                             WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  // Carry-in reads the incoming VCC; the result accumulates from zero so that
  // inactive lanes are zeroed (matching hardware and the scalar bodies).
  const uint64_t vcc_in = ReadCarry ? wf.vcc_mask(exec) : 0;
  uint64_t vcc_out = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    // Expand the incoming VCC bits for this chunk to a 0/1-per-lane vector.
    const uint64_t cin_bits = (vcc_in >> base) & chunk_full;
    const auto cin = util::simd_u32_lanes_from_bits(cin_bits);
    const auto r = carry_op(a, b, cin);
    dst.template store_native<T>(base, r.value, chunk);
    // Pack the per-lane carry mask into the low W bits, then merge into VCC for
    // active lanes only (clear active bits, set from carry; preserve the rest).
    const uint64_t carry_bits = util::simd_mask_to_bits(r.carry);
    vcc_out = (vcc_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  write_result(vcc_out);
  return true;
}

/// Unconstrained fallback for the carry path; see the binary-path note above.
template <bool ReadCarry, typename Inst, typename CarryOp, typename WriteResult>
[[nodiscard]] bool try_execute_binary_vop2_carry_simd(Inst &, Wavefront &, CarryOp, WriteResult) {
  return false;
}

/// VOP2 ternary (fused multiply-add) SIMD fast path for literal-addend and
/// literal-multiplier forms. Reads src0/vsrc1 and writes vdst; it deliberately
/// does not read vdst. The accumulator-shaped VOP2 FMA/MAC forms use
/// try_execute_ternary_vop2_acc_simd below so register observation follows the
/// instruction-visible read set exactly.
///
/// F32 operations use fma_f32_simd to preserve architectural NaN payloads.
template <typename T, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop2_simd(Inst &inst, Wavefront &wf,
                                                        util::native<T> k, FmaOp fma_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto d = util::native<T>{};
    const auto r = fma_op(a, b, d, k);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the ternary path; see the binary-path note above.
template <typename T, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop2_simd(Inst &, Wavefront &, util::native<T>, FmaOp) {
  return false;
}

/// VOP2 ternary SIMD fast path for dst-accumulate FMA/MAC forms:
/// `dst = fma(src0, vsrc1, dst)`. The destination is acquired as a read-write
/// view so read observation and write resolution use the same write-role VGPR
/// mapping.
template <typename T, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop2_acc_simd(Inst &inst, Wavefront &wf,
                                                            util::native<T> k, FmaOp fma_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto d = acc.template load_native<T>(base);
    const auto r = fma_op(a, b, d, k);
    acc.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename T, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop2_acc_simd(Inst &, Wavefront &, util::native<T>, FmaOp) {
  return false;
}

/// @brief MODE-aware VOP2 F16 fused multiply-add SIMD fast path.
template <F16Vop2FmaShape Shape, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fma_vop2_f16_simd(Inst &inst, Wavefront &wf,
                                                        uint32_t literal = 0) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  RegisterAccess::OperandReadView src0 = regs.read_operand(inst.src0, exec);
  RegisterAccess::OperandReadView src1 = regs.read_operand(inst.vsrc1, exec);
  const auto literal_value = util::broadcast<T>(literal);
  if constexpr (Shape == F16Vop2FmaShape::Accumulate) {
    RegisterAccess::OperandReadWriteView acc = regs.readwrite_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto result = fma_f16_mode_simd(
          src0.template load_native<T>(base), src1.template load_native<T>(base),
          acc.template load_native<T>(base), false, false, false, false, false, false,
          wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), 0, false, wf.fp16_ovfl(),
          floating_clamp_nan_to_zero(wf), fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()));
      acc.template store_native<T>(base, result, chunk);
    }
  } else {
    RegisterAccess::OperandWriteView dst = regs.write_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto a = src0.template load_native<T>(base);
      const auto b = src1.template load_native<T>(base);
      const auto result =
          Shape == F16Vop2FmaShape::AddLiteral
              ? fma_f16_mode_simd(a, b, literal_value, false, false, false, false, false, false,
                                  wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), 0, false,
                                  wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf),
                                  fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()))
              : fma_f16_mode_simd(a, literal_value, b, false, false, false, false, false, false,
                                  wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), 0, false,
                                  wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf),
                                  fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()));
      dst.template store_native<T>(base, result, chunk);
    }
  }
  return true;
}

template <F16Vop2FmaShape Shape, typename Inst>
[[nodiscard]] bool try_execute_fma_vop2_f16_simd(Inst &, Wavefront &, uint32_t = 0) {
  return false;
}

/// 64-bit-lane VOP2 fused-multiply-add SIMD fast path for the v_fmac_f64
/// dst-accumulate form: `dst = fma(src0, vsrc1, dst)`. Reads all three operands
/// as `native<T>` (T = double) through the split lo/hi VGPR-pair path and
/// masked-stores the result. fma_f64_mode_simd executes the native<double>
/// fused operation under the architectural FP_ROUND/FP_DENORM mode and
/// replaces NaN-input lanes with the scalar helper result so NaN payload policy
/// remains exact.
template <typename T, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop2_f64_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  RegisterAccess::OperandRead64View src0 = regs.read_operand64(inst.src0, exec);
  RegisterAccess::OperandRead64View src1 = regs.read_operand64(inst.vsrc1, exec);
  RegisterAccess::OperandReadWrite64View acc = regs.readwrite_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto d = acc.template load_native<T>(base); // dst-accumulate
    const auto result =
        fma_f64_mode_simd(a, b, d, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64());
    acc.template store_native<T>(base, result, chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 ternary path; see the binary-path note.
template <typename T, typename Inst>
[[nodiscard]] bool try_execute_ternary_vop2_f64_simd(Inst &, Wavefront &) {
  return false;
}

/// 64-bit-lane VOP2 binary SIMD fast path (v_add_f64 / v_mul_f64 /
/// v_max_num_f64 / v_min_num_f64). VOP2 has no abs/neg/omod/clamp fields and
/// reads its second source as `vsrc1` (not `src1`); otherwise identical to the
/// f64 FMA vop2 path minus the dst-accumulate operand. add/mul are bit-exact;
/// fmax/fmin carry the accepted NaN-payload / signed-zero-tie carve-out (same as
/// the f64 vop3 binary path and every other min/max).
template <typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_f64_simd(Inst &inst, Wavefront &wf,
                                                           BinOp bin_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.vsrc1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    dst.template store_native<T>(base, bin_op(a, b), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 vop2 binary path.
template <typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop2_f64_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// v_cndmask_b32 SIMD fast path: dst[lane] = (VCC bit) ? vsrc1 : src0. VCC is an
/// input side-channel here (no carry-out). The per-lane select bits for a chunk
/// are read from VCC at the chunk's bit offset, expanded to a 0/1-per-lane
/// vector, and used to blend src0/vsrc1 with `where`. A pure 32-bit bit select,
/// so the result is bit-identical to the scalar body for every input.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cndmask_vop2_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t vcc = wf.vcc_mask(exec);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t sel_bits = (vcc >> base) & chunk_full;
    auto r = a;
    util::stdx::where(util::simd_mask_from_bits<util::native<T>>(sel_bits), r) = b;
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the cndmask path; see the binary-path note above.
template <typename Inst> [[nodiscard]] bool try_execute_cndmask_vop2_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP2/VOP3 dst-accumulate integer dot product (the "c" forms:
/// v_dot2c_i32_i16, v_dot4c_i32_i8, v_dot8c_i32_i4). The accumulator is the
/// DESTINATION register (inst.vdst), read as the accumulate source and written
/// as the result. VOP2 reads its 2nd source as inst.vsrc1, VOP3 as inst.src1
/// (selected by the Vop3 flag via if constexpr). All *c int forms are signed;
/// products fit in int32, while accumulation wraps in uint32_t bits. The scalar
/// bodies carry no op_sel/neg/clamp.
template <int ElemBits, bool Vop3, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_dotc_int_simd(Inst &inst, Wavefront &wf) {
  static_assert(ElemBits == 16 || ElemBits == 8 || ElemBits == 4, "dotc ElemBits must be 16/8/4");
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  if constexpr (Vop3) {
    if (!inst.src1.simd_capable())
      return false;
  } else {
    if (!inst.vsrc1.simd_capable())
      return false;
  }
  constexpr int N = 32 / ElemBits;
  constexpr uint32_t kElemMask = (ElemBits == 16) ? 0xFFFFu : (ElemBits == 8) ? 0xFFu : 0xFu;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  using U = util::native<uint32_t>;
  using I = util::native<int32_t>;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = [&]() {
    if constexpr (Vop3)
      return regs.read_operand(inst.src1, exec);
    else
      return regs.read_operand(inst.vsrc1, exec);
  }();
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = src0.template load_native<uint32_t>(base);
    const U raw1 = src1.template load_native<uint32_t>(base);
    U sum = acc.template load_native<uint32_t>(base); // accumulator from dst
    for (int i = 0; i < N; ++i) {
      const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
      const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
      const I a = std::bit_cast<I>(simd_sign_extend_u32(ea, ElemBits));
      const I b = std::bit_cast<I>(simd_sign_extend_u32(eb, ElemBits));
      sum += std::bit_cast<U>(a * b);
    }
    acc.template store_native<uint32_t>(base, sum, chunk);
  }
  return true;
}

template <int ElemBits, bool Vop3, typename Inst>
[[nodiscard]] bool try_execute_dotc_int_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP2/VOP3 dst-accumulate f16 dot product (v_dot2c_f32_f16). Two f16 products
/// of src0 and src1/vsrc1 (widened via f16_to_f32_simd) plus the f32
/// accumulator read from the DESTINATION register. Bracketing matches the
/// scalar `facc += a0*b0 + a1*b1` exactly: acc + ((a0*b0)+(a1*b1)). No
/// op_sel/neg/clamp. NaN-payload divergence accepted (shared f16 carve-out).
template <bool Vop3, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_dotc_f16_simd(Inst &inst, Wavefront &wf) {
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3 || wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5)
    return false;
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  if constexpr (Vop3) {
    if (!inst.src1.simd_capable())
      return false;
  } else {
    if (!inst.vsrc1.simd_capable())
      return false;
  }
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = [&]() {
    if constexpr (Vop3)
      return regs.read_operand(inst.src1, exec);
    else
      return regs.read_operand(inst.vsrc1, exec);
  }();
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = src0.template load_native<uint32_t>(base);
    const U raw1 = src1.template load_native<uint32_t>(base);
    const F acc_value = std::bit_cast<F>(acc.template load_native<uint32_t>(base));
    const F a0 = util::f16_to_f32_simd(raw0 & 0xFFFFu);
    const F a1 = util::f16_to_f32_simd(raw0 >> 16);
    const F b0 = util::f16_to_f32_simd(raw1 & 0xFFFFu);
    const F b1 = util::f16_to_f32_simd(raw1 >> 16);
    const F r = acc_value + (a0 * b0 + a1 * b1);
    acc.template store_native<uint32_t>(base, std::bit_cast<U>(r), chunk);
  }
  return true;
}

template <bool Vop3, typename Inst>
[[nodiscard]] bool try_execute_dotc_f16_simd(Inst &, Wavefront &) {
  return false;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP2_H_
