// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_H_

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

/// Execute the paired integer operations found in streaming-store kernels.
/// Resolve checked views once and load both slot results before either write,
/// since the two destinations may alias the other slot's sources. Opcode IDs
/// come from the generated ISA rather than being duplicated in this helper.
template <uint16_t kMovOp, uint16_t kAddOp, uint16_t kLshlOp, typename Slot>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopd_integer_pair_simd(Wavefront &wf, const Slot &x,
                                                             const Slot &y) {
  if (wf.wf_size() != 32 || simd_force_scalar())
    return false;
  const auto is_integer_slot = [](const Slot &slot) {
    if (slot.neg != 0 || slot.has_src2_operand || slot.src2_is_imm)
      return false;
    if (slot.op != kMovOp && slot.op != kAddOp && slot.op != kLshlOp)
      return false;
    return slot.dst->simd_capable() && slot.src0->simd_capable() &&
           (slot.op == kMovOp || slot.src1->simd_capable());
  };
  if (!is_integer_slot(x) || !is_integer_slot(y))
    return false;
  const uint64_t exec = static_cast<uint32_t>(wf.exec());
  if (exec == 0)
    return true;
  RegisterAccess registers(wf);
  const auto x_src0 = registers.read_operand(*x.src0, exec);
  const auto y_src0 = registers.read_operand(*y.src0, exec);
  std::optional<RegisterAccess::OperandReadView> x_src1;
  std::optional<RegisterAccess::OperandReadView> y_src1;
  if (x.op != kMovOp)
    x_src1.emplace(registers.read_operand(*x.src1, exec));
  if (y.op != kMovOp)
    y_src1.emplace(registers.read_operand(*y.src1, exec));
  const auto x_dst = registers.write_operand(*x.dst, exec);
  const auto y_dst = registers.write_operand(*y.dst, exec);
  const auto evaluate = [](const Slot &slot, const auto &src0_view, const auto &src1_view,
                           uint32_t lane_base) {
    const auto src0 = src0_view.template load_native<uint32_t>(lane_base);
    switch (slot.op) {
    case kMovOp:
      return src0;
    case kAddOp:
      return src0 + src1_view->template load_native<uint32_t>(lane_base);
    case kLshlOp:
      return simd_lshl_u32(src1_view->template load_native<uint32_t>(lane_base), src0);
    default:
      throw util::UnimplementedInst("unsupported VOPD integer SIMD operation");
    }
  };
  constexpr uint32_t kWidth = util::native_width_v<uint32_t>;
  for (uint32_t lane_base = 0; lane_base < wf.wf_size(); lane_base += kWidth) {
    const uint64_t lane_mask = (exec >> lane_base) & util::mask<uint64_t>(kWidth);
    if (lane_mask == 0)
      continue;
    const auto x_result = evaluate(x, x_src0, x_src1, lane_base);
    const auto y_result = evaluate(y, y_src0, y_src1, lane_base);
    x_dst.template store_native<uint32_t>(lane_base, x_result, lane_mask);
    y_dst.template store_native<uint32_t>(lane_base, y_result, lane_mask);
  }
  return true;
}

template <uint16_t kMovOp, uint16_t kAddOp, uint16_t kLshlOp, typename Slot>
[[nodiscard]] bool try_execute_vopd_integer_pair_simd(Wavefront &, const Slot &, const Slot &) {
  return false;
}

/// VOP1 unary SIMD fast path. Reads `src0` as `Tin`, applies `un_op`
/// (`native<Tin> -> native<Tout>`), masked-stores the result to `vdst` as
/// `Tout`. `Tin` and `Tout` are both 32-bit lane types (possibly different,
/// e.g. int32->float32 for v_cvt_f32_i32). Same contract as the VOP2 path:
/// returns true when the SIMD path executed; false on the `force_scalar`
/// override or when either operand reports `!simd_capable()`.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop1_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<Tout>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<Tin>(base);
    const auto r = un_op(a);
    dst.template store_native<Tout>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the unary path; see the binary-path note above.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_simd(Inst &, Wavefront &, UnOp) {
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

/// 64-bit-lane VOP1 unary SIMD fast path. The 64-bit counterpart of
/// try_execute_unary_vop1_simd: reads src0 as `native<T>` through a 64-bit
/// RegisterAccess operand view, applies `un_op` (`native<T> ->
/// native<T>`), and masked-stores the result to vdst. `T` is `double` for the
/// f64 math ops (ceil/floor/trunc/rndne/fract/rcp/rsq/sqrt) and `uint64_t` for
/// the pure 64-bit move (v_mov_b64). Same contract as the other paths: returns
/// true when the SIMD path executed; false on the `force_scalar` override or
/// when either operand reports `!simd_capable()`.
///
/// The rounding ops map to `vroundpd`, sqrt to `vsqrtpd`, and `1.0 / x` to
/// `vdivpd` — all correctly-rounded IEEE operations, bit-identical to the scalar
/// `std::ceil`/`std::sqrt`/... for every finite and infinite input. NaN-*input*
/// lanes may differ in propagated NaN payload (accepted — the result is a NaN
/// either way), guarded by the UtilSimd.*F64*_VectorMatchesScalar_BitExact tests.
template <typename T, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop1_f64_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    dst.template store_native<T>(base, un_op(a), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 unary path; see the binary-path note.
template <typename T, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_f64_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// Mixed-width VOP1 conversion SIMD fast path, f64 source -> 32-bit dst
/// (v_cvt_f32_f64, v_cvt_i32_f64, v_cvt_u32_f64). Reads src0 as `native<double>`
/// through a 64-bit RegisterAccess operand view, applies `cvt_op`
/// (`native<double> -> narrow32<Tout>`), and masked-stores the `native_width64`
/// 32-bit results to vdst. The double->Tout step is a single `static_simd_cast`
/// (cvt_f32_f64 maps to vcvtpd2ps, correctly rounded; the int forms clamp/NaN in
/// the double domain first, then one cast). Bit-identical to the scalar body for
/// finite/Inf inputs; a NaN *result* may differ in payload (accepted), which the
/// A/B test skips. Returns true when the SIMD path executed.
template <typename Tout, typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_f64_to_b32_simd(Inst &inst, Wavefront &wf, CvtOp cvt_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = src0.template load_native<double>(base);
    const util::narrow32<Tout> r = cvt_op(s);
    dst.template store_narrow<Tout>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64->b32 cvt path; see the binary-path note.
template <typename Tout, typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_f64_to_b32_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// Mixed-width VOP1 conversion SIMD fast path, 32-bit source -> f64 dst
/// (v_cvt_f64_f32, v_cvt_f64_i32, v_cvt_f64_u32). Reads src0 as `narrow32<Tin>`
/// (native_width64 32-bit lanes), applies `cvt_op` (`narrow32<Tin> ->
/// native<double>`), and masked-stores the result to the 64-bit vdst through a
/// 64-bit RegisterAccess write view. Each conversion is an exact widening
/// `static_simd_cast` (vcvtps2pd / int->double), bit-identical to the scalar body
/// for every input. Returns true when the SIMD path executed.
template <typename Tin, typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_b32_to_f64_simd(Inst &inst, Wavefront &wf, CvtOp cvt_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto in = src0.template load_narrow<Tin>(base);
    dst.template store_native<double>(base, cvt_op(in), chunk);
  }
  return true;
}

/// Unconstrained fallback for the b32->f64 cvt path; see the binary-path note.
template <typename Tin, typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_b32_to_f64_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// VOP3 mixed-width f64-source -> 32-bit-fp-dst fast path: the 64-bit-in /
/// 32-bit-out counterpart of the f64 unary FP glue, for v_frexp_exp_i32_f64
/// (the lone f64 VOP3 cvt whose body keeps the modifiers around an
/// f64 -> float(exp) -> bit_cast tail). Reads src0 as native<double>
/// (native_width64 lanes), applies src0 abs/neg in the f64 domain, runs cvt_op
/// (native<double> -> narrow32<float> = the exponent as float), then applies the
/// result omod/clamp INLINE at narrow32 width (apply_vop3_dst_mod_f32 is
/// native<float>-wide — a different width than narrow32<float>), and stores the
/// 32-bit results. All steps bit-exact: frexp_exp_f64_simd is the proven VOP1
/// helper (op 48), apply_vop3_src_mod_f64 is the same helper the tested f64
/// unary ops use, and the (float)(uint32) cast + power-of-two omod + ordered
/// clamp mirror the scalar tail (the exp is always finite, so NaN handling is
/// moot).
template <typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_vop3_f64_to_b32_fp_simd(Inst &inst, Wavefront &wf,
                                                                  CvtOp cvt_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = apply_vop3_src_mod_f64<0>(src0.template load_native<double>(base), abs, neg);
    util::narrow32<float> v = cvt_op(s);
    // Inline omod/clamp at narrow32<float> width (mirrors apply_vop3_dst_mod):
    // IEEE-exact power-of-two scale, ordered-compare saturation to [0,1].
    if (omod == 1)
      v = v * 2.0f;
    else if (omod == 2)
      v = v * 4.0f;
    else if (omod == 3)
      v = v * 0.5f;
    if (clamp) {
      if (floating_clamp_nan_to_zero(wf))
        util::stdx::where(util::stdx::isnan(v), v) = 0.0f;
      util::stdx::where(v <= 0.0f, v) = 0.0f;
      util::stdx::where(v > 1.0f, v) = 1.0f;
    }
    if (omod != 0) {
      // fixed_size_simd is not guaranteed to be trivially copyable, so avoid
      // whole-vector bit_cast here. This predicate selects both subnormals and
      // either signed zero, which OMOD maps to canonical +0.
      util::stdx::where(util::stdx::abs(v) < std::numeric_limits<float>::min(), v) = 0.0f;
    }
    dst.template store_narrow<float>(base, v, chunk);
  }
  return true;
}

template <typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_vop3_f64_to_b32_fp_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// VOP3 v_cvt_f32_f16 fast path. The generic form reads the low f16 half and
/// writes the full f32 dword, matching non-true16 scalar write_lane semantics.
/// The true16 form selects src0 with op_sel[0] before widening; the destination
/// is f32 in both cases, so no destination half merge is needed.
template <bool True16, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_f32_f16_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  std::optional<fp_mode::ScopedEnvironment> environment;
  if (omod || clamp)
    environment.emplace(wf.fp_round_mode_f32());
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto raw = src0.template load_native<T>(base);
    if constexpr (True16)
      raw = select_vop3_true16_src(raw, opsel, 0);
    else
      raw = raw & util::broadcast<T>(0xffffu);
    const auto src = apply_vop3_src_mod_f32<0>(cvt_f32_f16_mode_simd(raw, wf), abs, neg);
    const auto result = apply_vop3_dst_mod_f32(src, omod, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, std::bit_cast<util::native<T>>(result), chunk);
  }
  return true;
}

template <bool True16, typename Inst>
[[nodiscard]] bool try_execute_cvt_f32_f16_vop3_simd(Inst &, Wavefront &) {
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

/// v_cndmask_b32 VOP3 form: dst[lane] = (sel[lane]) ? src1 : src0, where `sel`
/// is the wave-mask value read from scalar `src2` (instead of the fixed VCC
/// used by the VOP2 form). Wave32-only gfx1250 uses one SGPR; wave64-capable
/// targets use a pair when running Wave64. VOP3 source modifiers are bitwise sign modifiers for
/// this B32 select: abs clears bit 31 and neg flips bit 31 before selection.
/// `src2` is an SGPR/inline operand, not a VGPR, so it does not participate in
/// the simd_capable gate; src0/src1/vdst do.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cndmask_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t sel64 = read_wave_mask_scalar(inst.src2, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_b32_src_mod(src0.template load_native<T>(base), inst.inst_.abs,
                                          inst.inst_.neg, 0);
    const auto b = apply_vop3_b32_src_mod(src1.template load_native<T>(base), inst.inst_.abs,
                                          inst.inst_.neg, 1);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    auto r = a;
    util::stdx::where(util::simd_mask_from_bits<util::native<T>>(sel_bits), r) = b;
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 cndmask path; see the binary-path note.
template <typename Inst> [[nodiscard]] bool try_execute_cndmask_vop3_simd(Inst &, Wavefront &) {
  return false;
}

/// v_cndmask_b16 VOP3 form: same per-lane select as cndmask_vop3 but the
/// dst (and each source) is the low 16 bits of the 32-bit VGPR; the high 16
/// are zeroed (matching the scalar body's `uint32_t(uint16_t(...))` pattern).
/// The select shape is identical to the b32 form — the only addition is a
/// `& 0xFFFFu` mask before the masked store. RDNA3+; CDNA4 does not decode.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cndmask_b16_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t sel64 = read_wave_mask_scalar(inst.src2, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    auto r = a;
    util::stdx::where(util::simd_mask_from_bits<util::native<T>>(sel_bits), r) = b;
    r = r & util::native<T>(0xFFFFu);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_cndmask_b16_vop3_simd(Inst &, Wavefront &) {
  return false;
}

/// VOPC compare SIMD fast path: per active EXEC lane, `cmp_op(src0, vsrc1)`
/// produces a `simd_mask` whose bit is packed into VCC at the lane position;
/// inactive-lane VCC bits are preserved (mirroring the scalar body, which
/// flips only active-lane bits). VOPC writes VCC only — there is no vdst
/// operand and CDNA4 has no v_cmpx (EXEC-writing) form, so this single shape
/// covers every compare. `T` is the 32-bit lane read type (uint32_t raw
/// encodings for the f16/f32 relations, int32_t/uint32_t for the integer ones);
/// the f16 and 16-bit integer relations read as 32-bit lanes and narrow/convert
/// inside the functor. The VCC merge is identical to the carry path's.
///
/// Float relations never see host floats: their functors call
/// comparison::evaluate on the raw encodings with the captured MODE
/// input-denormal policy, the same evaluation the scalar body uses, so host
/// DAZ cannot alter a lane and the compares are bit-exact for every input
/// including NaN/Inf/±0/subnormals (no accepted-divergence carve-out, unlike
/// fma / min-max).
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the VOPC path; see the binary-path note above.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc_simd<T>(inst, wf, op, [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// 64-bit-lane VOPC compare SIMD fast path (f64/i64/u64 relations). Identical to
/// try_execute_vopc_simd but reads each operand as `native<T>` (T = uint64_t
/// raw encodings for f64, int64_t / uint64_t for the integer relations) through
/// 64-bit RegisterAccess operand views, so it processes `native_width64` lanes
/// per chunk. f64 functors evaluate through comparison::evaluate as above.
/// Same VCC merge.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                  WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = cmp_bits64<T>(a, b, cmp_op);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the 64-bit VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc64_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc64_simd<T>(inst, wf, op,
                                    [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// Mixed-width v_cmp_class_f64 SIMD fast path. v_cmp_class_f64 tests a 64-bit f64
/// src0 against a 32-bit class mask in vsrc1, so unlike the relational VOPC64 path
/// the two operands have different widths: src0 is read as `native<uint64_t>` raw
/// bits through a 64-bit RegisterAccess operand view, and the per-lane
/// mask as a `native_width64`-wide `narrow32<uint32_t>`. The functor
/// classifies the f64 from its raw bits and tests the class against the mask,
/// returning a `native_width64`-wide mask packed into VCC exactly like the other
/// VOPC paths (active lanes only, inactive bits preserved). The classification is
/// pure bit decode, bit-exact with the scalar body for every input.
template <typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_class_f64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                          WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto mask_src = regs.read_operand(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = src0.template load_native<uint64_t>(base);
    const auto mask = mask_src.template load_narrow<uint32_t>(base);
    const uint64_t cmp_bits = cmp_class_f64_bits(s, mask, cmp_op);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the f64 class path; see the binary-path note.
template <typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_class_f64_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_class_f64_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc_class_f64_simd(inst, wf, op,
                                         [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// VOP3 v_cmp_class_f16/f32 SIMD fast path (32-bit value). The VOP3 form differs
/// from the VOPC form in three ways, all handled here: (1) the raw result is
/// passed to a caller-provided architectural commit; (2) the per-instruction
/// `abs`/`neg` source modifiers are applied
/// to src0's selected raw bits before classification; `signmask` is passed per op
/// (0x8000 for f16, 0x80000000 for f32, since both share a uint32 lane); (3) the
/// class mask is read from `inst.src1`, not `inst.vsrc1`. In true16 mode, f16
/// VOP3 inputs use op_sel to select each source half before the classify functor
/// sees the value and mask.
template <bool True16, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3_class_b32_simd(Inst &inst, Wavefront &wf,
                                                          uint32_t signmask, CmpOp cmp_op,
                                                          WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool do_abs = (inst.inst_.abs & (1u << 0)) != 0;
  const bool do_neg = (inst.inst_.neg & (1u << 0)) != 0;
  const bool true16 = True16 && signmask == 0x8000u;
  const uint32_t opsel = true16 ? vop3_opsel(inst.inst_) : 0u;
  const auto sm = util::broadcast<T>(signmask);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = src0.template load_native<T>(base);
    auto b = src1.template load_native<T>(base);
    if (true16) {
      a = select_vop3_true16_src(a, opsel, 0);
      b = select_vop3_true16_src(b, opsel, 1);
    }
    if (do_abs)
      a = a & ~sm;
    if (do_neg)
      a = a ^ sm;
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the VOP3 b32 class path; see the binary-path note.
template <bool True16, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vop3_class_b32_simd(Inst &, Wavefront &, uint32_t, CmpOp,
                                                   WriteResult) {
  return false;
}

/// VOP3 v_cmp_class_f64 SIMD fast path. The 64-bit-value counterpart of
/// try_execute_vop3_class_b32_simd: src0 is read as `native<uint64_t>` raw bits
/// through a 64-bit RegisterAccess operand view and the class mask as a
/// `narrow32<uint32_t>` from `inst.src1`. The packed result is passed to the
/// generated wrapper's result writer, which owns the final SGPR/EXEC merge.
/// abs/neg are applied to the 64-bit raw bits (signmask 0x8000000000000000),
/// using the same classify functor as the VOPC f64 class path.
template <typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3_class_f64_simd(Inst &inst, Wavefront &wf,
                                                          uint64_t signmask, CmpOp cmp_op,
                                                          WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool do_abs = (inst.inst_.abs & (1u << 0)) != 0;
  const bool do_neg = (inst.inst_.neg & (1u << 0)) != 0;
  const auto sm = util::broadcast64<uint64_t>(signmask);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto mask_src = regs.read_operand(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto s = src0.template load_native<uint64_t>(base);
    if (do_abs)
      s = s & ~sm;
    if (do_neg)
      s = s ^ sm;
    const auto mask = mask_src.template load_narrow<uint32_t>(base);
    const uint64_t cmp_bits = cmp_class_f64_bits(s, mask, cmp_op);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the VOP3 f64 class path; see the binary-path note.
template <typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vop3_class_f64_simd(Inst &, Wavefront &, uint64_t, CmpOp,
                                                   WriteResult) {
  return false;
}

/// VOP3 integer/bitwise binary SIMD fast path. Same shape as
/// try_execute_binary_vop2_simd but reads the VOP3 operands `src0`/`src1`
/// (instead of `src0`/`vsrc1`). The generated integer/bitwise VOP3 bodies apply
/// no source/result modifiers other than integer CLAMP. Saturating integer
/// arithmetic falls back to the scalar body; the plain op is bit-identical to
/// that body when CLAMP is clear. T is a 32-bit integer lane type.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (inst.inst_.clamp != 0u || simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
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

/// Unconstrained fallback for the VOP3 integer binary path; see the VOP2
/// binary-path note above.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 binary operations whose operands are encoded as true16 sources but
/// whose destination is a full b32 pack result.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_true16_src_simd(Inst &inst, Wavefront &wf,
                                                                  BinOp bin_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto r = bin_op(a, b);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_true16_src_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f16 binary fast path. The generic form matches the ordinary scalar
/// body's low-half read plus full-dword zero-extending write. The true16 form
/// selects source halves with op_sel[0:1] and writes the destination half per
/// the ISA's op_sel[3] policy. The packed-f16 binary functors do not apply
/// abs/neg/omod/clamp, so both forms bail to scalar whenever a modifier is
/// present.
template <bool True16, typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_f16_simd(Inst &inst, Wavefront &wf,
                                                           BinOp bin_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (inst.inst_.abs != 0u || inst.inst_.neg != 0u || inst.inst_.omod != 0u ||
      inst.inst_.clamp != 0u)
    return false;
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  if constexpr (True16) {
    auto dst = regs.readwrite_operand(inst.vdst, exec);
    if (!dst.has_storage())
      return false;
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto a = src0.template load_native<T>(base);
      auto b = src1.template load_native<T>(base);
      a = select_vop3_true16_src(a, opsel, 0);
      b = select_vop3_true16_src(b, opsel, 1);
      const auto r = bin_op(a, b);
      const auto out_half = r & util::broadcast<T>(0xffffu);
      auto prev = dst.template load_native<T>(base);
      auto out = (opsel & 0x8u) ? ((prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16))
                                : ((prev & util::broadcast<T>(0xffff0000u)) | out_half);
      if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
        out = out_half;
      dst.template store_native<T>(base, out, chunk);
    }
  } else {
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
  }
  return true;
}

template <bool True16, typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_f16_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f32 binary SIMD fast path. Reads `src0`/`src1`, applies the per-source
/// abs/neg modifiers, runs `bin_op`, then applies the result omod/clamp — the
/// exact order of the generated scalar body (abs->neg per source, op,
/// omod->clamp on the result). The modifier helpers are bit-exact, so unlike the
/// VOP2 path this fast path stays correct even when modifiers are set; no bail.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_fp_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(src1.template load_native<T>(base), abs, neg);
    const auto r =
        apply_vop3_dst_mod_f32(bin_op(a, b), omod, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f32 binary path.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_fp_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 raw-lane VOPC compare SIMD fast path (32-bit lane). The VOP3 form
/// of v_cmp_<rel>_<i16|u16|i32|u32|f16|f32> reads src0/src1 (not src0/vsrc1) and
/// writes the per-lane compare result into an arbitrary wave-mask scalar
/// destination through a caller-provided architectural commit. The functor
/// receives the raw source encodings: integer relations use no modifiers, and
/// float relations apply their captured abs/neg fields in shared/comparison.h.
/// The raw result contains active EXEC lanes only. Returns true when the SIMD
/// path executed.
template <typename T, bool True16 = false, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_vop3_int_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                         WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t dst = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = src0.template load_native<T>(base);
    auto b = src1.template load_native<T>(base);
    if constexpr (True16) {
      a = select_vop3_true16_src(a, vop3_opsel(inst.inst_), 0);
      b = select_vop3_true16_src(b, vop3_opsel(inst.inst_), 1);
    }
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(dst);
  return true;
}

/// Unconstrained fallback for the VOP3 integer VOPC path; see the binary-path note.
template <typename T, bool True16 = false, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_vop3_int_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

/// 64-bit-lane VOP3 raw-lane VOPC compare SIMD fast path (i64/u64/f64).
/// Identical to try_execute_vopc_vop3_int_simd but reads each operand as
/// `native<T>` (T = int64_t / uint64_t) through 64-bit RegisterAccess operand
/// views, so it processes `native_width64` lanes per chunk. The caller performs
/// the result commit; any float modifiers are applied by the functor.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_vop3_int_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                           WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t dst = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = cmp_bits64<T>(a, b, cmp_op);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(dst);
  return true;
}

/// Unconstrained fallback for the 64-bit VOP3 integer VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc64_vop3_int_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

/// VOP3 f64 binary SIMD fast path. 64-bit-lane counterpart of
/// try_execute_binary_vop3_fp_simd: reads src0/src1 as `native<double>` through
/// 64-bit RegisterAccess operand views, applies the per-source abs/neg
/// VOP3 modifiers in the f64 domain (apply_vop3_src_mod_f64), runs `bin_op`,
/// applies the result omod/clamp (apply_vop3_dst_mod_f64), and masked-stores
/// through a 64-bit RegisterAccess write view. All modifier helpers are
/// bit-exact, so the fast path stays correct even with modifiers set; no bail.
template <typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                            BinOp bin_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f64(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(src1.template load_native<T>(base), abs, neg);
    const auto r =
        apply_vop3_dst_mod_f64(bin_op(a, b), omod, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f64 binary path; see the binary-path note.
template <typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_fp64_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f64 unary SIMD fast path. 64-bit-lane counterpart of
/// try_execute_unary_vop3_fp_simd: reads `src0` as `native<double>`, applies
/// the src0 abs/neg modifiers (apply_vop3_src_mod_f64), runs `un_op`, applies
/// the result omod/clamp (apply_vop3_dst_mod_f64). All bit-exact.
template <typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp64_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f64(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(un_op(a), omod, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f64 unary path; see the binary-path note.
template <typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp64_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// VOP3 f16 unary SIMD fast path. Mirrors the scalar body's
/// f16_to_f32 -> abs/neg -> op -> omod/clamp -> f32_to_f16_mode chain. The generic
/// form reads the low source half and zero-extends the full destination dword;
/// the true16 form selects the source half and writes the selected destination
/// half per the ISA's op_sel[3] policy.
/// With rounded_result, the operation supplies an architectural half and OMOD
/// acts on that half before CLAMP. Other operations retain promoted arithmetic.
/// All steps bit-exact per the f16 VOP3 cmp slice's widening probe (f16_to_f32
/// + f32_to_f16_mode verified against the scalar helper incl. NaN payload).
template <bool True16, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp16_simd(Inst &inst, Wavefront &wf, UnOp un_op,
                                                           bool rounded_result = false) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f16(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  const auto modify_result = [&](util::native<float> value) {
    if (!rounded_result)
      return apply_vop3_dst_mod_f32(value, omod, clamp, floating_clamp_nan_to_zero(wf));
    return util::map_native_convert_scalar<float, float>(value, [&](float lane) {
      lane = fp_mode::apply_omod_f16(lane, omod, wf.fp16_ovfl());
      return clamp ? clamp_floating_result(lane, wf) : lane;
    });
  };
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  if constexpr (True16) {
    auto dst = regs.readwrite_operand(inst.vdst, exec);
    if (!dst.has_storage())
      return false;
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto raw = src0.template load_native<T>(base);
      raw = select_vop3_true16_src(raw, opsel, 0);
      const auto in = util::f16_to_f32_simd(raw);
      const auto a = apply_vop3_src_mod_f32<0>(in, abs, neg);
      const auto r = modify_result(un_op(a));
      const auto out_half =
          finalize_omod_f16_bits_simd(util::f32_to_f16_mode_simd(r, wf.fp16_ovfl()), omod);
      auto prev = dst.template load_native<T>(base);
      auto out = (opsel & 0x8u) ? ((prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16))
                                : ((prev & util::broadcast<T>(0xffff0000u)) | out_half);
      if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
        out = out_half;
      dst.template store_native<T>(base, out, chunk);
    }
  } else {
    auto dst = regs.write_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto raw = src0.template load_native<T>(base) & util::broadcast<T>(0xffffu);
      const auto in = util::f16_to_f32_simd(raw);
      const auto a = apply_vop3_src_mod_f32<0>(in, abs, neg);
      const auto r = modify_result(un_op(a));
      const auto out =
          finalize_omod_f16_bits_simd(util::f32_to_f16_mode_simd(r, wf.fp16_ovfl()), omod) &
          util::broadcast<T>(0xffffu);
      dst.template store_native<T>(base, out, chunk);
    }
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f16 unary path; see the binary-path note.
template <bool True16, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp16_simd(Inst &, Wavefront &, UnOp, bool = false) {
  return false;
}

/// VOP3 integer/bitwise ternary SIMD fast path. Reads `src0`/`src1`/`src2`,
/// runs `tern_op(a, b, c)`, and masked-stores the result. The generated scalar
/// bodies for these ternary integer ops apply no source/result modifiers except
/// integer CLAMP. Saturating arithmetic falls back to the scalar body; the plain
/// functor is bit-identical to that body when CLAMP is clear. T is a 32-bit
/// integer lane type (typically uint32_t). Same SIMD-capable / EXEC-chunk loop
/// as the binary VOP3 path.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_simd(Inst &inst, Wavefront &wf, TernOp tern_op) {
  if (inst.inst_.clamp != 0u || simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.src1.simd_capable() || !inst.src2.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto c = src2.template load_native<T>(base);
    const auto r = tern_op(a, b, c);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer ternary path; see binary-path note.
template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 ternary operations with true16 SRC0/SRC1 and a full-width SRC2/result,
/// such as V_MAD_[IU]32_[IU]16.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_true16_src01_simd(Inst &inst, Wavefront &wf,
                                                                     TernOp tern_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (inst.inst_.clamp != 0u || simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.src1.simd_capable() || !inst.src2.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto c = src2.template load_native<T>(base);
    const auto r = tern_op(a, b, c);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_true16_src01_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 ternary operations whose three sources and destination are true16,
/// such as min3/max3/med3 i16/u16 on RDNA true16 encodings.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_true16_simd(Inst &inst, Wavefront &wf,
                                                               TernOp tern_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.readwrite_operand(inst.vdst, exec);
  if (!dst.has_storage())
    return false;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto c = select_vop3_true16_src(src2.template load_native<T>(base), opsel, 2);
    const auto r = tern_op(a, b, c);
    const auto out_half = r & util::broadcast<T>(0xffffu);
    auto prev = dst.template load_native<T>(base);
    auto out = (opsel & 0x8u) ? ((prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16))
                              : ((prev & util::broadcast<T>(0xffff0000u)) | out_half);
    if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
      out = out_half;
    dst.template store_native<T>(base, out, chunk);
  }
  return true;
}

template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_true16_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 f32 ternary SIMD fast path (FMA / MAD family). Reads src0/src1/src2 as
/// `native<float>`, applies the per-source abs/neg VOP3 modifiers, runs
/// `tern_op(a, b, c)`, applies CLAMP and optionally OMOD. FIXUP handles OMOD
/// inside its operation using guest rounding. FMA uses the shared hardware
/// NaN selection policy in both scalar and SIMD execution.
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp_simd(Inst &inst, Wavefront &wf, FmaOp tern_op,
                                                           bool apply_omod = true,
                                                           bool force_output_flush = false) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod =
      apply_omod ? effective_vop3_omod_f32(wf, inst.inst_.omod, force_output_flush) : 0;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(src1.template load_native<T>(base), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(src2.template load_native<T>(base), abs, neg);
    const auto r =
        apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp_simd(Inst &, Wavefront &, FmaOp, bool = true,
                                                    bool = false) {
  return false;
}

/// VOP3 f16 ternary SIMD fast path. Mirrors the scalar f16 chain across three
/// sources: widen each via util::f16_to_f32_simd, apply f32 abs/neg, run
/// `tern_op` on native<float>, apply omod/clamp, narrow via f32_to_f16_mode_simd.
/// The generic form zero-extends the full destination dword; the true16 form
/// selects all source halves and writes the selected destination half per the
/// ISA's op_sel[3] policy.
template <bool True16, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp16_simd(Inst &inst, Wavefront &wf,
                                                             FmaOp tern_op,
                                                             bool preserve_nan_payload = false) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f16(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const auto narrow = [&](util::native<float> value) {
    if (!preserve_nan_payload)
      return util::f32_to_f16_mode_simd(value, wf.fp16_ovfl());
    return util::native<uint32_t>([&](auto index) {
      return narrow_div_fixup_f16(static_cast<float>(value[index]), wf.fp16_ovfl());
    });
  };
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  if constexpr (True16) {
    auto dst = regs.readwrite_operand(inst.vdst, exec);
    if (!dst.has_storage())
      return false;
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto a_raw = src0.template load_native<T>(base);
      auto b_raw = src1.template load_native<T>(base);
      auto c_raw = src2.template load_native<T>(base);
      a_raw = select_vop3_true16_src(a_raw, opsel, 0);
      b_raw = select_vop3_true16_src(b_raw, opsel, 1);
      c_raw = select_vop3_true16_src(c_raw, opsel, 2);
      const auto a = apply_vop3_src_mod_f32<0>(util::f16_to_f32_simd(a_raw), abs, neg);
      const auto b = apply_vop3_src_mod_f32<1>(util::f16_to_f32_simd(b_raw), abs, neg);
      const auto c = apply_vop3_src_mod_f32<2>(util::f16_to_f32_simd(c_raw), abs, neg);
      const auto r =
          apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
      const auto out_half = finalize_omod_f16_bits_simd(narrow(r), omod);
      auto prev = dst.template load_native<T>(base);
      auto out = (opsel & 0x8u) ? ((prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16))
                                : ((prev & util::broadcast<T>(0xffff0000u)) | out_half);
      if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
        out = out_half;
      dst.template store_native<T>(base, out, chunk);
    }
  } else {
    auto dst = regs.write_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto a_raw = src0.template load_native<T>(base) & util::broadcast<T>(0xffffu);
      auto b_raw = src1.template load_native<T>(base) & util::broadcast<T>(0xffffu);
      auto c_raw = src2.template load_native<T>(base) & util::broadcast<T>(0xffffu);
      const auto a = apply_vop3_src_mod_f32<0>(util::f16_to_f32_simd(a_raw), abs, neg);
      const auto b = apply_vop3_src_mod_f32<1>(util::f16_to_f32_simd(b_raw), abs, neg);
      const auto c = apply_vop3_src_mod_f32<2>(util::f16_to_f32_simd(c_raw), abs, neg);
      const auto r =
          apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
      const auto out = finalize_omod_f16_bits_simd(narrow(r), omod) & util::broadcast<T>(0xffffu);
      dst.template store_native<T>(base, out, chunk);
    }
  }
  return true;
}

template <bool True16, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp16_simd(Inst &, Wavefront &, FmaOp, bool = false) {
  return false;
}

/// @brief MODE-aware VOP3 F16 FMA SIMD fast path.
template <bool True16, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fma_vop3_fp16_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t omod = fp_mode::effective_f16_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(),
                                                    wf.ieee_mode(), false, inst.inst_.omod);
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  if constexpr (True16) {
    auto dst = regs.readwrite_operand(inst.vdst, exec);
    if (!dst.has_storage())
      return false;
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
      auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
      auto c = select_vop3_true16_src(src2.template load_native<T>(base), opsel, 2);
      const auto out_half = fma_f16_mode_simd(
          a, b, c, inst.inst_.abs & 1u, inst.inst_.abs & 2u, inst.inst_.abs & 4u,
          inst.inst_.neg & 1u, inst.inst_.neg & 2u, inst.inst_.neg & 4u, wf.fp_round_mode_f16_f64(),
          wf.fp_denorm_mode_f16_f64(), omod, inst.inst_.clamp, wf.fp16_ovfl(),
          floating_clamp_nan_to_zero(wf), fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()));
      auto prev = dst.template load_native<T>(base);
      auto out = (opsel & 0x8u) ? ((prev & 0x0000ffffu) | (out_half << 16))
                                : ((prev & 0xffff0000u) | out_half);
      if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
        out = out_half;
      dst.template store_native<T>(base, out, chunk);
    }
  } else {
    auto dst = regs.write_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto out = fma_f16_mode_simd(
          src0.template load_native<T>(base), src1.template load_native<T>(base),
          src2.template load_native<T>(base), inst.inst_.abs & 1u, inst.inst_.abs & 2u,
          inst.inst_.abs & 4u, inst.inst_.neg & 1u, inst.inst_.neg & 2u, inst.inst_.neg & 4u,
          wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), omod, inst.inst_.clamp,
          wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf),
          fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()));
      dst.template store_native<T>(base, out, chunk);
    }
  }
  return true;
}

template <bool True16, typename Inst>
[[nodiscard]] bool try_execute_fma_vop3_fp16_simd(Inst &, Wavefront &) {
  return false;
}

/// @brief MODE-aware VOP3 F64 FMA SIMD fast path.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fma_vop3_fp64_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t omod = fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(),
                                                wf.ieee_mode(), inst.inst_.omod);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto src2 = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<double>(base),
                                             inst.inst_.abs, inst.inst_.neg);
    const auto b = apply_vop3_src_mod_f64<1>(src1.template load_native<double>(base),
                                             inst.inst_.abs, inst.inst_.neg);
    const auto c = apply_vop3_src_mod_f64<2>(src2.template load_native<double>(base),
                                             inst.inst_.abs, inst.inst_.neg);
    auto result =
        fma_f64_mode_simd(a, b, c, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64());
    result = finish_f64_mode_simd(result, wf.fp_round_mode_f16_f64(), omod, inst.inst_.clamp,
                                  floating_clamp_nan_to_zero(wf));
    dst.template store_native<double>(base, result, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_fma_vop3_fp64_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 f64 ternary SIMD fast path. 64-bit-lane counterpart: read src0/src1/src2
/// through 64-bit RegisterAccess operand views, apply abs/neg in f64, run
/// `tern_op`, apply CLAMP and optionally OMOD, and store through a 64-bit RegisterAccess write
/// view.
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                             FmaOp tern_op,
                                                             bool apply_omod = true) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = apply_omod ? effective_vop3_omod_f64(wf, inst.inst_.omod) : 0;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto src2 = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(src1.template load_native<T>(base), abs, neg);
    const auto c = apply_vop3_src_mod_f64<2>(src2.template load_native<T>(base), abs, neg);
    const auto r =
        apply_vop3_dst_mod_f64(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp64_simd(Inst &, Wavefront &, FmaOp, bool = true) {
  return false;
}

/// VOP3 dst-accumulate FMA fast path (f32). Counterpart of
/// try_execute_ternary_vop3_fp_simd for the v_fmac / v_mac family, where the
/// third FMA operand IS the destination register (no separate src2 Operand
/// in the per-isa codegen class). The scalar body reads vdst as the
/// accumulator without applying abs/neg to it (per scalar; verified for
/// v_fmac_f32_vop3 + v_mac_f32_vop3). SIMD mirrors: read inst.vdst as the
/// third operand, apply abs/neg to src0/src1 only, run `tern_op`, apply
/// result omod/clamp, masked-store back to inst.vdst (overwriting accumulator).
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp_simd(Inst &inst, Wavefront &wf, FmaOp tern_op,
                                                        bool force_output_flush = false) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod, force_output_flush);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(src1.template load_native<T>(base), abs, neg);
    const auto c = acc.template load_native<T>(base); // accumulator, NO modifier
    const auto r =
        apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
    acc.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_fmac_vop3_fp_simd(Inst &, Wavefront &, FmaOp, bool = false) {
  return false;
}

/// VOP3 dst-accumulate FMA fast path (f16). f16 widen chain across src0/src1
/// + vdst (accumulator). NO abs/neg on accumulator (per scalar). The generic
/// form treats vdst as a low-half f16 value and zero-extends the full dword;
/// the true16 form selects src0/src1 and the accumulator/destination half with
/// OPSEL.
template <bool True16, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp16_simd(Inst &inst, Wavefront &wf,
                                                          FmaOp tern_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f16(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  if constexpr (True16)
    if (!acc.has_storage())
      return false;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a_raw = src0.template load_native<T>(base);
    auto b_raw = src1.template load_native<T>(base);
    auto c_raw = acc.template load_native<T>(base);
    auto prev = c_raw;
    if constexpr (True16) {
      a_raw = select_vop3_true16_src(a_raw, opsel, 0);
      b_raw = select_vop3_true16_src(b_raw, opsel, 1);
      c_raw = (opsel & 0x8u) ? (c_raw >> 16) : c_raw;
      c_raw = c_raw & util::broadcast<T>(0xffffu);
    } else {
      a_raw = a_raw & util::broadcast<T>(0xffffu);
      b_raw = b_raw & util::broadcast<T>(0xffffu);
      c_raw = c_raw & util::broadcast<T>(0xffffu);
    }
    const auto a = apply_vop3_src_mod_f32<0>(util::f16_to_f32_simd(a_raw), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(util::f16_to_f32_simd(b_raw), abs, neg);
    const auto c = util::f16_to_f32_simd(c_raw); // accumulator, no modifier
    const auto r =
        apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
    const auto out_half =
        finalize_omod_f16_bits_simd(util::f32_to_f16_mode_simd(r, wf.fp16_ovfl()), omod) &
        util::broadcast<T>(0xffffu);
    auto out = out_half;
    if constexpr (True16) {
      if (opsel & 0x8u)
        out = (prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16);
      else if (cdna_vop3_low_dst_zeroes_high(wf))
        out = out_half;
      else
        out = (prev & util::broadcast<T>(0xffff0000u)) | out_half;
    }
    acc.template store_native<T>(base, out, chunk);
  }
  return true;
}

template <bool True16, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_fmac_vop3_fp16_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// @brief MODE-aware VOP3 F16 destination-accumulate FMA SIMD fast path.
template <bool True16, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp16_mode_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t opsel = vop3_opsel(inst.inst_);
  const uint32_t omod = fp_mode::effective_f16_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(),
                                                    wf.ieee_mode(), false, inst.inst_.omod);
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto acc = regs.readwrite_operand(inst.vdst, exec);
  if constexpr (True16)
    if (!acc.has_storage())
      return false;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = src0.template load_native<T>(base);
    auto b = src1.template load_native<T>(base);
    auto c = acc.template load_native<T>(base);
    const auto prev = c;
    if constexpr (True16) {
      a = select_vop3_true16_src(a, opsel, 0);
      b = select_vop3_true16_src(b, opsel, 1);
      c = (opsel & 0x8u) ? (c >> 16) : c;
    }
    const auto out_half = fma_f16_mode_simd(
        a, b, c, inst.inst_.abs & 1u, inst.inst_.abs & 2u, false, inst.inst_.neg & 1u,
        inst.inst_.neg & 2u, false, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64(), omod,
        inst.inst_.clamp, wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf),
        fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()));
    auto out = out_half;
    if constexpr (True16) {
      if (opsel & 0x8u)
        out = (prev & 0x0000ffffu) | (out_half << 16);
      else if (!cdna_vop3_low_dst_zeroes_high(wf))
        out = (prev & 0xffff0000u) | out_half;
    }
    acc.template store_native<T>(base, out, chunk);
  }
  return true;
}

template <bool True16, typename Inst>
[[nodiscard]] bool try_execute_fmac_vop3_fp16_mode_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 dst-accumulate FMA fast path (f64).
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                          FmaOp tern_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f64(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto acc = regs.readwrite_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(src1.template load_native<T>(base), abs, neg);
    const auto c = acc.template load_native<T>(base); // accumulator, no mod
    const auto r =
        apply_vop3_dst_mod_f64(tern_op(a, b, c), omod, clamp, floating_clamp_nan_to_zero(wf));
    acc.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_fmac_vop3_fp64_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// @brief MODE-aware VOP3 F64 destination-accumulate FMA SIMD fast path.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp64_mode_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t omod = fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(),
                                                wf.ieee_mode(), inst.inst_.omod);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto acc = regs.readwrite_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<double>(base),
                                             inst.inst_.abs, inst.inst_.neg);
    const auto b = apply_vop3_src_mod_f64<1>(src1.template load_native<double>(base),
                                             inst.inst_.abs, inst.inst_.neg);
    const auto c = acc.template load_native<double>(base);
    auto result =
        fma_f64_mode_simd(a, b, c, wf.fp_round_mode_f16_f64(), wf.fp_denorm_mode_f16_f64());
    result = finish_f64_mode_simd(result, wf.fp_round_mode_f16_f64(), omod, inst.inst_.clamp,
                                  floating_clamp_nan_to_zero(wf));
    acc.template store_native<double>(base, result, chunk);
  }
  return true;
}

template <typename Inst>
[[nodiscard]] bool try_execute_fmac_vop3_fp64_mode_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 mixed-width LDEXP fast path with explicit guest FP controls.
/// Reads src0 as native<float>, applies src0 abs/neg in f32, reads src1 as
/// native<int32_t> (per-lane exponent), runs `op(a, e)`, applies
/// result omod/clamp, then stores through a RegisterAccess write view.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ldexp_vop3_fp32_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  // The final floating clamp must not inherit host DAZ/FTZ or exception state.
  fp_mode::detail::ScopedFenv environment(wf.fp_round_mode_f32());
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto exp_src = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
    const auto e = exp_src.template load_native<int32_t>(base);
    const util::native<float> values = op(a, e);
    const util::native<float> scaled([&](auto index) {
      return div_apply_omod(static_cast<float>(values[index]), wf.fp_round_mode_f32(), omod);
    });
    const auto r = apply_vop3_dst_mod_f32(scaled, 0, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_ldexp_vop3_fp32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3 mixed-width F64 LDEXP fast path with explicit guest FP controls.
/// Reads src0 as native<double> via a 64-bit RegisterAccess operand view,
/// applies src0 abs/neg in f64,
/// reads src1 as narrow32<int32_t> (native_width64-wide), runs `op(a, e)`,
/// applies result omod/clamp, and stores through a 64-bit RegisterAccess write
/// view.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ldexp_vop3_fp64_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  // Keep the final clamp in the same guest environment as the scalar path.
  fp_mode::detail::ScopedFenv environment(wf.fp_round_mode_f16_f64());
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f64(wf, inst.inst_.omod);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto exp_src = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto e = exp_src.template load_narrow<int32_t>(base);
    const util::native<double> values = op(a, e);
    const util::native<double> scaled([&](auto index) {
      return div_apply_omod(static_cast<double>(values[index]), wf.fp_round_mode_f16_f64(), omod);
    });
    const auto r = apply_vop3_dst_mod_f64(scaled, 0, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_ldexp_vop3_fp64_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3 f32 unary SIMD fast path. Reads `src0` as f32, applies the src0 abs/neg
/// modifiers, runs `un_op` (`native<float> -> native<float>`), then applies the
/// result omod/clamp — the scalar body's order (abs->neg, op, omod->clamp). Tin
/// and Tout are both float32_t (the plain int/cvt unary VOP3 forms apply no
/// modifiers and reuse the VOP1 unary path directly). The modifier helpers are
/// bit-exact, so the fast path stays correct with modifiers set; no bail.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp_simd(Inst &inst, Wavefront &wf, UnOp un_op,
                                                         bool force_output_flush = false) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = effective_vop3_omod_f32(wf, inst.inst_.omod, force_output_flush);
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<Tout>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<Tin>(base), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(un_op(a), omod, clamp, floating_clamp_nan_to_zero(wf));
    dst.template store_native<Tout>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f32 unary path.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp_simd(Inst &, Wavefront &, UnOp, bool = false) {
  return false;
}

/// @brief VOP3 FMAS keeps batched register access and rounds each scaled FMA once.
/// @details Integer significands provide the same guest rounding and denormal
/// behavior as the scalar executor, independent of host floating-point controls.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_div_fmas_f32_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t vcc = wf.vcc_mask(exec);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(src1.template load_native<T>(base), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(src2.template load_native<T>(base), abs, neg);
    util::native<T> r(T{0});
    for (std::size_t i = 0; i < W; ++i) {
      if ((chunk >> i) & 1u)
        r[i] = div_fmas(static_cast<T>(a[i]), static_cast<T>(b[i]), static_cast<T>(c[i]),
                        ((vcc >> (base + i)) & 1u) != 0, wf.fp_round_mode_f32(),
                        wf.fp_denorm_mode_f32());
    }
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_div_fmas_f32_simd(Inst &, Wavefront &) {
  return false;
}

/// @brief F64 counterpart with exact 106-bit products and fused exponent scaling.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_div_fmas_f64_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t vcc = wf.vcc_mask(exec);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto src2 = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(src0.template load_native<T>(base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(src1.template load_native<T>(base), abs, neg);
    const auto c = apply_vop3_src_mod_f64<2>(src2.template load_native<T>(base), abs, neg);
    util::native<T> r(T{0});
    for (std::size_t i = 0; i < W; ++i) {
      if ((chunk >> i) & 1u)
        r[i] = div_fmas(static_cast<T>(a[i]), static_cast<T>(b[i]), static_cast<T>(c[i]),
                        ((vcc >> (base + i)) & 1u) != 0, wf.fp_round_mode_f16_f64(),
                        wf.fp_denorm_mode_f16_f64());
    }
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_div_fmas_f64_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 64-bit-lane reverse-shift fast path (v_lshlrev_b64 / v_lshrrev_b64 /
/// v_ashrrev_i64). The shift amount is a 32-bit src0 (read as a native_width64
/// narrow lane, widened to 64-bit and masked to [0,63]); the shifted value is
/// the 64-bit src1; the result is 64-bit. `shift_op(value, shift)` receives both
/// as `native<uint64_t>` (shift already widened+masked), so the functor is a
/// plain `v << sh` / `v >> sh` (logical) or an arithmetic-shift cast for the
/// signed form — bit-identical to the scalar body's `& 63u` shift.
template <typename Inst, typename ShiftOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_shift64_vop3_simd(Inst &inst, Wavefront &wf,
                                                        ShiftOp shift_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  // src0 = 32-bit shift amount (narrow lane), src1 = 64-bit value, dst = 64-bit.
  RegisterAccess regs(wf);
  auto shift_src = regs.read_operand(inst.src0, exec);
  auto value_src = regs.read_operand64(inst.src1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = shift_src.template load_narrow<uint32_t>(base);
    const auto v = value_src.template load_native<uint64_t>(base);
    const auto sh = util::stdx::static_simd_cast<util::native<uint64_t>>(s) & 63ull;
    dst.template store_native<uint64_t>(base, shift_op(v, sh), chunk);
  }
  return true;
}

template <typename Inst, typename ShiftOp>
[[nodiscard]] bool try_execute_shift64_vop3_simd(Inst &, Wavefront &, ShiftOp) {
  return false;
}

/// VOP3 v_lshl_add_u64 fast path: dst = (src0 << (src1 & 63)) + src2, all 64-bit
/// except the 32-bit shift amount src1. The scalar body shifts by the raw src1
/// (C++ UB at >=64, but x86 masks the count to 6 bits); masking to 63 here
/// reproduces that x86 scalar result for every shift value.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_lshl_add_u64_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  // src0 = 64-bit value, src1 = 32-bit shift (narrow), src2 = 64-bit addend.
  RegisterAccess regs(wf);
  auto value_src = regs.read_operand64(inst.src0, exec);
  auto shift_src = regs.read_operand(inst.src1, exec);
  auto addend_src = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto v = value_src.template load_native<uint64_t>(base);
    const auto s = shift_src.template load_narrow<uint32_t>(base);
    const auto c = addend_src.template load_native<uint64_t>(base);
    const auto sh = util::stdx::static_simd_cast<util::native<uint64_t>>(s) & 63ull;
    dst.template store_native<uint64_t>(base, simd_lshl_u64(v, sh) + c, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_lshl_add_u64_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 wide 32x32->64 multiply-add fast path (v_mad_u64_u32 / v_mad_i64_i32).
/// src0/src1 are 32-bit multiplicands (read as narrow lanes), src2 is the 64-bit
/// addend, dst is 64-bit, and sdst is the per-lane overflow/carryout mask.
/// `mad_op(s0, s1, c)` receives the two narrow operands and the 64-bit addend
/// and returns both the low 64-bit result and carry/overflow mask.
template <typename Inst, typename MadOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_mad_wide64_vop3_result_simd(Inst &inst, Wavefront &wf,
                                                                  MadOp mad_op,
                                                                  WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if constexpr (requires { inst.inst_.clamp; }) {
    if (inst.inst_.clamp)
      return false;
  }
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t carry_out = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_narrow<uint32_t>(base);
    const auto b = src1.template load_narrow<uint32_t>(base);
    const auto c = src2.template load_native<uint64_t>(base);
    const auto r = mad_op(a, b, c);
    dst.template store_native<uint64_t>(base, r.value, chunk);
    const uint64_t carry_bits = util::simd_mask_to_bits(r.carry);
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  write_result(carry_out);
  return true;
}

template <typename Inst, typename MadOp, typename WriteResult>
[[nodiscard]] bool try_execute_mad_wide64_vop3_result_simd(Inst &, Wavefront &, MadOp,
                                                           WriteResult) {
  return false;
}

template <typename Inst, typename MadOp>
[[nodiscard]] inline bool try_execute_mad_wide64_vop3_simd(Inst &inst, Wavefront &wf,
                                                           MadOp mad_op) {
  return try_execute_mad_wide64_vop3_result_simd(
      inst, wf, mad_op, [&](uint64_t result) { write_wave_mask_scalar(inst.sdst, wf, result); });
}

/// VOP3 carry-OUT binary fast path (v_add_co/sub_co/subrev_co_u32). No carry-in
/// (the SimdCarry functor's third arg is a zero vector); the per-lane carry-out
/// is written to the SGPR-pair `sdst` (not the fixed VCC). In wave32, explicit
/// scalar mask destinations write only the low SGPR and preserve the high SGPR;
/// wave64 writes the full pair.
/// `sdst` is an SGPR operand, so it does not participate in the simd_capable
/// gate; src0/src1/vdst do. (These VOP3 forms carry no `src2` member, so the
/// carry-in path lives in the separate _cin glue below.)
template <typename Inst, typename CarryOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_co_result_simd(Inst &inst, Wavefront &wf,
                                                                 CarryOp carry_op,
                                                                 WriteResult write_result) {
  if (simd_force_scalar() || inst.inst_.clamp || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t carry_out = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  const auto zero_cin = util::broadcast<T>(0u);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto r = carry_op(a, b, zero_cin);
    dst.template store_native<T>(base, r.value, chunk);
    const uint64_t carry_bits = util::simd_mask_to_bits(r.carry);
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  write_result(carry_out);
  return true;
}

template <typename Inst, typename CarryOp, typename WriteResult>
[[nodiscard]] bool try_execute_binary_vop3_co_result_simd(Inst &, Wavefront &, CarryOp,
                                                          WriteResult) {
  return false;
}

template <typename Inst, typename CarryOp>
[[nodiscard]] inline bool try_execute_binary_vop3_co_simd(Inst &inst, Wavefront &wf,
                                                          CarryOp carry_op) {
  return try_execute_binary_vop3_co_result_simd(
      inst, wf, carry_op, [&](uint64_t result) { write_wave_mask_scalar(inst.sdst, wf, result); });
}

/// VOP3 carry-IN binary fast path (v_addc_co/subb_co/subbrev_co_u32). Same as
/// the _co glue but the per-lane carry-in is read from the SGPR-pair `src2`
/// (these forms have a src2 member) and expanded to a 0/1-per-lane vector;
/// carry-out goes to `sdst` with the same wave32/wave64 width rule as _co.
/// src2/sdst are SGPR operands (not gated).
template <typename Inst, typename CarryOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_cin_result_simd(Inst &inst, Wavefront &wf,
                                                                  CarryOp carry_op,
                                                                  WriteResult write_result) {
  if (simd_force_scalar() || inst.inst_.clamp || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t cin_all = read_wave_mask_scalar(inst.src2, wf);
  uint64_t carry_out = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cin_bits = (cin_all >> base) & chunk_full;
    const auto cin = util::simd_u32_lanes_from_bits(cin_bits);
    const auto r = carry_op(a, b, cin);
    dst.template store_native<T>(base, r.value, chunk);
    const uint64_t carry_bits = util::simd_mask_to_bits(r.carry);
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  write_result(carry_out);
  return true;
}

template <typename Inst, typename CarryOp, typename WriteResult>
[[nodiscard]] bool try_execute_binary_vop3_cin_result_simd(Inst &, Wavefront &, CarryOp,
                                                           WriteResult) {
  return false;
}

template <typename Inst, typename CarryOp>
[[nodiscard]] inline bool try_execute_binary_vop3_cin_simd(Inst &inst, Wavefront &wf,
                                                           CarryOp carry_op) {
  return try_execute_binary_vop3_cin_result_simd(
      inst, wf, carry_op, [&](uint64_t result) { write_wave_mask_scalar(inst.sdst, wf, result); });
}

/// VOP3P fma_mix / mad_mix SIMD fast path. Six ops share one body because all
/// six differ only in (a) the f16-vs-f32 widening shape per source and (b) the
/// f16-lo/f16-hi/f32 narrowing shape on the destination. The generated scalar
/// bodies use `a * b + c` for legacy profiles and explicit FMA for CDNA5.
/// The Fused parameter selects the matching arithmetic policy.
///
/// Per-source data fetch is gated by `op_sel_hi` (src0/src1) and
/// `op_sel_hi_2` (src2): when the bit is 0 the source is read as f32; when
/// 1 the source is read as a 32-bit word and the low or high f16 half
/// (selected by the matching `op_sel` bit) is widened via f16_to_f32_simd.
/// Per-source abs is gated by the field named `neg_hi` in the shared VOP3P
/// layout for this mix-family encoding, and per-source sign-flip is gated by
/// `neg` (xor of bit 31). Result-clamp saturates to [0, 1] via stdx::where.
///
/// All modifier fields are uniform across the wave so the per-source mode
/// branches live outside the chunk loop and feed into the same `a*b+c`
/// inner kernel regardless of fetch shape, keeping the SIMD path branch-
/// predictable on every chunk.
template <FmaMixDst DstMode, bool Fused = false, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_fma_mix_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;

  if constexpr (simd_backend::has_fma && !Fused && DstMode == FmaMixDst::F32) {
    // Legacy scalar a*b+c and stdx SIMD expressions can contract differently
    // with hardware FMA, including GCC with UBSan. Preserve the scalar path
    // for those profiles; CDNA5 explicitly uses fused arithmetic in both
    // execution paths.
    return false;
  }
  using T = float32_t;
  const uint32_t op_sel = packed_opsel(inst.inst_);
  const uint32_t op_sel_hi = packed_opsel_hi(inst.inst_);
  const uint32_t op_sel_hi_2 = packed_opsel_hi_2(inst.inst_);
  const uint32_t abs = inst.inst_.neg_hi;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  using U = util::native<uint32_t>;
  using F = util::native<float>;
  const U kSignBit(0x80000000u);
  const U kAbsMask(~0x80000000u);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto load_src = [&](const RegisterAccess::OperandReadView &op, uint32_t base,
                      uint32_t src_selector, uint32_t sel_hi_bit, uint32_t sel_bit,
                      uint32_t abs_bit, uint32_t neg_bit) -> F {
    F v;
    if (sel_hi_bit) {
      U raw = op.template load_native<uint32_t>(base);
      U halves = is_inline_float_src(src_selector) ? util::f32_to_f16_simd(std::bit_cast<F>(raw))
                                                   : (sel_bit ? (raw >> 16) : (raw & 0xFFFFu));
      v = util::f16_to_f32_simd(halves);
    } else {
      v = op.template load_native<float>(base);
    }
    if (abs_bit)
      v = std::bit_cast<F>(std::bit_cast<U>(v) & kAbsMask);
    if (neg_bit)
      v = std::bit_cast<F>(std::bit_cast<U>(v) ^ kSignBit);
    return v;
  };
  auto compute_result = [&](uint32_t base) {
    F a = load_src(src0, base, inst.inst_.src0, op_sel_hi & 1u, op_sel & 1u, abs & 1u, neg & 1u);
    F b = load_src(src1, base, inst.inst_.src1, (op_sel_hi >> 1) & 1u, (op_sel >> 1) & 1u,
                   (abs >> 1) & 1u, (neg >> 1) & 1u);
    F c = load_src(src2, base, inst.inst_.src2, op_sel_hi_2, (op_sel >> 2) & 1u, (abs >> 2) & 1u,
                   (neg >> 2) & 1u);
    F r;
    if constexpr (Fused)
      r = fma_f32_simd(a, b, c, wf);
    else
      r = fma_mix_mul_add(a, b, c);
    if (clamp)
      r = apply_vop3_dst_mod_f32(r, 0, 1, floating_clamp_nan_to_zero(wf));
    return r;
  };
  if constexpr (DstMode == FmaMixDst::F32) {
    fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
    auto dst = regs.write_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      dst.template store_native<float>(base, compute_result(base), chunk);
    }
  } else {
    auto dst = regs.readwrite_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      U h;
      if constexpr (Fused) {
        const F a =
            load_src(src0, base, inst.inst_.src0, op_sel_hi & 1u, op_sel & 1u, abs & 1u, neg & 1u);
        const F b =
            load_src(src1, base, inst.inst_.src1, op_sel_hi & 2u, op_sel & 2u, abs & 2u, neg & 2u);
        const F c =
            load_src(src2, base, inst.inst_.src2, op_sel_hi_2, op_sel & 4u, abs & 4u, neg & 4u);
        h = mixed_fma_f16_simd<true>(a, b, c, wf.fp_round_mode_f16_f64(), clamp, wf.fp16_ovfl(),
                                     floating_clamp_nan_to_zero(wf));
      } else {
        h = mixed_fma_f16_simd<false>(F(0), F(0), compute_result(base), wf.fp_round_mode_f16_f64(),
                                      false, wf.fp16_ovfl(), false);
      }
      U prev = dst.template load_native<uint32_t>(base);
      U packed;
      if constexpr (DstMode == FmaMixDst::F16_LO) {
        packed = (prev & 0xFFFF0000u) | h;
      } else { // F16_HI
        packed = (prev & 0x0000FFFFu) | (h << 16);
      }
      dst.template store_native<uint32_t>(base, packed, chunk);
    }
  }
  return true;
}

template <FmaMixDst DstMode, bool Fused = false, typename Inst>
[[nodiscard]] bool try_execute_vop3p_fma_mix_simd(Inst &, Wavefront &) {
  return false;
}

template <PackedFloatOp Op, bool Bf16, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_packed_float_simd(Inst &inst, Wavefront &wf) {
  constexpr bool Ternary = Op == PackedFloatOp::FMA;
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if constexpr (Ternary) {
    if (!inst.src2.simd_capable())
      return false;
  }
  fp_mode::ScopedEnvironment nearest_environment(0);
  constexpr std::size_t W = util::native_width64;
  using U = util::narrow32<uint32_t>;
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  std::optional<decltype(src0)> src2;
  if constexpr (Ternary)
    src2.emplace(regs.read_operand(inst.src2, exec));
  auto dst = regs.write_operand(inst.vdst, exec);
  const uint32_t low = packed_opsel(inst.inst_);
  uint32_t high = packed_opsel_hi(inst.inst_);
  if constexpr (Ternary)
    high |= packed_opsel_hi_2(inst.inst_) << 2;
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    const uint64_t mask = (exec >> base) & util::mask<uint64_t>(W);
    if (!mask)
      continue;
    auto select = [&](U raw, auto &operand, uint32_t selector, unsigned index) {
      if constexpr (!Bf16) {
        if (pk16_src_needs_narrowing(selector, operand.size_bits()))
          raw = U([&](auto i) {
            return uint32_t(util::f32_to_f16(std::bit_cast<float>(uint32_t(raw[i]))));
          });
      }
      return select_packed_halves(raw, low, high, index);
    };
    const U a = select(src0.template load_narrow<uint32_t>(base), inst.src0, inst.inst_.src0, 0);
    const U b = select(src1.template load_narrow<uint32_t>(base), inst.src1, inst.inst_.src1, 1);
    U c(0);
    if constexpr (Ternary)
      c = select(src2->template load_narrow<uint32_t>(base), inst.src2, inst.inst_.src2, 2);
    auto half = [](U value, unsigned index, unsigned shift, uint32_t neg) {
      return ((value >> shift) & U(0xffffu)) ^ U(((neg >> index) & 1u) << 15);
    };
    U lo = packed_float_half_simd<Op, Bf16>(half(a, 0, 0, inst.inst_.neg),
                                            half(b, 1, 0, inst.inst_.neg),
                                            half(c, 2, 0, inst.inst_.neg), wf, inst.inst_.clamp);
    U hi = packed_float_half_simd<Op, Bf16>(
        half(a, 0, 16, inst.inst_.neg_hi), half(b, 1, 16, inst.inst_.neg_hi),
        half(c, 2, 16, inst.inst_.neg_hi), wf, inst.inst_.clamp);
    dst.template store_narrow<uint32_t>(base, lo | (hi << 16), mask);
  }
  return true;
}

template <PackedFloatOp Op, bool Bf16, typename Inst>
[[nodiscard]] bool try_execute_packed_float_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3P packed-16 binary integer SIMD fast path. Apply op_sel/op_sel_hi
/// to each source before the per-half functor. Integer CLAMP retains scalar
/// execution for exact saturation. The functor receives and returns packed
/// u32 lanes, including any operation-specific masking before repacking.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_int_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.clamp)
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_packed_halves(src0.template load_native<T>(base),
                                        packed_opsel(inst.inst_), packed_opsel_hi(inst.inst_), 0);
    const auto b = select_packed_halves(src1.template load_native<T>(base),
                                        packed_opsel(inst.inst_), packed_opsel_hi(inst.inst_), 1);
    const auto r = op(a, b);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_binary_int_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-16 ternary integer SIMD fast path. Apply all three source
/// half selectors before the functor; op_sel_hi_2 selects the third source's
/// high output half. Integer CLAMP retains scalar execution for exact
/// per-half saturation. Packed integer MAD does not apply neg/neg_hi.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_int_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.clamp)
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_packed_halves(src0.template load_native<T>(base),
                                        packed_opsel(inst.inst_), packed_opsel_hi(inst.inst_), 0);
    const auto b = select_packed_halves(src1.template load_native<T>(base),
                                        packed_opsel(inst.inst_), packed_opsel_hi(inst.inst_), 1);
    const auto c =
        select_packed_halves(src2.template load_native<T>(base), packed_opsel(inst.inst_),
                             packed_opsel_hi(inst.inst_) | (packed_opsel_hi_2(inst.inst_) << 2), 2);
    const auto r = op(a, b, c);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_int_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-f32 binary fast path (v_pk_add_f32 / v_pk_mul_f32). In a VGPR
/// pair {N, N+1} register N is the LO f32 of every lane, N+1 the HI f32, so each
/// half is a native-width native<float> read of one register (read_pkf32_halves)
/// and the per-half arithmetic runs at full native width. OP_SEL chooses each
/// source half independently, including broadcasts and swapped halves.
/// neg/neg_hi bits 0/1 sign-flip the respective half. MODE and CLAMP match
/// the scalar helper. CDNA5 scalar sources replicate one DWORD independently
/// of OP_SEL; earlier CDNA preserves scalar register pairs.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_f32_simd(Inst &inst, Wavefront &wf,
                                                               uint32_t op_sel, uint32_t op_sel_hi,
                                                               Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
  auto flush_input = [&wf](util::native<float> value) {
    return (wf.fp_denorm_mode_f32() & 1u) ? value : util::flush_denorm_f32_simd(value);
  };
  auto binary = [&op](util::native<float> first, util::native<float> second) {
    util::native<float> result = op(first, second);
    // Preserve the first input's quieted NaN whenever that input is NaN.
    // Host SIMD arithmetic can otherwise choose the second input's payload.
    util::stdx::where(util::stdx::isnan(first), result) = std::bit_cast<util::native<float>>(
        std::bit_cast<util::native<uint32_t>>(first) | 0x00400000u);
    return result;
  };
  auto flush_output = [&wf, &inst](util::native<float> value) {
    if (inst.inst_.clamp)
      value = apply_vop3_dst_mod_f32(value, 0, 1, floating_clamp_nan_to_zero(wf));
    return (wf.fp_denorm_mode_f32() & 2u) ? value : util::flush_denorm_f32_simd(value);
  };
  constexpr std::size_t W = util::native_width_v<float>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  RegisterAccess regs(wf);
  const auto scalar_mode = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 ? ScalarPairMode::Replicate32
                                                                      : ScalarPairMode::Preserve;
  auto src0 = regs.read_operand_pair32(inst.src0, exec, scalar_mode);
  auto src1 = regs.read_operand_pair32(inst.src1, exec, scalar_mode);
  auto dst = regs.write_operand_pair32(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const PkF32Halves a = read_pkf32_halves(src0, base);
    const PkF32Halves b = read_pkf32_halves(src1, base);
    const util::native<float> r_lo =
        binary(flush_input(pkf32_neg((op_sel & 1u) ? a.hi : a.lo, neg0_lo)),
               flush_input(pkf32_neg((op_sel & 2u) ? b.hi : b.lo, neg1_lo)));
    const util::native<float> r_hi =
        binary(flush_input(pkf32_neg((op_sel_hi & 1u) ? a.hi : a.lo, neg0_hi)),
               flush_input(pkf32_neg((op_sel_hi & 2u) ? b.hi : b.lo, neg1_hi)));
    dst.template store_native_pair<float>(base, flush_output(r_lo), flush_output(r_hi), chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_binary_f32_simd(Inst &, Wavefront &, uint32_t, uint32_t,
                                                        Op) {
  return false;
}

template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_f32_simd(Inst &inst, Wavefront &wf, Op op) {
  return try_execute_vop3p_pk_binary_f32_simd(inst, wf, inst.inst_.op_sel, inst.inst_.op_sel_hi,
                                              op);
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_binary_f32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-f32 ternary fast path (v_pk_fma_f32). 3-source FMA per half;
/// same per-register native<float> read/write as the binary form. OP_SEL
/// chooses each source half independently; op_sel_hi_2 selects src2-hi.
/// neg/neg_hi bits 0/1/2 sign-flip the respective half. MODE and CLAMP match
/// the scalar helper, including FMA NaN payload selection.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_f32_simd(Inst &inst, Wavefront &wf,
                                                                uint32_t op_sel, uint32_t op_sel_hi,
                                                                uint32_t op_sel_hi_2, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  fp_mode::ScopedEnvironment environment(wf.fp_round_mode_f32());
  auto flush_input = [&wf](util::native<float> value) {
    return (wf.fp_denorm_mode_f32() & 1u) ? value : util::flush_denorm_f32_simd(value);
  };
  auto flush_output = [&wf, &inst](util::native<float> value) {
    if (inst.inst_.clamp)
      value = apply_vop3_dst_mod_f32(value, 0, 1, floating_clamp_nan_to_zero(wf));
    return (wf.fp_denorm_mode_f32() & 2u) ? value : util::flush_denorm_f32_simd(value);
  };
  constexpr std::size_t W = util::native_width_v<float>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg2_lo = inst.inst_.neg & 4u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  const bool neg2_hi = inst.inst_.neg_hi & 4u;
  RegisterAccess regs(wf);
  const auto scalar_mode = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 ? ScalarPairMode::Replicate32
                                                                      : ScalarPairMode::Preserve;
  auto src0 = regs.read_operand_pair32(inst.src0, exec, scalar_mode);
  auto src1 = regs.read_operand_pair32(inst.src1, exec, scalar_mode);
  auto src2 = regs.read_operand_pair32(inst.src2, exec, scalar_mode);
  auto dst = regs.write_operand_pair32(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const PkF32Halves a = read_pkf32_halves(src0, base);
    const PkF32Halves b = read_pkf32_halves(src1, base);
    const PkF32Halves c = read_pkf32_halves(src2, base);
    const util::native<float> r_lo =
        op(flush_input(pkf32_neg((op_sel & 1u) ? a.hi : a.lo, neg0_lo)),
           flush_input(pkf32_neg((op_sel & 2u) ? b.hi : b.lo, neg1_lo)),
           flush_input(pkf32_neg((op_sel & 4u) ? c.hi : c.lo, neg2_lo)));
    const util::native<float> r_hi =
        op(flush_input(pkf32_neg((op_sel_hi & 1u) ? a.hi : a.lo, neg0_hi)),
           flush_input(pkf32_neg((op_sel_hi & 2u) ? b.hi : b.lo, neg1_hi)),
           flush_input(pkf32_neg(op_sel_hi_2 ? c.hi : c.lo, neg2_hi)));
    dst.template store_native_pair<float>(base, flush_output(r_lo), flush_output(r_hi), chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_f32_simd(Inst &, Wavefront &, uint32_t, uint32_t,
                                                         uint32_t, Op) {
  return false;
}

template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_f32_simd(Inst &inst, Wavefront &wf, Op op) {
  return try_execute_vop3p_pk_ternary_f32_simd(inst, wf, inst.inst_.op_sel, inst.inst_.op_sel_hi,
                                               inst.inst_.op_sel_hi_2, op);
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_f32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P v_pk_mov_b32 SIMD fast path. Each src is a 64-bit SGPR or VGPR pair.
/// SGPR pairs are broadcast across lanes by RegisterAccess' scalar fallback.
/// op_sel[0] selects the low output dword from src0, and op_sel[1] selects the
/// high output dword from src1. The fast path is limited to the assembler's
/// default op_sel_hi value and writes through a 64-bit RegisterAccess view.
/// Functorless / fixed-op.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_mov_b32_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel_hi != 3u)
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  using U64 = util::native<uint64_t>;
  const U64 kHiMask(0xFFFFFFFF00000000ULL);
  const U64 kLoMask(0x00000000FFFFFFFFULL);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U64 s0 = src0.template load_native<uint64_t>(base);
    const U64 s1 = src1.template load_native<uint64_t>(base);
    const U64 out_lo = (inst.inst_.op_sel & 1u) ? ((s0 >> 32) & kLoMask) : (s0 & kLoMask);
    const U64 out_hi = (inst.inst_.op_sel & 2u) ? (s1 & kHiMask) : ((s1 & kLoMask) << 32);
    const U64 out = out_lo | out_hi;
    dst.template store_native<uint64_t>(base, out, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_vop3p_mov_b32_simd(Inst &, Wavefront &) {
  return false;
}

template <int ElemBits, bool Signed, typename Inst>
[[nodiscard]] bool try_execute_vop3p_dot_int_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3P v_dot2_f32_{f16,bf16} SIMD fast path. Two half-precision products plus
/// an f32 accumulator collapse into a single f32 lane: result = a0*b0 + a1*b1 +
/// acc (plain `*` / `+`, left-to-right — matching the scalar, NOT a contracted
/// fma). op_sel / op_sel_hi pick the halves of src0/src1 (gated to the default
/// packing op_sel == 0 && op_sel_hi == 3); neg / neg_hi flip the src0/src1
/// product-operand signs and neg bit 2 flips the accumulator. The encoded CLAMP
/// field is ignored for floating DOT instructions. NaN-input payload divergence accepted (same
/// carve-out as the pk_fma
/// slices). @tparam Fmt selects the f16 vs bf16 widening.
template <Vop3pDotHalfFormat Fmt, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_dot_f16_simd(Inst &inst, Wavefront &wf) {
  // Route integer accumulation policies through the exact scalar instruction body.
  if (isa_properties(wf.cu().arch()).float_dot_accumulation != FloatDotAccumulation::HostF32)
    return false;
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (packed_opsel(inst.inst_) != 0u || packed_opsel_hi(inst.inst_) != 3u)
    return false;
  // Decline exactly the sources the scalar body re-narrows, so the two paths
  // cannot disagree. src2 is a genuine f32 accumulator, so an inline constant
  // there is already correct.
  if (pk16_src_needs_narrowing(inst.inst_.src0, inst.src0.size_bits()) ||
      pk16_src_needs_narrowing(inst.inst_.src1, inst.src1.size_bits()))
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  const U kSignBit(0x80000000u);
  const bool neg_a0 = inst.inst_.neg & 1u;
  const bool neg_b0 = inst.inst_.neg & 2u;
  const bool neg_acc = inst.inst_.neg & 4u;
  const bool neg_a1 = inst.inst_.neg_hi & 1u;
  const bool neg_b1 = inst.inst_.neg_hi & 2u;
  const auto widen = [](U raw) {
    if constexpr (Fmt == Vop3pDotHalfFormat::BF16)
      return util::bf16_to_f32_simd(raw);
    else
      return util::f16_to_f32_simd(raw);
  };
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = src0.template load_native<uint32_t>(base);
    const U raw1 = src1.template load_native<uint32_t>(base);
    F acc = src2.template load_native<float>(base);
    F a0 = widen(raw0 & 0xFFFFu);
    F a1 = widen(raw0 >> 16);
    F b0 = widen(raw1 & 0xFFFFu);
    F b1 = widen(raw1 >> 16);
    if (neg_a0)
      a0 = std::bit_cast<F>(std::bit_cast<U>(a0) ^ kSignBit);
    if (neg_b0)
      b0 = std::bit_cast<F>(std::bit_cast<U>(b0) ^ kSignBit);
    if (neg_a1)
      a1 = std::bit_cast<F>(std::bit_cast<U>(a1) ^ kSignBit);
    if (neg_b1)
      b1 = std::bit_cast<F>(std::bit_cast<U>(b1) ^ kSignBit);
    if (neg_acc)
      acc = std::bit_cast<F>(std::bit_cast<U>(acc) ^ kSignBit);
    F r = a0 * b0 + a1 * b1 + acc;
    dst.template store_native<float>(base, r, chunk);
  }
  return true;
}

template <Vop3pDotHalfFormat Fmt, typename Inst>
[[nodiscard]] bool try_execute_vop3p_dot_f16_simd(Inst &, Wavefront &) {
  return false;
}

template <int ElemBits, typename Inst>
[[nodiscard]] bool try_execute_vop3p_dot_int_mixed_simd(Inst &, Wavefront &) {
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

/// Common observed word access for local true16 and packed conversion paths.
/// The callback always sees unsigned words; it owns arithmetic and narrowing.
/// Partial-byte stores preserve the other half without reporting a false read.
template <unsigned Arity, bool E32, bool HalfDst, unsigned HalfInputs, typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_words_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst))
    return false;
  auto canonical = [](const auto &operand) {
    if constexpr (E32)
      return e32_word_operand(operand);
    else
      return operand;
  };
  auto first = canonical(inst.src0);
  auto destination = canonical(inst.vdst);
  using OperandT = decltype(first);
  std::optional<OperandT> second, third;
  if constexpr (Arity >= 2) {
    if constexpr (requires { inst.src1; })
      second.emplace(canonical(inst.src1));
    else
      second.emplace(canonical(inst.vsrc1));
  }
  if constexpr (Arity >= 3)
    third.emplace(canonical(inst.src2));
  if (!first.simd_capable() || !destination.simd_capable() || (second && !second->simd_capable()) ||
      (third && !third->simd_capable()))
    return false;
  uint32_t selectors = 0;
  if constexpr (E32) {
    auto high = [](const auto &operand) {
      return e32_packed_vgpr_operand(operand) && (operand.encoding_value() & 0x80);
    };
    selectors = high(inst.src0) ? 1 : 0;
    if constexpr (Arity >= 2) {
      if constexpr (requires { inst.src1; })
        selectors |= high(inst.src1) ? 2 : 0;
      else
        selectors |= high(inst.vsrc1) ? 2 : 0;
    }
    selectors |= (inst.inst_.vdst & 0x80) ? 8 : 0;
  } else if constexpr (HalfDst || HalfInputs) {
    selectors = vop3_opsel(inst.inst_);
  }
  using U = util::native<uint32_t>;
  constexpr uint32_t W = U::size();
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  auto bytes = [&](unsigned i) -> uint8_t {
    return (HalfInputs & (1u << i)) ? ((selectors & (1u << i)) ? 0xc : 0x3) : 0xf;
  };
  RegisterAccess regs(wf);
  auto a = regs.read_operand(first, exec, bytes(0));
  std::optional<RegisterAccess::OperandReadView> b, c;
  if constexpr (Arity >= 2)
    b.emplace(regs.read_operand(*second, exec, bytes(1)));
  if constexpr (Arity >= 3)
    c.emplace(regs.read_operand(*third, exec, bytes(2)));
  const bool high_dst = HalfDst && (selectors & 8);
  const bool zero_high = HalfDst && !E32 && !high_dst && cdna_vop3_low_dst_zeroes_high(wf);
  auto dst =
      regs.write_operand(destination, exec, HalfDst && !zero_high ? (high_dst ? 0xc : 3) : 0xf);
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    const uint64_t chunk = (exec >> base) & util::mask<uint64_t>(W);
    if (!chunk)
      continue;
    auto load = [&](const auto &view, unsigned i) {
      U v = view.template load_native<uint32_t>(base);
      if (HalfInputs & (1u << i))
        v = (v >> ((selectors & (1u << i)) ? 16 : 0)) & U(0xffffu);
      return v;
    };
    U result;
    if constexpr (Arity == 1)
      result = op(load(a, 0));
    else if constexpr (Arity == 2) {
      if constexpr (requires { op(load(a, 0), load(*b, 1), base); })
        result = op(load(a, 0), load(*b, 1), base);
      else
        result = op(load(a, 0), load(*b, 1));
    } else
      result = op(load(a, 0), load(*b, 1), load(*c, 2));
    if constexpr (HalfDst)
      result = (result & U(0xffffu)) << (high_dst ? 16 : 0);
    dst.template store_native<uint32_t>(base, result, chunk);
  }
  return true;
}

template <unsigned Arity, bool E32, bool HalfDst, unsigned HalfInputs, typename Inst, typename Op>
[[nodiscard]] bool try_execute_words_simd(Inst &, Wavefront &, Op) {
  return false;
}

template <typename T, typename Inst, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_div_scale_simd(Inst &inst, Wavefront &wf,
                                                     WriteResult commit) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr uint32_t W = util::native<T>::size();
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t rounding = sizeof(T) == 4 ? wf.fp_round_mode_f32() : wf.fp_round_mode_f16_f64();
  const uint32_t denorm = sizeof(T) == 4 ? wf.fp_denorm_mode_f32() : wf.fp_denorm_mode_f16_f64();
  RegisterAccess regs(wf);
  auto read = [&](const auto &operand) {
    if constexpr (sizeof(T) == 8)
      return regs.read_operand64(operand, exec);
    else
      return regs.read_operand(operand, exec);
  };
  auto a = read(inst.src0), b = read(inst.src1), c = read(inst.src2);
  auto dst = [&] {
    if constexpr (sizeof(T) == 8)
      return regs.write_operand64(inst.vdst, exec);
    else
      return regs.write_operand(inst.vdst, exec);
  }();
  uint64_t mask = wf.vcc();
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    uint64_t chunk = (exec >> base) & util::mask<uint64_t>(W);
    if (!chunk)
      continue;
    auto modify = [&](auto value, unsigned index) {
      using U = util::native<typename DivisionFormat<T>::Bits>;
      U bits = std::bit_cast<U>(value);
      if constexpr (requires { inst.inst_.abs; })
        if (inst.inst_.abs & (1u << index))
          bits &= U(~DivisionFormat<T>::sign);
      if (inst.inst_.neg & (1u << index))
        bits ^= U(DivisionFormat<T>::sign);
      return std::bit_cast<util::native<T>>(bits);
    };
    auto [value, bits] = div_scale_simd<T>(
        modify(a.template load_native<T>(base), 0), modify(b.template load_native<T>(base), 1),
        modify(c.template load_native<T>(base), 2), rounding, denorm);
    dst.template store_native<T>(base, value, chunk);
    mask = (mask & ~(chunk << base)) | ((bits & chunk) << base);
  }
  commit(mask);
  return true;
}

template <typename T, typename Inst, typename WriteResult>
[[nodiscard]] bool try_execute_div_scale_simd(Inst &, Wavefront &, WriteResult) {
  return false;
}

/// VOPD uses one instruction-wide read phase. Both slot results are buffered
/// before either destination is exposed for writing, including F64 pairs.
template <bool Extended, typename Slot>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopd_simd(const Slot &x, const Slot &y, Wavefront &wf) {
  if (simd_force_scalar())
    return false;
  auto capable = [](const Slot &slot) {
    return slot.src0->simd_capable() && (slot.op == 8 || slot.src1->simd_capable()) &&
           slot.dst->simd_capable() && (!slot.has_src2_operand || slot.src2->simd_capable());
  };
  if (!capable(x) || !capable(y))
    return false;
  // Each arithmetic slot must match its own MODE precision before either
  // slot reads operands. Other policies use the mode-aware scalar executor.
  auto is_arithmetic = [](const Slot &slot) {
    return slot.op <= 7 || (Extended && (slot.op == 19 || (slot.op >= 32 && slot.op <= 34)));
  };
  auto matches_mode = [&](const Slot &slot) {
    if (!is_arithmetic(slot))
      return true;
    const bool f64 = Extended && slot.op >= 32;
    return fp_mode::native_arithmetic_matches(
        f64 ? wf.fp_round_mode_f16_f64() : wf.fp_round_mode_f32(),
        f64 ? wf.fp_denorm_mode_f16_f64() : wf.fp_denorm_mode_f32());
  };
  if (!matches_mode(x) || !matches_mode(y))
    return false;
  // Matching controls still permit arithmetic to raise host exception flags.
  // Preserve the caller's environment just as the scalar arithmetic path does.
  std::optional<fp_mode::ScopedEnvironment> environment;
  if (is_arithmetic(x) || is_arithmetic(y))
    environment.emplace(0);
  struct Results {
    alignas(util::native<uint32_t>) uint32_t words[64]{};
    alignas(util::native<uint64_t>) uint64_t pairs[64]{};
  } xr, yr;
  const uint64_t exec = wf.exec();
  RegisterAccess regs(wf);
  auto compute = [&](const Slot &slot, Results &out) {
    if constexpr (Extended) {
      if (slot.op >= 32) {
        using D = util::native<double>;
        using U = util::native<uint64_t>;
        constexpr uint32_t W = D::size();
        auto a = regs.read_operand64(*slot.src0, exec);
        auto b = regs.read_operand64(*slot.src1, exec);
        std::optional<RegisterAccess::OperandRead64View> c;
        if (slot.op == 32)
          c.emplace(regs.read_operand64(*slot.src2, exec));
        for (uint32_t base = 0; base < wf.wf_size(); base += W) {
          if (!((exec >> base) & util::mask<uint64_t>(W)))
            continue;
          auto load = [&](const auto &view, unsigned i) {
            U bits = view.template load_native<uint64_t>(base);
            if (slot.neg & (1u << i))
              bits ^= U(0x8000000000000000ULL);
            return std::bit_cast<D>(bits);
          };
          D av = load(a, 0), bv = load(b, 1), result;
          switch (slot.op) {
          case 32:
            result = util::stdx::fma(av, bv, load(*c, 2));
            break;
          case 33:
            result = av + bv;
            break;
          case 34:
            result = av * bv;
            break;
          // The host's signed-zero and NaN selection is also used by the scalar
          // executor. Keep it for selection ops on hosts with broken F64 masks.
          case 35:
            result = D([&](auto i) { return std::fmax(double(av[i]), double(bv[i])); });
            break;
          case 36:
            result = D([&](auto i) { return std::fmin(double(av[i]), double(bv[i])); });
            break;
          default:
            return false;
          }
          std::bit_cast<U>(result).copy_to(out.pairs + base, util::stdx::vector_aligned);
        }
        return true;
      }
    }
    using U = util::native<uint32_t>;
    using I = util::native<int32_t>;
    using F = util::native<float>;
    constexpr uint32_t W = U::size();
    auto a = regs.read_operand(*slot.src0, exec);
    std::optional<RegisterAccess::OperandReadView> b, c, acc;
    if (slot.op != 8)
      b.emplace(regs.read_operand(*slot.src1, exec));
    if (slot.has_src2_operand && slot.op != 9)
      c.emplace(regs.read_operand(*slot.src2, exec));
    if (slot.op == 0)
      acc.emplace(regs.read_operand(*slot.dst, exec));
    const uint64_t condition =
        slot.op == 9 ? (slot.uses_vcc ? wf.vcc_mask(exec) : read_wave_mask_scalar(*slot.src2, wf))
                     : 0;
    for (uint32_t base = 0; base < wf.wf_size(); base += W) {
      if (!((exec >> base) & util::mask<uint64_t>(W)))
        continue;
      U av = a.template load_native<uint32_t>(base);
      U bv = b ? b->template load_native<uint32_t>(base) : U(0);
      U cv = c ? c->template load_native<uint32_t>(base) : U(slot.src2_imm);
      if ((slot.op <= 11 && slot.op != 8) || (Extended && slot.op == 19)) {
        if (slot.neg & 1)
          av ^= U(0x80000000u);
        if (slot.neg & 2)
          bv ^= U(0x80000000u);
        if (slot.neg & 4)
          cv ^= U(0x80000000u);
      }
      const F af = std::bit_cast<F>(av), bf = std::bit_cast<F>(bv), cf = std::bit_cast<F>(cv);
      U result;
      switch (slot.op) {
      case 0:
        result = std::bit_cast<U>(fma_f32_simd(af, bf, acc->template load_native<float>(base), wf));
        break;
      case 1:
      case 19:
        result = std::bit_cast<U>(fma_f32_simd(af, bf, cf, wf));
        break;
      case 2:
        result = std::bit_cast<U>(fma_f32_simd(af, cf, bf, wf));
        break;
      case 3:
        result = std::bit_cast<U>(binary_f32_simd<fp_mode::Arithmetic::MUL>(af, bf, wf));
        break;
      case 4:
        result = std::bit_cast<U>(binary_f32_simd<fp_mode::Arithmetic::ADD>(af, bf, wf));
        break;
      case 5:
        result = std::bit_cast<U>(af - bf);
        break;
      case 6:
        result = std::bit_cast<U>(bf - af);
        break;
      case 7:
        result = std::bit_cast<U>(af * bf);
        util::stdx::where(((av & U(0x7fffffffu)) == U(0)) || ((bv & U(0x7fffffffu)) == U(0)),
                          result) = U(0);
        break;
      case 8:
        result = av;
        break;
      case 9: {
        result = av;
        util::stdx::where(util::simd_mask_from_bits<U>(condition >> base), result) = bv;
        break;
      }
      case 10:
      case 11: {
        F selected = bf;
        if (slot.op == 10)
          util::stdx::where(af > bf, selected) = af;
        else
          util::stdx::where(af < bf, selected) = af;
        for (uint32_t i = 0; i < W; ++i)
          if (std::isnan(af[i]) || std::isnan(bf[i]) || (af[i] == 0 && bf[i] == 0))
            selected[i] = slot.op == 10 ? std::fmax(float(af[i]), float(bf[i]))
                                        : std::fmin(float(af[i]), float(bf[i]));
        result = std::bit_cast<U>(selected);
        break;
      }
      case 16:
        result = av + bv;
        break;
      case 17:
        result = bv << (av & U(31));
        break;
      case 18:
        if constexpr (Extended)
          result = bitop3_words(av, bv, U(0), uint8_t(slot.src2_imm));
        else
          result = av & bv;
        break;
      case 20:
        result = av - bv;
        break;
      case 21:
        result = bv >> (av & U(31));
        break;
      case 22:
        result =
            std::bit_cast<U>(std::bit_cast<I>(bv) >> util::stdx::static_simd_cast<I>(av & U(31)));
        break;
      case 23:
        result = std::bit_cast<U>(util::stdx::max(std::bit_cast<I>(av), std::bit_cast<I>(bv)));
        break;
      case 24:
        result = std::bit_cast<U>(util::stdx::min(std::bit_cast<I>(av), std::bit_cast<I>(bv)));
        break;
      default:
        return false;
      }
      result.copy_to(out.words + base, util::stdx::vector_aligned);
    }
    return true;
  };
  if (!compute(x, xr) || !compute(y, yr))
    return false;
  auto store = [&](const Slot &slot, const Results &result) {
    if constexpr (Extended) {
      if (slot.op >= 32) {
        auto dst = regs.write_operand64(*slot.dst, exec);
        constexpr uint32_t W = util::native<uint64_t>::size();
        for (uint32_t base = 0; base < wf.wf_size(); base += W)
          dst.template store_native<uint64_t>(
              base, util::native<uint64_t>(result.pairs + base, util::stdx::vector_aligned),
              (exec >> base) & util::mask<uint64_t>(W));
        return;
      }
    }
    auto dst = regs.write_operand(*slot.dst, exec);
    constexpr uint32_t W = util::native<uint32_t>::size();
    for (uint32_t base = 0; base < wf.wf_size(); base += W)
      dst.template store_native<uint32_t>(
          base, util::native<uint32_t>(result.words + base, util::stdx::vector_aligned),
          (exec >> base) & util::mask<uint64_t>(W));
  };
  store(x, xr);
  store(y, yr);
  return true;
}

template <bool Extended, typename Slot>
[[nodiscard]] bool try_execute_vopd_simd(const Slot &, const Slot &, Wavefront &) {
  return false;
}

template <bool Vop3, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_packed_fmac_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  auto &second = [&]() -> auto & {
    if constexpr (Vop3)
      return inst.src1;
    else
      return inst.vsrc1;
  }();
  if (!second.simd_capable())
    return false;
  uint32_t abs = 0, neg = 0, omod = 0, clamp = 0;
  if constexpr (Vop3) {
    abs = inst.inst_.abs;
    neg = inst.inst_.neg;
    clamp = inst.inst_.clamp;
    omod = fp_mode::effective_f16_omod(wf.cu().arch(), wf.fp_denorm_mode_f16_f64(), wf.ieee_mode(),
                                       true, inst.inst_.omod);
  }
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto a = regs.read_operand(inst.src0, exec), b = regs.read_operand(second, exec);
  auto dst = regs.readwrite_operand(inst.vdst, exec);
  using U = util::native<uint32_t>;
  constexpr uint32_t W = U::size();
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    uint64_t mask = (exec >> base) & util::mask<uint64_t>(W);
    if (!mask)
      continue;
    U av = a.template load_native<uint32_t>(base), bv = b.template load_native<uint32_t>(base);
    // Inline constants denote one half value broadcast to both elements;
    // literals and register sources already contain an independent packed pair.
    auto inline_pair = [](U raw, const auto &operand, uint32_t selector) {
      if (pk16_src_needs_narrowing(selector, operand.size_bits()))
        raw = util::f32_to_f16_simd(std::bit_cast<util::native<float>>(raw));
      if (dot2_src_needs_half_replication(selector)) {
        raw &= U(0xffffu);
        raw |= raw << 16;
      }
      return raw;
    };
    av = inline_pair(av, inst.src0, inst.inst_.src0);
    if constexpr (Vop3)
      bv = inline_pair(bv, second, inst.inst_.src1);
    U cv = dst.template load_native<uint32_t>(base);
    auto compute = [&](unsigned shift) {
      return fma_f16_mode_simd(
          (av >> shift) & U(0xffffu), (bv >> shift) & U(0xffffu), (cv >> shift) & U(0xffffu),
          abs & 1, abs & 2, false, neg & 1, neg & 2, false, wf.fp_round_mode_f16_f64(),
          wf.fp_denorm_mode_f16_f64(), omod, clamp, wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf),
          fp_mode::quiets_nan(wf.cu().arch(), wf.ieee_mode()));
    };
    U low = compute(0), high = compute(16);
    dst.template store_native<uint32_t>(base, low | (high << 16), mask);
  }
  return true;
}

template <bool Vop3, typename Inst>
[[nodiscard]] bool try_execute_packed_fmac_simd(Inst &, Wavefront &) {
  return false;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_H_
