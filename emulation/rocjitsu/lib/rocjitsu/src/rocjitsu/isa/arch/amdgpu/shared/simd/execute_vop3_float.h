// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3_FLOAT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3_FLOAT_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

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

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3_FLOAT_H_
