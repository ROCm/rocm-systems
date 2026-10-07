// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3P_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3P_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

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

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3P_H_
