// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOPD_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOPD_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

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

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOPD_H_
