// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Continuation of simd_glue.h after its shared helper definitions.
// Consumers include simd_glue.h rather than this implementation fragment.

namespace rocjitsu::amdgpu {

/// Apply a three-input truth table to all bits of a scalar or SIMD word.
template <typename U> inline U bitop3_words(U a, U b, U c, uint8_t table) {
  const auto choose = [](U mask, U yes, U no) { return (mask & yes) | (~mask & no); };
  const auto bit = [table](unsigned i) { return U(0) - U((table >> i) & 1u); };
  return choose(a, choose(b, choose(c, bit(7), bit(6)), choose(c, bit(5), bit(4))),
                choose(b, choose(c, bit(3), bit(2)), choose(c, bit(1), bit(0))));
}

// Source operand kinds admit both scalar and vector selectors. is_vgpr()
// describes that capability, not whether this particular selector names a VGPR.
template <typename Op> inline bool e32_packed_vgpr_operand(const Op &op) {
  using Kind = decltype(op.opr_type_);
  return op.size_bits() == 16 && !op.delegate() &&
         (op.opr_type_ == Kind::OPR_VGPR || (op.is_vgpr() && op.encoding_value() >= 256));
}

/// e32 true16 selectors encode the half in VGPR index bit 7. Construct a
/// full-word operand for the same physical register; keep staged DPP operands
/// intact because their half selection has already been applied.
template <typename Op> inline Op e32_word_operand(const Op &op) {
  if (!e32_packed_vgpr_operand(op))
    return op;
  using Kind = decltype(op.opr_type_);
  const IsaExecutionBackend backend{.operand_backend = Op::full_execution_backend()};
  ScopedIsaExecutionBackend scope(&backend);
  Op word(32, Kind::OPR_VGPR, op.encoding_value() & 0x7f);
  word.set_vgpr_msb_role(op.vgpr_msb_role());
  return word;
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

/// Vector exponent classification and exact normal-range scaling. Only lanes
/// requiring subnormal rounding, overflow or exceptional-value handling use
/// the scalar architectural primitive.
template <typename Float>
inline auto div_scale_simd(util::native<Float> value, util::native<Float> denominator,
                           util::native<Float> numerator, uint32_t rounding, uint32_t denorm) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  using U = util::native<Bits>;
  using I = util::native<std::make_signed_t<Bits>>;
  U v = std::bit_cast<U>(value), d = std::bit_cast<U>(denominator), n = std::bit_cast<U>(numerator);
  auto exp = [](U x) {
    return util::stdx::static_simd_cast<I>((x & U(F::infinity)) >> F::fraction);
  };
  if (!(denorm & 1u)) {
    util::stdx::where((v & U(F::infinity)) == U(0), v) = v & U(F::sign);
    util::stdx::where((d & U(F::infinity)) == U(0), d) = d & U(F::sign);
    util::stdx::where((n & U(F::infinity)) == U(0), n) = n & U(F::sign);
  }
  I de = exp(d), ne = exp(n), delta = ne - de;
  I adjustment(0);
  // Apply conditions from last to first to retain the scalar rule priority.
  util::stdx::where(ne <= I(sizeof(Float) == 4 ? 24 : 53), adjustment) = I(F::scale);
  const auto denominator_selected =
      util::stdx::static_simd_cast<I>(v) == util::stdx::static_simd_cast<I>(d);
  auto post = delta <= I(-F::threshold);
  util::stdx::where(post && !denominator_selected, adjustment) = I(F::scale);
  util::stdx::where(post && denominator_selected, adjustment) = I(0);
  const auto large_denominator = de >= I(2 * F::bias - 1);
  util::stdx::where(large_denominator, adjustment) = I(-F::scale);
  util::stdx::where(large_denominator && post && !denominator_selected, adjustment) = I(0);
  util::stdx::where(de == I(0), adjustment) = I(F::scale);
  post = post && de != I(0);
  const auto large_delta = delta >= I(F::threshold);
  util::stdx::where(large_delta, adjustment) = I(0);
  util::stdx::where(large_delta && denominator_selected, adjustment) = I(F::scale);
  post = post || large_delta;
  I ve = exp(v), scaled_exp = ve + adjustment;
  U result = (v & U(~F::infinity)) | (util::stdx::static_simd_cast<U>(scaled_exp) << F::fraction);
  uint64_t post_bits = 0;
  for (std::size_t i = 0; i < U::size(); ++i) {
    if (post[i])
      post_bits |= uint64_t{1} << i;
    if (ve[i] == 0 || ve[i] == int(F::infinity >> F::fraction) || scaled_exp[i] <= 0 ||
        scaled_exp[i] >= int(F::infinity >> F::fraction) || (d[i] & ~F::sign) == 0 ||
        (n[i] & ~F::sign) == 0) {
      const auto scalar =
          div_scale<Float>(value[i], denominator[i], numerator[i], rounding, denorm);
      result[i] = std::bit_cast<Bits>(scalar.value);
      post_bits = (post_bits & ~(uint64_t{1} << i)) | (uint64_t(scalar.post_scale) << i);
    }
  }
  return std::pair{std::bit_cast<util::native<Float>>(result), post_bits};
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
        slot.op == 9 ? (slot.uses_vcc ? wf.vcc() : read_wave_mask_scalar(*slot.src2, wf)) : 0;
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
        result = std::bit_cast<U>(util::stdx::fma(af, bf, acc->template load_native<float>(base)));
        break;
      case 1:
      case 19:
        result = std::bit_cast<U>(util::stdx::fma(af, bf, cf));
        break;
      case 2:
        result = std::bit_cast<U>(util::stdx::fma(af, cf, bf));
        break;
      case 3:
        result = std::bit_cast<U>(af * bf);
        break;
      case 4:
        result = std::bit_cast<U>(af + bf);
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
        U choose([&](auto i) { return uint32_t((condition >> (base + i)) & 1u); });
        result = av;
        util::stdx::where(choose != U(0), result) = bv;
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
          wf.fp_denorm_mode_f16_f64(), omod, clamp, wf.fp16_ovfl(), floating_clamp_nan_to_zero(wf));
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
