// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Guest conversion policies shared by scalar and SIMD implementations.
// Source flushing precedes conversion; destination rounding precedes OMOD and CLAMP.

#pragma once

#include "common/bits.h"
#include "conversion/conversion16.h"
#include "conversion/conversion32.h"
#include "conversion/conversion64.h"
#include "conversion/model/conversion.h"
#include "rj_arith/rj_arith.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
namespace conversion_detail {
namespace model = conversion_model::conversion;
namespace fmt = conversion_model::fp_format;

// Resolve one floating destination's policy from validated API controls.
template <typename From, typename To>
auto float_policy(uint64_t flags, uint32_t mode, bool quiet_nan = true) {
  model::ToFloat<From, To> result;
  if constexpr (!model::is_integer_v<From>) {
    result.abs = (mode & RJ_ARITH_ALU_ABS_A) != 0;
    result.neg = (mode & RJ_ARITH_ALU_NEG_A) != 0;
  }
  result.input.flush_inputs = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  result.rounding.direction = model::mode_rounding(flags >> 24);
  result.rounding.flush_tiny =
      (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (mode & RJ_ARITH_ALU_OMOD_MASK);
  result.rounding.saturate = std::is_same_v<To, fmt::F16> && (flags & RJ_ARITH_FP16_OVFL);
  result.output = {(mode >> 6) & 3, bool(mode & RJ_ARITH_ALU_CLAMP),
                   !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF), uint32_t((flags >> 24) & 3),
                   bool(flags & RJ_ARITH_FP16_OVFL)};
  result.quiet_nan = quiet_nan;
  return result;
}

// Convert selected FP16/FP32 or integer encodings, leaving halfword insertion to the caller.
template <Conversion16 Op, typename V>
V convert16(V raw, uint64_t flags, uint32_t mode, bool quiet_nan = true) {
  if constexpr (Op != Conversion16::FloatToHalf) {
    raw = (raw >> ((mode & RJ_ARITH_ALU_HIGH_A) ? 16 : 0)) & V(0xffffU);
  }
  if constexpr (Op == Conversion16::SignedToHalf) {
    return float_policy<model::I16, fmt::F16>(flags, mode)(raw);
  } else if constexpr (Op == Conversion16::UnsignedToHalf) {
    return float_policy<model::U16, fmt::F16>(flags, mode)(raw);
  } else if constexpr (Op == Conversion16::FloatToHalf) {
    return float_policy<fmt::F32, fmt::F16>(flags, mode)(raw);
  } else if constexpr (Op == Conversion16::HalfToFloat) {
    return float_policy<fmt::F16, fmt::F32>(flags, mode, quiet_nan)(raw);
  } else {
    raw = float_policy<fmt::F16, fmt::F32>(flags & ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                                           mode & (RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A))(raw);
    using Int = std::conditional_t<Op == Conversion16::HalfToSigned, int16_t, uint16_t>;
    return model::f32_to_integer<Int>(raw, model::Rounding::TOWARD_ZERO, model::NanResult::ZERO) &
           V(0xffffU);
  }
}

// Convert FP32/integer encodings using the instruction's destination rounding policy.
template <Conversion32 Op, typename V> V convert32(V raw, uint64_t flags, uint32_t mode) {
  if constexpr (Op == Conversion32::SignedToFloat) {
    return float_policy<model::I32, fmt::F32>(flags, mode)(raw);
  } else if constexpr (Op == Conversion32::UnsignedToFloat) {
    return float_policy<model::U32, fmt::F32>(flags, mode)(raw);
  } else {
    raw = conversion_model::source_modifier::apply<fmt::F32>(
        raw, 0, (mode & RJ_ARITH_ALU_ABS_A) != 0, (mode & RJ_ARITH_ALU_NEG_A) != 0);
    raw = conversion_model::input_denormal::flush_input<fmt::F32>(
        raw, {bool(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS)});
    constexpr auto rounding = Op == Conversion32::Nearest ? model::Rounding::NEAREST_UP
                              : Op == Conversion32::Floor ? model::Rounding::FLOOR
                                                          : model::Rounding::TOWARD_ZERO;
    constexpr auto nan = Op == Conversion32::Nearest || Op == Conversion32::Floor
                             ? model::NanResult::SATURATE_BY_SIGN
                             : model::NanResult::ZERO;
    using Int = std::conditional_t<Op == Conversion32::FloatToUnsigned, uint32_t, int32_t>;
    return model::f32_to_integer<Int>(raw, rounding, nan);
  }
}

// Convert to or from FP64; raw lanes are wide enough for either format.
template <Conversion64 Op, typename V>
auto convert64(V raw, uint64_t flags, uint32_t mode, bool quiet_nan) {
  if constexpr (Op == Conversion64::SignedToDouble) {
    return float_policy<model::I32, fmt::F64>(flags, mode)(raw);
  } else if constexpr (Op == Conversion64::UnsignedToDouble) {
    return float_policy<model::U32, fmt::F64>(flags, mode)(raw);
  } else if constexpr (Op == Conversion64::FloatToDouble) {
    return float_policy<fmt::F32, fmt::F64>(flags, mode, quiet_nan)(raw);
  } else {
    static_assert(Op == Conversion64::DoubleToFloat);
    return float_policy<fmt::F64, fmt::F32>(flags, mode)(raw);
  }
}
// Resolve byte packing's source modifiers, flushing, and guest FP32 rounding.
inline auto byte_pack_policy(uint64_t flags, uint32_t mode) {
  model::PackU8 result;
  result.abs = (mode & RJ_ARITH_ALU_ABS_A) != 0;
  result.neg = (mode & RJ_ARITH_ALU_NEG_A) != 0;
  result.input.flush_inputs = flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
  result.rounding = model::mode_rounding(flags >> 24);
  return result;
}

} // namespace conversion_detail
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
