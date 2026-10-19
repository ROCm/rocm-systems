// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/output_modifier.h"
#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"

#include <cstdint>
#include <utility>

namespace rocjitsu::amdgpu::floating_operation {

/// @brief Bit i of each modifier field applies to source i.
struct SourceModifiers {
  uint32_t abs = 0;
  uint32_t neg = 0;
};

/// @brief ABS/NEG -> operation -> OMOD -> CLAMP, for scalar or SIMD raw bits.
/// @details All sources and the result use Fmt. The operation owns input
/// flushing, destination rounding, and any output flush before modifiers.
/// Signed input flushing commutes with ABS/NEG, so the operation may flush the
/// modified raw source before widening it.
template <typename Fmt, typename Op, typename... Vs>
constexpr auto apply(const SourceModifiers &source, const output_modifier::Policy &output,
                     const Op &operation, Vs... values) {
  const auto result = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return operation(source_modifier::apply<Fmt>(values, I, source.abs, source.neg)...);
  }(std::index_sequence_for<Vs...>{});
  return output_modifier::apply<Fmt>(result, output);
}

/// @brief Capture the modifier policies once, then apply them around each operation.
template <typename Fmt, typename Op> struct WithModifiers {
  SourceModifiers source;
  output_modifier::Policy output;
  Op operation;

  template <typename... Vs> constexpr auto operator()(Vs... values) const {
    return apply<Fmt>(source, output, operation, values...);
  }
};

/// @brief Whether `Op` is a WithModifiers wrapper that applies ABS/NEG and OMOD/CLAMP.
/// @details SIMD helpers may accept modifier-enabled instructions when this
/// wrapper supplies the modifier stages. Unwrapped operations keep their
/// existing guards.
template <typename Op> inline constexpr bool applies_modifiers_v = false;
template <typename Fmt, typename Op>
inline constexpr bool applies_modifiers_v<WithModifiers<Fmt, Op>> = true;

} // namespace rocjitsu::amdgpu::floating_operation
