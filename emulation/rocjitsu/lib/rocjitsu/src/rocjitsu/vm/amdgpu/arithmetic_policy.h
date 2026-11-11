// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @brief Per-CU arithmetic policy saved in checkpoints, and host execution flags
/// rebuilt on restore for the current CPU.

#pragma once

#include "util/simd_control.h"

#include <cstdint>
#include <string_view>

namespace rocjitsu::amdgpu {

/// @brief rj-arith semantics modes, before placement in the flags bit field.
enum class ArithmeticSemantics : uint8_t {
  Loose = 0,
  BitExactEmpiricalAllowLoose = 1,
  BitExactEmpiricalNoFpStatus = 2,
  BitExactEmpirical = 3,
  // Temporary per-instruction migration policy; outside the 0..3 ordering.
  Compatibility = 7,
};

/// @brief Per-CU arithmetic policy; omission selects temporary Compatibility.
struct ArithmeticPolicy {
  ArithmeticSemantics semantics = ArithmeticSemantics::Compatibility;
};

/// @brief Parse an arithmetic semantics name, rejecting unknown or reserved modes.
ArithmeticPolicy parse_arithmetic_policy(std::string_view semantics);

/// @brief Return the configuration spelling of an arithmetic semantics mode.
const char *arithmetic_semantics_name(ArithmeticSemantics semantics);

/// @brief Instruction-independent rj-arith policy initialized before CU execution.
class ArithmeticExecution {
public:
  explicit ArithmeticExecution(ArithmeticPolicy policy = {});

  /// @brief Return cached CPU/semantics flags with the live scalar-test override.
  /// Guest instruction controls are added separately by the execution adapters.
  uint64_t flags() const { return util::force_scalar() ? flags_ & ~cpu_flags_mask : flags_; }

private:
  // The low 16 bits contain CPU level and feature flags; checked against the API in the
  // constructor.
  static constexpr uint64_t cpu_flags_mask = 0xffff;
  uint64_t flags_;
};

} // namespace rocjitsu::amdgpu
