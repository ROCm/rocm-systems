// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_ARITHMETIC_POLICY_H_
#define ROCJITSU_VM_AMDGPU_ARITHMETIC_POLICY_H_

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
  uint64_t flags() const;

private:
  uint64_t flags_;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_ARITHMETIC_POLICY_H_
