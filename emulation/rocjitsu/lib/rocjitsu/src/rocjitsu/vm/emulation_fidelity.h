// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file emulation_fidelity.h
/// @brief Hardware-contract fidelity policy for RocJitsu simulation.

#ifndef ROCJITSU_VM_EMULATION_FIDELITY_H_
#define ROCJITSU_VM_EMULATION_FIDELITY_H_

#include <stdexcept>
#include <string>
#include <string_view>

namespace rocjitsu {

/// @brief Selects how the emulator handles known hardware-contract violations.
///
/// Strict is fail-closed for checks RocJitsu explicitly models. It does not
/// imply that every hardware behavior or failure mode has already been modeled.
enum class EmulationFidelity {
  Permissive,
  Strict,
};

[[nodiscard]] inline EmulationFidelity parse_emulation_fidelity(std::string_view value) {
  if (value.empty() || value == "permissive")
    return EmulationFidelity::Permissive;
  if (value == "strict")
    return EmulationFidelity::Strict;
  throw std::invalid_argument("unknown emulation fidelity mode: " + std::string(value));
}

[[nodiscard]] inline constexpr std::string_view
emulation_fidelity_name(EmulationFidelity fidelity) {
  switch (fidelity) {
  case EmulationFidelity::Permissive:
    return "permissive";
  case EmulationFidelity::Strict:
    return "strict";
  }
  return "unknown";
}

} // namespace rocjitsu

#endif // ROCJITSU_VM_EMULATION_FIDELITY_H_
