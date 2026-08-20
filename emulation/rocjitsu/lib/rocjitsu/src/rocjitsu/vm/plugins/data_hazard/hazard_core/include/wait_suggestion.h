// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The wording of the "add this wait" advice attached to every hazard. This is
// the single place the text is spelled out; frontends override
// SimulatorInstructionFormatter::format_wait_suggestion only where a simulator
// needs to word it differently.

#pragma once

#include "hazard_events.h"
#include "types.h"

#include <string>

namespace hazard_core {

/// Mnemonic of the wait instruction that drains @p type.
///
/// @param type The wait counter type
/// @return The mnemonic, or nullptr for counters that no wait can resolve
///         (NONE, XCNT which tracks address translation, and ASYNC)
inline const char *wait_counter_mnemonic(WaitCntType type) {
  switch (type) {
  case WaitCntType::VMEM:
    return "s_wait_loadcnt";
  case WaitCntType::SMEM:
    return "s_wait_kmcnt";
  case WaitCntType::LDS:
    return "s_wait_dscnt";
  case WaitCntType::STORE:
    return "s_wait_storecnt";
  case WaitCntType::TENSOR:
    return "s_wait_tensorcnt";
  case WaitCntType::NONE:
  case WaitCntType::XCNT:
  case WaitCntType::ASYNC:
  // SEMA orders LDS accesses between the waves of a wavegroup rather than
  // draining a counter of this wave, so the advice it needs names both halves
  // of the signal/wait pair and is worded where that race is reported.
  case WaitCntType::SEMA:
    break;
  }
  return nullptr;
}

/// What a suggestion tells the reader to wait on: the register the hazard names,
/// or the LDS address it names.
enum class HazardResourceLabel {
  Register,
  LdsAddress,
};

/// Build the wait suggestion for a hazard.
///
/// @param type     The wait counter that would resolve the hazard
/// @param access   Whether the hazardous access reads or writes the resource
/// @param resource Which resource the hazard message names
/// @return A suggestion like "Add s_wait_loadcnt 0 before reading this register",
///         or an empty string when no wait resolves @p type
inline std::string make_wait_suggestion(WaitCntType type, HazardAccessKind access,
                                        HazardResourceLabel resource) {
  const char *mnemonic = wait_counter_mnemonic(type);
  if (mnemonic == nullptr)
    return {};

  std::string suggestion = "Add ";
  suggestion += mnemonic;
  suggestion +=
      access == HazardAccessKind::Write ? " 0 before writing this " : " 0 before reading this ";
  suggestion += resource == HazardResourceLabel::LdsAddress ? "LDS address" : "register";
  return suggestion;
}

/// Get a human-readable suggestion for the wait instruction needed to
/// resolve a register hazard of the given type.
///
/// @param type The wait counter type
/// @return A suggestion string, or empty when no wait resolves @p type
inline std::string get_wait_suggestion(WaitCntType type) {
  switch (type) {
  case WaitCntType::STORE:
    // A pending store still reads its source register, so the hazard is a
    // later write to that register rather than a read of it.
    return make_wait_suggestion(type, HazardAccessKind::Write, HazardResourceLabel::Register);
  case WaitCntType::TENSOR:
    // Tensor loads land in LDS, so the hazard is a later access to that LDS
    // address rather than to a register.
    return make_wait_suggestion(type, HazardAccessKind::Read, HazardResourceLabel::LdsAddress);
  default:
    return make_wait_suggestion(type, HazardAccessKind::Read, HazardResourceLabel::Register);
  }
}

} // namespace hazard_core
