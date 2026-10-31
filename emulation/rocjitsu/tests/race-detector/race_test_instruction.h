// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/race_detector/core/instruction_snapshot.h"

#include <array>

namespace rocjitsu::plugins::race_detector {

// Core tests construct memory events without executing ISA. A complete s_nop
// encoding supplies their instruction metadata without coupling them to a decoder.
inline InstructionSnapshot testInstruction(uint64_t pc, uint32_t word = 0xbf800000) {
  return {pc, std::array{word}};
}

} // namespace rocjitsu::plugins::race_detector
