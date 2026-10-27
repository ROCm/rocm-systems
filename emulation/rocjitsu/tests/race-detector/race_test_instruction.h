// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/race_detector/core/instruction_snapshot.h"

namespace rocjitsu::plugins::race_detector {

// Core tests construct memory events without executing ISA. A complete s_nop
// encoding supplies their instruction metadata without coupling them to a decoder.
inline InstructionSnapshot testInstruction(uint64_t pc) {
  return {.pc = pc, .words = {0xbf800000}, .word_count = 1};
}

} // namespace rocjitsu::plugins::race_detector
