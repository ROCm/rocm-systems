// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simulator_api.h"

#include <sstream>

namespace hazard_core {

std::string
SimulatorInstructionFormatter::format_location(const InstructionDescriptor &instruction) const {
  std::ostringstream out;
  out << "pc 0x" << std::hex << instruction.pc;
  return out.str();
}

std::string
SimulatorInstructionFormatter::format_wait_suggestion(WaitCntType kind, HazardAccessKind access,
                                                      HazardResourceLabel resource) const {
  return make_wait_suggestion(kind, access, resource);
}

} // namespace hazard_core
