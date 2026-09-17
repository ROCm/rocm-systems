// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/register_set.h"
#include <cstdint>

namespace rocjitsu {
class Instruction;
}

namespace rocjitsu::consan {

/// Local proof that a VGPR has one value across the current participating lanes.
/// Start fresh at each basic block. EXEC changes discard all facts; only plain
/// broadcasts and copies establish new ones. This deliberately does not merge
/// facts across CFG edges or reason about inactive lanes.
class UniformAddressTracker {
public:
  [[nodiscard]] bool contains(uint16_t vgpr) const {
    return vgpr < kTrackedVgprs && uniform_.contains({RegClass::VGPR, vgpr, 1});
  }
  void observe(const Instruction &instruction);

private:
  static constexpr uint16_t kTrackedVgprs = 256;
  RegisterSet uniform_;
};
} // namespace rocjitsu::consan
