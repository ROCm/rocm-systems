// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hazard_tracker.h
/// @brief GFX12 s_delay_alu auto-insertion for instruction emission.

#pragma once

#include "rocjitsu/code/patch/instruction_builder.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {

/// @brief Tracks instruction pipeline classes and auto-inserts s_delay_alu.
///
/// GFX12 requires s_delay_alu between dependent instructions from different
/// pipeline stages. This tracker maintains a short history of emitted
/// instructions and inserts the appropriate delay when a dependency exists.
class HazardTracker {
public:
  enum class Pipeline : uint8_t {
    None = 0,
    VALU = 1,
    TRANS = 2,
    SALU = 3,
  };

  void emit(std::vector<uint32_t> &words, uint32_t word, Pipeline pipeline);
  void emit2(std::vector<uint32_t> &words, uint32_t w0, uint32_t w1, Pipeline pipeline);
  void emit_raw(std::vector<uint32_t> &words, uint32_t word);

private:
  void maybe_insert_delay(std::vector<uint32_t> &words, Pipeline consumer);
  void advance(Pipeline producer);

  struct Slot {
    Pipeline pipeline = Pipeline::None;
    uint8_t distance = 0;
  };
  Slot slots_[2]{};
};

} // namespace rocjitsu
