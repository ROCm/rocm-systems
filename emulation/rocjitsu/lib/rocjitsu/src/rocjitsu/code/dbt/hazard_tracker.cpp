// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/hazard_tracker.h"

#include "rocjitsu/isa/arch/amdgpu/rdna4/opcodes.h"

namespace rocjitsu {

void HazardTracker::maybe_insert_delay(std::vector<uint32_t> &words, Pipeline consumer) {
  if (consumer == Pipeline::None)
    return;

  // s_delay_alu simm16: instid0[3:0] at bits[3:0], instid1[3:0] at bits[10:7].
  // Each instid = pipeline[1:0] | (distance[2:0] << 2). 0 = no dependency.
  uint16_t simm16 = 0;
  for (int i = 0; i < 2; ++i) {
    auto &s = slots_[i];
    if (s.pipeline == Pipeline::None || s.distance > 4)
      continue;
    uint8_t dep = static_cast<uint8_t>(s.pipeline) | (s.distance << 2);
    if (i == 0)
      simm16 |= dep;
    else
      simm16 |= (static_cast<uint16_t>(dep) << 7);
  }

  if (simm16 != 0) {
    words.push_back(pack_sopp(rdna4::kSDelayAluSopp, simm16));
  }
}

void HazardTracker::advance(Pipeline producer) {
  for (auto &s : slots_) {
    if (s.pipeline != Pipeline::None)
      ++s.distance;
    if (s.distance > 7)
      s.pipeline = Pipeline::None;
  }
  if (producer != Pipeline::None) {
    slots_[1] = slots_[0];
    slots_[0] = {producer, 0};
  }
}

void HazardTracker::emit(std::vector<uint32_t> &words, uint32_t word, Pipeline pipeline) {
  maybe_insert_delay(words, pipeline);
  words.push_back(word);
  advance(pipeline);
}

void HazardTracker::emit2(std::vector<uint32_t> &words, uint32_t w0, uint32_t w1,
                          Pipeline pipeline) {
  maybe_insert_delay(words, pipeline);
  words.push_back(w0);
  words.push_back(w1);
  advance(pipeline);
}

void HazardTracker::emit_raw(std::vector<uint32_t> &words, uint32_t word) {
  words.push_back(word);
  advance(Pipeline::None);
}

} // namespace rocjitsu
