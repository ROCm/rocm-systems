// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/hazard_tracker.h"

namespace rocjitsu {
namespace {

[[nodiscard]] uint8_t encode_delay_instid(HazardTracker::Pipeline pipeline, uint8_t distance) {
  // OPR_DELAY INSTID values are enumerated by the GFX12 ISA, not bit-packed
  // as pipeline | distance. distance is zero-based in the tracker: 0 means the
  // producer is the immediately preceding instruction, which is *_DEP_1.
  switch (pipeline) {
  case HazardTracker::Pipeline::VALU:
    return distance < 4 ? static_cast<uint8_t>(1 + distance) : 0;
  case HazardTracker::Pipeline::TRANS:
    return distance < 3 ? static_cast<uint8_t>(5 + distance) : 0;
  case HazardTracker::Pipeline::SALU:
    return distance < 3 ? static_cast<uint8_t>(9 + distance) : 0;
  case HazardTracker::Pipeline::None:
    return 0;
  }
  return 0;
}

} // namespace

void HazardTracker::maybe_insert_delay(std::vector<uint32_t> &words, Pipeline consumer) {
  if (consumer == Pipeline::None)
    return;

  // s_delay_alu simm16: INSTID0 is bits[3:0]. INSTID1/INSTSKIP can describe a
  // second delayed VALU instruction, but this tracker only models the next
  // instruction. Emit one dependency for that instruction instead of packing an
  // unmodeled second descriptor into the same wait.
  for (auto &s : slots_) {
    const uint8_t instid = encode_delay_instid(s.pipeline, s.distance);
    if (instid == 0)
      continue;
    constexpr uint8_t kSoppDelayAlu = 7;
    words.push_back(pack_sopp(kSoppDelayAlu, instid));
    return;
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
