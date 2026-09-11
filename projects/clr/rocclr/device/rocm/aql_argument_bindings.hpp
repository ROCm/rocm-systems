// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "aql_resident_program.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace amd::roc::aql_resident {
// Compiled ordinary-AQL kernarg-address -> resident binding relocations.
// No kernel metadata lookup, FAST packet reconstruction, or GPU dereference at
// bind time. Other dispatch changes deliberately take the validated encoder.
class ArgumentBindingPlan {
 public:
  static constexpr uint32_t kNoArgument = std::numeric_limits<uint32_t>::max();
  template <typename Allocator>
  bool compile(const std::vector<uint8_t, Allocator>& packets,
               const std::vector<uint32_t>& targetOffsets, const BindingPlan& plan) {
    if (packets.size() > UINT32_MAX || packets.size()/64 != targetOffsets.size() ||
        packets.size()%64) return false;
    ArgumentBindingPlan next;
    next.reference_.assign(packets.begin(), packets.end());
    next.bindingCount_ = plan.bindingCount();
    for (size_t i = 0; i < targetOffsets.size(); ++i) {
      if (targetOffsets[i] == kNoArgument) continue;
      for (uint32_t half = 0; half < 2; ++half) {
        const uint64_t target = uint64_t(targetOffsets[i]) + half*4;
        if (target % 4 || target > UINT32_MAX) return false;
        const auto entry = std::lower_bound(plan.entries().begin(), plan.entries().end(),
            target & ~uint64_t(7), [](const BindingFixup& e, uint64_t offset) {
              return e.targetOffset < offset;
            });
        if (entry == plan.entries().end() || entry->targetOffset != (target & ~uint64_t(7)) ||
            entry->bindingOffset || entry->bindingSlot >= next.bindingCount_) return false;
        next.words_.push_back({static_cast<uint32_t>(i*64+40+half*4),
                              entry->bindingSlot, uint32_t((target-entry->targetOffset)*8)});
        next.slots_.push_back(entry->bindingSlot);
      }
      next.requiredPointers_.push_back(static_cast<uint32_t>(i*64+40));
    }
    std::sort(next.slots_.begin(), next.slots_.end());
    next.slots_.erase(std::unique(next.slots_.begin(), next.slots_.end()), next.slots_.end());
    *this = std::move(next);
    return true;
  }

  template <typename Allocator>
  bool bind(const std::vector<uint8_t, Allocator>& packets, std::vector<uint64_t>& values,
            bool& changed) const {
    if (packets.size() != reference_.size() || values.size() != bindingCount_) return false;
    // Every non-address byte must match the already validated dispatch layout.
    // This includes geometry, scratch, reserved fields and completion signals.
    for (size_t offset = 0; offset < packets.size(); offset += 64) {
      if (std::memcmp(packets.data()+offset, reference_.data()+offset, 40) ||
          std::memcmp(packets.data()+offset+48, reference_.data()+offset+48, 16)) return false;
    }
    for (auto offset : requiredPointers_) {
      uint64_t address;
      std::memcpy(&address, packets.data()+offset, 8);
      if (!address) return false;
    }
    // No failure or allocation after mutation begins. Odd SET payload starts
    // can split the pointer across binding qwords; preserve companion words.
    bool any = false;
    for (const auto& word : words_) {
      uint32_t value;
      std::memcpy(&value, packets.data()+word.source, 4);
      const uint64_t mask = uint64_t(0xffffffffu) << word.shift;
      auto& target = values[word.slot];
      const uint64_t replacement = (target & ~mask) | (uint64_t(value) << word.shift);
      any |= replacement != target;
      target = replacement;
    }
    changed = any;
    return true;
  }
  // Only after the full encoder and existing template accepted a state change.
  // Relocation locations/layout are unchanged; copy without allocation so a
  // geometry update followed by an argument update uses the new reference.
  template <typename Allocator>
  bool rebaseValidated(const std::vector<uint8_t, Allocator>& packets) {
    if (packets.size() != reference_.size()) return false;
    std::memcpy(reference_.data(), packets.data(), packets.size());
    return true;
  }
  const std::vector<uint32_t>& slots() const { return slots_; }
 private:
  struct Word { uint32_t source, slot, shift; };
  std::vector<uint8_t> reference_;
  std::vector<Word> words_;
  std::vector<uint32_t> requiredPointers_;
  std::vector<uint32_t> slots_;
  size_t bindingCount_ = 0;
};
}  // namespace amd::roc::aql_resident
