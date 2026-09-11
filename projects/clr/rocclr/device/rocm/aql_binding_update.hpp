// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "aql_resident_program.hpp"
#include <algorithm>
#include <cstring>

namespace amd::roc::aql_resident {
// A per-submission upload description. 'previous' MUST describe the image on
// the target physical queue, not the most recently updated graph globally.
// Sparse entries use the same relocation ABI and GPU kernel as full updates.
class BindingUpdate {
 public:
  PlanStatus prepare(const BindingPlan& plan, const std::vector<uint64_t>& previous,
                     const std::vector<uint64_t>& current,
                     const std::vector<uint32_t>& argumentSlots = {}) {
    const auto status = plan.validateBindings(current);
    if (status != PlanStatus::Ok) return status;
    if (previous.size() != current.size()) return PlanStatus::Binding;
    for (size_t i = 0; i < argumentSlots.size(); ++i) {
      if (argumentSlots[i] >= current.size() || (i && argumentSlots[i] <= argumentSlots[i-1])) {
        return PlanStatus::Binding;
      }
    }
    BindingUpdate next;
    size_t changed = 0;
    bool argumentsOnly = !argumentSlots.empty();
    for (const auto& entry : plan.entries()) {
      if (previous[entry.bindingSlot] != current[entry.bindingSlot]) {
        ++changed;
        argumentsOnly &= std::binary_search(argumentSlots.begin(), argumentSlots.end(), entry.bindingSlot);
      }
    }
    if (changed) {
      const size_t denseBytes = current.size()*sizeof(uint64_t);
      const size_t sparseBytes = changed*(sizeof(uint64_t)+sizeof(BindingFixup));
      const size_t argumentBytes = argumentSlots.size()*sizeof(uint64_t);
      next.arguments_ = argumentsOnly && argumentBytes <= sparseBytes && argumentBytes < denseBytes;
      next.sparse_ = !next.arguments_ && sparseBytes < denseBytes;
      next.count_ = static_cast<uint32_t>(next.arguments_ ? argumentSlots.size() :
                                         next.sparse_ ? changed : plan.entries().size());
      next.bytes_ = next.arguments_ ? argumentBytes : next.sparse_ ? sparseBytes : denseBytes;
      if (next.arguments_) {
        next.storage_.resize(argumentBytes);
        for (size_t i = 0; i < argumentSlots.size(); ++i) {
          std::memcpy(next.storage_.data()+i*8, &current[argumentSlots[i]], 8);
        }
      } else if (next.sparse_) {
        next.entriesOffset_ = changed*sizeof(uint64_t);
        next.storage_.resize(sparseBytes);
        uint32_t index = 0;
        for (auto entry : plan.entries()) {
          if (previous[entry.bindingSlot] == current[entry.bindingSlot]) continue;
          const uint64_t value = current[entry.bindingSlot];
          entry.bindingSlot = index;
          std::memcpy(next.storage_.data()+index*sizeof(uint64_t), &value, sizeof(value));
          std::memcpy(next.storage_.data()+next.entriesOffset_+index*sizeof(entry), &entry, sizeof(entry));
          ++index;
        }
      }
    }
    *this = std::move(next);
    return PlanStatus::Ok;
  }
  const void* data(const std::vector<uint64_t>& current) const {
    return (sparse_ || arguments_) ? static_cast<const void*>(storage_.data()) : current.data();
  }
  size_t bytes() const { return bytes_; }
  size_t entriesOffset() const { return entriesOffset_; }
  uint32_t count() const { return count_; }
  bool sparse() const { return sparse_; }
  bool arguments() const { return arguments_; }
 private:
  std::vector<uint8_t> storage_;
  size_t bytes_ = 0, entriesOffset_ = 0;
  uint32_t count_ = 0;
  bool sparse_ = false;
  bool arguments_ = false;
};
}  // namespace amd::roc::aql_resident
