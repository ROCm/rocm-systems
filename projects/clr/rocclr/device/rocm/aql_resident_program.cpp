// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_resident_program.hpp"
#include <algorithm>
#include <limits>
#include <cstring>

namespace amd::roc::aql_resident {

PlanStatus BindingPlan::compile(uint64_t imageBytes, uint32_t bindingCount,
                               const std::vector<MutableRange>& mutableRanges,
                               const std::vector<BindingFixup>& entries) {
  // GPU ABI uses 32-bit relative offsets and a 32-bit entry count.
  if (!imageBytes || imageBytes > std::numeric_limits<uint32_t>::max() ||
      entries.size() > std::numeric_limits<uint32_t>::max()) return PlanStatus::Bounds;
  auto ranges = mutableRanges;
  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    return a.offset < b.offset;
  });
  uint64_t previousEnd = 0;
  for (const auto& range : ranges) {
    const uint64_t end = uint64_t(range.offset) + range.length;
    if (!range.length || end > imageBytes) return PlanStatus::Bounds;
    if (range.offset < previousEnd) return PlanStatus::Overlap;
    previousEnd = end;
  }
  auto checked = entries;
  std::sort(checked.begin(), checked.end(), [](const auto& a, const auto& b) {
    return a.targetOffset < b.targetOffset;
  });
  previousEnd = 0;
  size_t rangeIndex = 0;
  for (const auto& entry : checked) {
    if (entry.targetOffset % alignof(uint64_t)) return PlanStatus::Alignment;
    if (entry.bindingSlot >= bindingCount) return PlanStatus::Binding;
    const uint64_t end = uint64_t(entry.targetOffset) + sizeof(uint64_t);
    if (end > imageBytes) return PlanStatus::Bounds;
    if (entry.targetOffset < previousEnd) return PlanStatus::Overlap;
    previousEnd = end;
    while (rangeIndex < ranges.size() &&
           uint64_t(ranges[rangeIndex].offset) + ranges[rangeIndex].length <= entry.targetOffset) {
      ++rangeIndex;
    }
    if (rangeIndex == ranges.size() || entry.targetOffset < ranges[rangeIndex].offset ||
        end > uint64_t(ranges[rangeIndex].offset) + ranges[rangeIndex].length) {
      return PlanStatus::Bounds;
    }
  }
  entries_.swap(checked);
  imageBytes_ = imageBytes;
  bindingCount_ = bindingCount;
  return PlanStatus::Ok;
}

PlanStatus BindingPlan::validateBindings(const std::vector<uint64_t>& bindings) const {
  if (!imageBytes_ || bindings.size() != bindingCount_) return PlanStatus::Binding;
  for (const auto& entry : entries_) {
    if (bindings[entry.bindingSlot] >
        std::numeric_limits<uint64_t>::max() - entry.bindingOffset) return PlanStatus::Overflow;
  }
  return PlanStatus::Ok;
}

PlanStatus PacketTemplate::compile(const std::vector<Packet>& packets) {
  if (packets.empty() || packets.size() > 65535) return PlanStatus::Bounds;
  PacketTemplate next;
  next.packets_ = packets;
  std::vector<MutableRange> ranges;
  std::vector<BindingFixup> entries;
  for (size_t i = 0; i < packets.size(); ++i) {
    const auto& packet = packets[i];
    if ((packet[0] & 0xff) != 0) return PlanStatus::Binding; // vendor AQL
    const uint32_t format = (packet[0] >> 16) & 0xff;
    uint16_t words = 0;
    if (format == 9) {
      // Scratch-free layouts need no binding for their zero requirement.
      words = (packet[9] & 1u) ? 0x1ffe : 0x0ffe;
    } else if (format == 10) {
      const uint32_t base = (packet[0] >> 24) & 15;
      const uint32_t count = (packet[0] >> 28) + 1;
      if (count > 14 || base + count > 16) return PlanStatus::Bounds;
      for (uint32_t word = 16-count; word < 16; ++word) words |= uint16_t(1u << word);
    } else if (format == 8) {
      // Fixed FAST_BARRIER; its ordering/fence semantics cannot change at bind.
    } else if (format == 7 && i + 1 == packets.size()) {
      if (packet[0] != (7u << 16)) return PlanStatus::Binding;
      for (size_t word = 1; word < 16; ++word) {
        if (packet[word]) return PlanStatus::Binding; // terminal only
      }
    } else {
      return PlanStatus::Binding;
    }
    if (i + 1 == packets.size() && format != 7) return PlanStatus::Binding;
    for (uint32_t word = 0; word < 16; ++word) {
      if (words & (1u << word)) continue;
      const uint32_t offset = static_cast<uint32_t>(i*64 + word*4);
      if (!next.immutableSpans_.empty() &&
          next.immutableSpans_.back().offset + next.immutableSpans_.back().bytes == offset) {
        next.immutableSpans_.back().bytes += 4;
      } else {
        next.immutableSpans_.push_back({offset, 4});
      }
    }
    for (uint32_t pair = 0; pair < 8; ++pair) {
      if (!(words & (3u << (2*pair)))) continue;
      const uint32_t offset = static_cast<uint32_t>(i*64 + pair*8);
      // Some pairs include a constant companion word (e.g. dispatch header).
      // bind() verifies that word is unchanged; the GPU writes it identically.
      ranges.push_back({offset, 8});
      entries.push_back({offset, static_cast<uint32_t>(entries.size()), 0});
    }
  }
  const size_t packetBytes = packets.size()*64;
  const size_t bytes = (packetBytes + entries.size()*sizeof(BindingFixup) + 63) & ~size_t(63);
  const auto status = next.plan_.compile(bytes, static_cast<uint32_t>(entries.size()), ranges, entries);
  if (status != PlanStatus::Ok) return status;
  next.image_.resize(bytes, 0);
  std::memcpy(next.image_.data(), packets.data(), packetBytes);
  if (!entries.empty()) std::memcpy(next.image_.data()+packetBytes, entries.data(), entries.size()*sizeof(BindingFixup));
  *this = std::move(next);
  return PlanStatus::Ok;
}

PlanStatus PacketTemplate::bind(const std::vector<Packet>& packets,
                               std::vector<uint64_t>& values) const {
  if (packets_.empty() || packets.size() != packets_.size()) return PlanStatus::Bounds;
  const auto* supplied = reinterpret_cast<const uint8_t*>(packets.data());
  const auto* original = reinterpret_cast<const uint8_t*>(packets_.data());
  for (const auto& span : immutableSpans_) {
    if (std::memcmp(supplied + span.offset, original + span.offset, span.bytes)) {
      return PlanStatus::Binding;
    }
  }
  // All rejecting checks precede output mutation. resize preserves the vector
  // on allocation failure and reuses storage on repeated successful binds.
  values.resize(plan_.entries().size());
  size_t binding = 0;
  for (const auto& entry : plan_.entries()) {
    uint64_t value;
    const size_t index = entry.targetOffset/64;
    const size_t word = (entry.targetOffset%64)/4;
    std::memcpy(&value, packets[index].data()+word, sizeof(value));
    values[binding++] = value;
  }
  return PlanStatus::Ok;
}

}  // namespace amd::roc::aql_resident
