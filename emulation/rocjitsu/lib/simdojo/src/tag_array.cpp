// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/components/tag_array.h"

#include "util/bit.h"

#include <bit>
#include <stdexcept>

namespace simdojo {

void TagArray::configure(uint64_t sets, uint64_t ways, uint64_t line_bytes) {
  if (sets == 0 || ways == 0 || line_bytes == 0)
    throw std::invalid_argument("TagArray dimensions must be positive");
  if (!util::is_power_of_2(sets) || !util::is_power_of_2(line_bytes))
    throw std::invalid_argument("TagArray set count and line size must be powers of two");

  const auto entry_count = util::checked_mul(sets, ways);
  if (!entry_count || *entry_count > kMaxEntries)
    throw std::length_error("TagArray geometry exceeds the largest array worth modelling");

  // Built first and moved in, so a rejected geometry -- or a throwing
  // allocation -- leaves the array as it was rather than half-replaced.
  std::vector<Entry> entries(static_cast<std::size_t>(*entry_count));

  entries_ = std::move(entries);
  sets_ = sets;
  ways_ = ways;
  line_bytes_ = line_bytes;
  line_shift_ = static_cast<uint32_t>(std::countr_zero(line_bytes));
  clock_ = 0;
}

void TagArray::require_configured() const {
  if (entries_.empty())
    throw std::logic_error("TagArray must be configured before it is used");
}

std::size_t TagArray::find(std::size_t base, uint64_t line, uint32_t vmid) const {
  for (uint64_t way = 0; way < ways_; ++way) {
    const std::size_t index = base + static_cast<std::size_t>(way);
    const Entry &entry = entries_[index];
    if (entry.valid && entry.line == line && entry.vmid == vmid)
      return index;
  }
  return entries_.size();
}

bool TagArray::access(uint64_t byte_address, uint32_t vmid) {
  require_configured();

  const uint64_t line = byte_address >> line_shift_;
  const std::size_t base = set_base(line);

  // One pass over the set: a miss has to look at every way anyway, so it picks
  // its victim on the way past rather than walking the set a second time.
  //
  // The victim is simply the oldest stamp. That needs no separate rule for a
  // free way, because a free way's stamp is zero and a resident way's is a
  // positive value of clock_, so a free way is always the older. No two
  // resident ways share a stamp, so the only ties are between free ways, and
  // which of those is taken is not observable through this interface.
  std::size_t victim = base;
  for (uint64_t way = 0; way < ways_; ++way) {
    const std::size_t index = base + static_cast<std::size_t>(way);
    Entry &entry = entries_[index];
    if (entry.valid && entry.line == line && entry.vmid == vmid) {
      entry.stamp = ++clock_;
      return true;
    }
    if (entry.stamp < entries_[victim].stamp)
      victim = index;
  }

  entries_[victim] = Entry{line, ++clock_, vmid, true};
  return false;
}

bool TagArray::contains(uint64_t byte_address, uint32_t vmid) const {
  require_configured();
  const uint64_t line = byte_address >> line_shift_;
  return find(set_base(line), line, vmid) != entries_.size();
}

bool TagArray::invalidate(uint64_t byte_address, uint32_t vmid) {
  require_configured();
  const uint64_t line = byte_address >> line_shift_;
  const std::size_t index = find(set_base(line), line, vmid);
  if (index == entries_.size())
    return false;
  // A whole Entry, not just the flag: the victim search reads a free way's
  // stamp as zero.
  entries_[index] = Entry{};
  return true;
}

void TagArray::invalidate_all() {
  require_configured();
  for (Entry &entry : entries_)
    entry = Entry{};
}

} // namespace simdojo
