// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file coalesce.h
/// @brief Turning per-lane addresses into the work the memory system sees.
///
/// @details Two reductions live here, and both are only possible because the
/// timing plane runs inside a functional simulator: they consume the addresses
/// the kernel actually computed, not addresses inferred from source. That
/// distinction is the single largest error source in published analytical GPU
/// models. The same model equations scored 1.3 per cent against hardware when
/// fed measured line counts and 92 per cent when fed counts derived from
/// source, and the reason is here: a wave64 dword load is two cache lines when
/// contiguous and sixty-four when divergent, a thirtyfold difference that no
/// static analysis recovers.
///
/// Everything in this header is allocation-free and operates on at most one
/// wavefront's worth of lanes. A vector memory instruction runs it once per
/// execution, which makes it the hottest path the model has, and a heap
/// allocation per access would cost more than the emulator spends executing the
/// instruction.

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace rocjitsu::timing {

/// @brief Largest lane count this header is built for.
///
/// @details A wave64 on the widest target. Everything below is sized from it.
inline constexpr std::size_t kMaxLanes = 64;

/// @brief The shift that turns a byte address into a line address.
///
/// @details Rounds @p line_bytes up to a power of two, which is also how the
/// hardware indexes: a cache with a line size that is not a power of two would
/// need a divider in its address path, and none has one.
inline constexpr std::uint64_t line_shift_for(std::uint64_t line_bytes) {
  if (line_bytes <= 1)
    return 0;
  return static_cast<std::uint64_t>(64 - std::countl_zero(line_bytes - 1));
}

/// @brief Distinct-value set for at most one wavefront's worth of values.
///
/// @details Open addressed, entirely on the stack, no allocation. Capacity is
/// twice kMaxLanes so the table never exceeds half full and the probe sequence
/// stays short. Insertion order is preserved, because a caller that wants to
/// walk the distinct lines wants them in the order the lanes produced them:
/// that keeps a contiguous access walking memory forwards, which is what a
/// prefetcher-shaped model downstream would expect to see.
class LaneSet {
public:
  /// @brief Add @p value if it is not present.
  /// @returns true when @p value was not already in the set.
  bool insert(std::uint64_t value) {
    // Fibonacci hashing on the top bits: line addresses differ in their low
    // bits after the shift, and a mask of the raw value would put a contiguous
    // access into one probe chain.
    std::size_t slot = static_cast<std::size_t>((value * 0x9E3779B97F4A7C15ULL) >> 57) & kMask;
    for (std::size_t probe = 0; probe <= kMask; ++probe) {
      if (!occupied_[slot]) {
        occupied_[slot] = true;
        keys_[slot] = value;
        ordered_[count_] = value;
        ++count_;
        return true;
      }
      if (keys_[slot] == value)
        return false;
      slot = (slot + 1) & kMask;
    }
    // Unreachable while callers respect kMaxLanes: the table is twice that
    // size, so a free slot always exists. Reported as a duplicate rather than
    // written out of bounds.
    return false;
  }

  /// @brief Number of distinct values held.
  std::size_t size() const { return count_; }
  /// @brief Whether nothing was inserted.
  bool empty() const { return count_ == 0; }
  /// @brief The @p index'th distinct value, in insertion order.
  std::uint64_t operator[](std::size_t index) const { return ordered_[index]; }
  /// @brief Begin iterator over the distinct values, in insertion order.
  const std::uint64_t *begin() const { return ordered_; }
  /// @brief End iterator over the distinct values.
  const std::uint64_t *end() const { return ordered_ + count_; }

private:
  static constexpr std::size_t kCapacity = 2 * kMaxLanes;
  static constexpr std::size_t kMask = kCapacity - 1;
  static_assert((kCapacity & kMask) == 0, "capacity must be a power of two for the probe mask");

  bool occupied_[kCapacity] = {};
  std::uint64_t keys_[kCapacity] = {};
  std::uint64_t ordered_[kCapacity] = {};
  std::size_t count_ = 0;
};

/// @brief Reduce a lane address list to the distinct lines it touches.
///
/// @param addresses One byte address per participating lane, in lane order.
/// @param line_shift Line granularity, from line_shift_for() or a tag array.
/// @param bytes_per_lane Bytes each lane transfers, so that a lane whose range
///        crosses a line boundary is counted against both lines.
/// @param pool Line addresses are appended here, aligned down to the line.
/// @returns How many lines were appended.
///
/// @details The caller keeps the pool: a request travelling down the hierarchy
/// names a range of it rather than carrying its own vector, so that splitting a
/// request into a hit part and a miss part costs a partition of a subrange
/// instead of two allocations. Appending never reorders what is already there.
inline std::uint32_t coalesce_lines(std::span<const std::uint64_t> addresses,
                                    std::uint64_t line_shift, std::uint64_t bytes_per_lane,
                                    std::vector<std::uint64_t> &pool) {
  LaneSet lines;
  const std::uint64_t width = std::max<std::uint64_t>(1, bytes_per_lane);
  for (std::uint64_t address : addresses) {
    // A lane's access is a byte range, not a point, and a range that is not
    // aligned to the line straddles two of them. Taking only the starting line
    // undercounts exactly the accesses that are hardest on the memory system:
    // a misaligned or strided stream touches one more line per lane than an
    // aligned one and moves correspondingly more traffic. Both read half their
    // measured duration until this loop covered the whole range.
    const std::uint64_t first = address >> line_shift;
    const std::uint64_t last = (address + width - 1) >> line_shift;
    for (std::uint64_t line = first; line <= last; ++line)
      lines.insert(line);
  }
  for (std::uint64_t line : lines)
    pool.push_back(line << line_shift);
  return static_cast<std::uint32_t>(lines.size());
}

/// @brief Re-reduce line addresses to a coarser granularity, in place.
///
/// @param lines Line addresses, overwritten with the distinct coarser lines.
/// @param count How many entries @p lines holds.
/// @param line_shift The new, coarser granularity.
/// @returns How many entries are now live at the front of @p lines.
///
/// @details A level whose lines are wider than the level above it must not
/// treat the finer lines as separate work. Sixty-four byte first-level lines
/// arriving at a hundred-and-twenty-eight byte second level are at most half as
/// many accesses there, and skipping this would double that level's modelled
/// traffic and halve its hit rate for no reason other than bookkeeping.
inline std::uint32_t recoalesce_lines(std::uint64_t *lines, std::uint32_t count,
                                      std::uint64_t line_shift) {
  if (lines == nullptr || count == 0)
    return 0;
  LaneSet distinct;
  for (std::uint32_t index = 0; index < count; ++index)
    distinct.insert(lines[index] >> line_shift);
  std::uint32_t written = 0;
  for (std::uint64_t line : distinct)
    lines[written++] = line << line_shift;
  return written;
}

/// @brief Cycles a local data share access takes, including bank conflicts.
///
/// @param addresses One byte address per participating lane, in lane order.
/// @param bytes_per_lane Bytes each lane transfers.
/// @param banks Banks the local data share is built from.
/// @param bank_bytes Bytes one bank holds per row.
/// @param lanes_per_phase Lanes the hardware resolves together for a dword
///        access; narrower for a wider access.
/// @returns Cycles the access occupies the local data share for, at least one.
///
/// @details Conflicts are resolved the way the hardware resolves them: within a
/// phase group, never across the whole wavefront, and the group narrows as the
/// access widens, so a dword access resolves a whole wave64 at once while a
/// dwordx4 access takes four passes over sixteen lanes. Lanes reading the same
/// address broadcast for free and only distinct addresses landing on one bank
/// serialise, which is why the multiplicity is counted over the distinct set
/// and not over the lanes.
///
/// Resolving across the whole wavefront instead makes every blocked GEMM read
/// as conflict bound, because a tile staged through the local data share is
/// laid out to be conflict free per phase and is almost never conflict free
/// across sixty-four lanes.
inline std::uint64_t lds_conflict_cycles(std::span<const std::uint64_t> addresses,
                                         std::uint32_t bytes_per_lane, std::uint64_t banks,
                                         std::uint64_t bank_bytes, std::uint64_t lanes_per_phase) {
  if (addresses.empty())
    return 0;
  banks = std::max<std::uint64_t>(1, banks);
  bank_bytes = std::max<std::uint64_t>(1, bank_bytes);
  // A dword is the unit the phase width is quoted in, so an access narrower
  // than one still resolves a full phase rather than a wider one.
  const std::uint64_t width = std::max<std::uint64_t>(4, bytes_per_lane);
  const std::uint64_t phase_lanes =
      std::max<std::uint64_t>(1, std::max<std::uint64_t>(1, lanes_per_phase) * 4 / width);

  // Folded rather than sized from `banks`, to stay allocation free. Folding
  // over-counts conflicts, which is the safe direction, and no shipped part has
  // come close to this many banks.
  constexpr std::size_t kBankSlots = 256;
  std::uint8_t multiplicity[kBankSlots];

  std::uint64_t total = 0;
  const std::size_t count = addresses.size();
  for (std::size_t start = 0; start < count; start += phase_lanes) {
    const std::size_t end = std::min(count, start + static_cast<std::size_t>(phase_lanes));
    LaneSet distinct;
    for (std::size_t lane = start; lane < end; ++lane)
      distinct.insert(addresses[lane]);

    std::memset(multiplicity, 0, sizeof(multiplicity));
    std::uint32_t worst = 1;
    for (std::uint64_t address : distinct) {
      const std::size_t bank = static_cast<std::size_t>((address / bank_bytes) % banks);
      // At most kMaxLanes distinct addresses reach one bank, so the counter
      // cannot wrap its byte.
      const std::uint8_t seen = ++multiplicity[bank % kBankSlots];
      worst = std::max<std::uint32_t>(worst, seen);
    }
    total += worst;
  }
  return std::max<std::uint64_t>(1, total);
}

} // namespace rocjitsu::timing
