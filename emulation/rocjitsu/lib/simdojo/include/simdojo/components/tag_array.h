// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tag_array.h
/// @brief A run-time configured set-associative tag array: residency without data.

#ifndef SIMDOJO_COMPONENTS_TAG_ARRAY_H_
#define SIMDOJO_COMPONENTS_TAG_ARRAY_H_

#include "util/bit.h"

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief A set-associative tag array with least-recently-used replacement.
///        Tags only, no data and no coherence state.
///
/// @details Deliberately not simdojo::Cache, and both exist for different
/// jobs. Cache takes its geometry as template parameters and allocates a real
/// data array; this takes geometry at run time and allocates tags only,
/// because a model that just wants to know whether a line *would* have hit
/// should not pay for the data behind the answer, and wants hundreds of
/// instances whose geometry comes from a configuration file rather than from
/// the source. Tags are not free: an entry is twenty-four bytes, so against
/// sixty-four-byte lines this costs about three-eighths of what the data would
/// -- a saving of roughly 2.7x, not of everything. Cache's replacement policy
/// shuffles a per-set recency list on every hit; this writes one stamp.
///
/// Copying one copies its entries, which for a large array is a large copy.
/// Callers that keep arrays in a container should hold or move them, not
/// copy them by accident.
///
/// The victim is the way with the oldest stamp, and a free way is always older
/// than any occupant, so no separate rule is needed to fill an empty set
/// before evicting from it. Nothing about the search consults an address, a
/// hash, or uninitialised memory, so two runs over the same access sequence
/// leave the same lines resident -- which is what makes a model built on this
/// reproducible.
class TagArray {
public:
  TagArray() = default;

  /// @brief Construct and configure in one step.
  /// @param sets Number of sets; must be a power of two.
  /// @param ways Associativity; must be positive.
  /// @param line_bytes Line size in bytes; must be a power of two.
  /// @throws std::invalid_argument or std::length_error, as configure() does.
  TagArray(uint64_t sets, uint64_t ways, uint64_t line_bytes) { configure(sets, ways, line_bytes); }

  TagArray(const TagArray &) = default;
  TagArray &operator=(const TagArray &) = default;

  /// @brief Move, leaving the source with no geometry rather than a stale one.
  ///
  /// @details The default move would take entries_ and leave sets_, ways_ and
  /// the line size behind, so the moved-from array would report configured()
  /// false while sets() and line_bytes() still answered with the geometry it
  /// no longer has -- a plausible non-zero answer from an object every
  /// operation throws on. Every field moves together instead.
  TagArray(TagArray &&other) noexcept { *this = std::move(other); }
  TagArray &operator=(TagArray &&other) noexcept;

  /// @brief Most entries a tag array may hold.
  ///
  /// @details A host allocation limit, not a hardware one. Geometry comes from
  /// a configuration file, where a units mistake -- sets given in bytes, say --
  /// asks for hundreds of terabytes; the request passes every arithmetic check,
  /// and the operating system then hands out pages until it kills the process,
  /// with nothing to point at. An entry is twenty-four bytes, so this bounds one
  /// array at a gigabyte and a half of host memory.
  static constexpr uint64_t kMaxEntries = 1ULL << 26;

  /// @brief Largest cache, in bytes, a tag array may model.
  ///
  /// @details kMaxEntries bounds what the array costs the host; this bounds
  /// what it claims to be, and the two catch different mistakes. Without it a
  /// line size is the one dimension nothing checks: sets * ways stays small
  /// while a line size given in the wrong units -- bits, or a stray shift --
  /// collapses every address in a normal virtual range onto one or two lines,
  /// so the array reports a hit for nearly everything and throws nothing. Four
  /// gigabytes is larger than any cache being modelled.
  static constexpr uint64_t kMaxCapacityBytes = 1ULL << 32;

  /// @brief Size the array and invalidate everything in it.
  ///
  /// @details Sets and line size must be powers of two, and are rejected
  /// rather than rounded. A rounded geometry is a different cache from the one
  /// the configuration asked for, and it would answer every question slightly
  /// wrong with nothing to point at; the constraint itself is the hardware's,
  /// which indexes with a mask because no cache has a divider in its address
  /// path.
  /// @param sets Number of sets; must be a power of two.
  /// @param ways Associativity; must be positive.
  /// @param line_bytes Line size in bytes; must be a power of two.
  /// @throws std::invalid_argument if a dimension is zero, or if sets or the
  ///         line size is not a power of two.
  /// @throws std::length_error if sets * ways exceeds kMaxEntries, or if
  ///         sets * ways * line_bytes exceeds kMaxCapacityBytes.
  void configure(uint64_t sets, uint64_t ways, uint64_t line_bytes);

  /// @brief Probe for the line holding @p byte_address, allocating on a miss.
  ///
  /// @details On a miss the line is installed, evicting the least recently
  /// used way of its set if every way is valid.
  /// @param byte_address Byte address to look up.
  /// @param vmid Address space the address belongs to. Part of the tag: two
  ///        guests can hold the same virtual address, and a shared array that
  ///        ignored this would report a hit for one on the other's line.
  /// @retval true The line was already resident.
  /// @retval false The line was not resident and has now been allocated.
  /// @throws std::logic_error if the array has no geometry yet.
  bool access(uint64_t byte_address, uint32_t vmid = 0);

  /// @brief Whether the line holding @p byte_address is resident, without
  ///        allocating it or disturbing replacement order.
  ///
  /// @details For a model asking about an access that does not allocate -- a
  /// non-temporal load, or a probe from another agent.
  /// @param byte_address Byte address to look up.
  /// @param vmid Address space the address belongs to.
  /// @retval true The line is resident.
  /// @retval false The line is not resident.
  /// @throws std::logic_error if the array has no geometry yet.
  bool contains(uint64_t byte_address, uint32_t vmid = 0) const;

  /// @brief Drop the line holding @p byte_address, if it is resident.
  /// @param byte_address Byte address to invalidate.
  /// @param vmid Address space the address belongs to.
  /// @retval true A resident line was dropped.
  /// @retval false Nothing was resident to drop.
  /// @throws std::logic_error if the array has no geometry yet.
  bool invalidate(uint64_t byte_address, uint32_t vmid = 0);

  /// @brief Drop every line, keeping the geometry.
  ///
  /// @details The recency counter is deliberately never reset, by any
  /// operation. It is sixty-four bits and nothing needs it to be small,
  /// whereas a reset while any line remained resident -- which invalidate()
  /// leaves and this does not -- would make that line look newer than every
  /// line filled afterwards and invert the replacement order. Keeping the one
  /// rule for both means neither has to be reasoned about separately.
  /// @throws std::logic_error if the array has no geometry yet.
  void invalidate_all();

  /// @brief Number of sets.
  uint64_t sets() const { return sets_; }
  /// @brief Associativity.
  uint64_t ways() const { return ways_; }
  /// @brief Line size in bytes.
  uint64_t line_bytes() const { return line_bytes_; }
  /// @brief Log2 of the line size, for turning a byte address into a line number.
  uint32_t line_shift() const { return line_shift_; }
  /// @brief Whether configure() has been called with a usable geometry.
  bool configured() const { return !entries_.empty(); }

private:
  /// @brief One way of one set.
  ///
  /// @details A free way is a default-constructed one, and every path that
  /// frees a way assigns a whole Entry rather than clearing the flag. That
  /// keeps the invariant the victim search relies on: an invalid way's stamp
  /// is zero, and a resident way's is a positive value of clock_.
  struct Entry {
    uint64_t line = 0;  ///< Line number; meaningless unless valid.
    uint64_t stamp = 0; ///< Value of clock_ at the last hit or fill; 0 if free.
    uint32_t vmid = 0;  ///< Address space; part of the tag.
    bool valid = false; ///< Whether this way holds a line.
  };

  /// @brief A decoded address: its line number, and where that line's set starts.
  struct Location {
    uint64_t line;    ///< Line number.
    std::size_t base; ///< Index of way zero of the set the line indexes into.
  };

  /// @brief Index of way zero of the set @p line indexes into.
  std::size_t set_base(uint64_t line) const {
    // sets_ - 1 is the index mask, so a zero set count would mask nothing off
    // and index past the array. configure() is the only writer of both fields
    // and rejects a zero, and the move above keeps them in step.
    assert(sets_ != 0);
    return static_cast<std::size_t>((line & (sets_ - 1)) * ways_);
  }

  /// @brief Decode @p byte_address once, for the three operations that need both halves.
  Location locate(uint64_t byte_address) const {
    const uint64_t line = byte_address >> line_shift_;
    return {line, set_base(line)};
  }

  /// @brief Whether @p entry is the resident way holding @p line of @p vmid.
  ///
  /// @details One definition of a tag match, so the probing paths and the
  /// allocating path cannot come to disagree about what is resident.
  static bool matches(const Entry &entry, uint64_t line, uint32_t vmid) {
    return entry.valid && entry.line == line && entry.vmid == vmid;
  }

  /// @brief Throw unless a geometry has been set.
  void require_configured() const;

  /// @brief Find the resident way holding @p line in the set based at @p base.
  /// @returns Index into entries_, or entries_.size() when not resident.
  std::size_t find(std::size_t base, uint64_t line, uint32_t vmid) const;

  uint64_t sets_ = 0;
  uint64_t ways_ = 0;
  uint64_t line_bytes_ = 0;
  uint32_t line_shift_ = 0;
  /// @brief Monotonic recency source. One increment per hit or fill, which is
  /// what a per-set recency list costs a shuffle for.
  uint64_t clock_ = 0;
  std::vector<Entry> entries_;
};

// Defined inline, like every other simdojo component: access() is called once
// per simulated memory access, LTO is off by default, and simdojo_headers is
// an INTERFACE target that several binaries link without the object library --
// an out-of-line definition would hand the first of them a link error.

inline void TagArray::configure(uint64_t sets, uint64_t ways, uint64_t line_bytes) {
  if (sets == 0 || ways == 0 || line_bytes == 0)
    throw std::invalid_argument("TagArray dimensions must be positive");
  if (!util::is_power_of_2(sets) || !util::is_power_of_2(line_bytes))
    throw std::invalid_argument("TagArray set count and line size must be powers of two");

  const auto entry_count = util::checked_mul(sets, ways);
  if (!entry_count || *entry_count > kMaxEntries)
    throw std::length_error("TagArray geometry exceeds the largest array worth modelling");

  const auto capacity_bytes = util::checked_mul(*entry_count, line_bytes);
  if (!capacity_bytes || *capacity_bytes > kMaxCapacityBytes)
    throw std::length_error("TagArray geometry exceeds the largest cache worth modelling");

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

inline TagArray &TagArray::operator=(TagArray &&other) noexcept {
  if (this == &other)
    return *this;
  entries_ = std::move(other.entries_);
  other.entries_.clear();
  sets_ = std::exchange(other.sets_, 0);
  ways_ = std::exchange(other.ways_, 0);
  line_bytes_ = std::exchange(other.line_bytes_, 0);
  line_shift_ = std::exchange(other.line_shift_, 0);
  clock_ = std::exchange(other.clock_, 0);
  return *this;
}

inline void TagArray::require_configured() const {
  if (entries_.empty())
    throw std::logic_error("TagArray must be configured before it is used");
}

inline std::size_t TagArray::find(std::size_t base, uint64_t line, uint32_t vmid) const {
  for (uint64_t way = 0; way < ways_; ++way) {
    const std::size_t index = base + static_cast<std::size_t>(way);
    if (matches(entries_[index], line, vmid))
      return index;
  }
  return entries_.size();
}

inline bool TagArray::access(uint64_t byte_address, uint32_t vmid) {
  require_configured();

  const auto [line, base] = locate(byte_address);

  // One pass over the set: a miss has to look at every way anyway, so it picks
  // its victim on the way past rather than walking the set a second time.
  //
  // The victim is simply the oldest stamp. That needs no separate rule for a
  // free way, because a free way's stamp is zero and a resident way's is a
  // positive value of clock_, so a free way is always the older. No two
  // resident ways share a stamp, so the only ties are between free ways, and
  // which of those is taken is not observable through this interface.
  std::size_t victim = base;
  uint64_t oldest = entries_[base].stamp;
  for (uint64_t way = 0; way < ways_; ++way) {
    const std::size_t index = base + static_cast<std::size_t>(way);
    Entry &entry = entries_[index];
    if (matches(entry, line, vmid)) {
      entry.stamp = ++clock_;
      return true;
    }
    if (entry.stamp < oldest) {
      oldest = entry.stamp;
      victim = index;
    }
  }

  // Designated, because line and stamp are both uint64_t: a positional list
  // would still compile if the two were ever reordered, and would store the
  // clock as the tag.
  entries_[victim] = Entry{.line = line, .stamp = ++clock_, .vmid = vmid, .valid = true};
  return false;
}

inline bool TagArray::contains(uint64_t byte_address, uint32_t vmid) const {
  require_configured();
  const auto [line, base] = locate(byte_address);
  return find(base, line, vmid) != entries_.size();
}

inline bool TagArray::invalidate(uint64_t byte_address, uint32_t vmid) {
  require_configured();
  const auto [line, base] = locate(byte_address);
  const std::size_t index = find(base, line, vmid);
  if (index == entries_.size())
    return false;
  // A whole Entry, not just the flag: the victim search reads a free way's
  // stamp as zero.
  entries_[index] = Entry{};
  return true;
}

inline void TagArray::invalidate_all() {
  require_configured();
  entries_.assign(entries_.size(), Entry{});
}

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_TAG_ARRAY_H_
