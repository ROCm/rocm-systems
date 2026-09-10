// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tag_array.h
/// @brief The set-associative tag store and LRU replacement shared by every
///        simdojo cache: residency, with or without data behind it.

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

/// @brief The per-way tag an occupancy-only TagArray keeps: which line, in
///        which address space, and whether the way holds one at all.
///
/// @details Also the shape BasicTagArray requires of any entry type: a
/// default-constructed entry is a free way, and `tag`, `vmid` and `valid` are
/// the fields the array reads and writes. Anything else in an entry -- a
/// cache's dirty bit and coherence state -- belongs to the caller, and is reset
/// to its default on every fill and invalidation.
struct TagEntry {
  uint64_t tag = 0;   ///< Line number shifted past the set index; meaningless unless valid.
  uint32_t vmid = 0;  ///< Address space; part of the tag.
  bool valid = false; ///< Whether this way holds a line.
};

/// @brief A run-time configured set-associative tag store with
///        least-recently-used replacement.
///
/// @details The one implementation of tag matching and replacement in simdojo.
/// simdojo::Cache is built on it, with an entry type that carries coherence
/// state and a data array beside it; TagArray is the same store with nothing
/// else, for a model that only wants to know whether a line *would* have hit.
///
/// Lookup and allocation are separate, so a cache controller can keep a miss
/// outstanding: find() probes without changing anything, touch() records a
/// use, and fill() installs a line when its data arrives and reports what it
/// displaced, so the controller can write the victim back or keep an
/// inclusive level consistent. access() is the probe-and-allocate shorthand
/// for a model that treats a miss as filled at once.
///
/// The victim is the way with the oldest stamp, and a free way always reads
/// as older than any occupant, so a set is filled before anything in it is
/// evicted. Stamps come from a per-set counter: LRU only ever compares the
/// ways of one set, and a per-set counter is what lets a caller that locks per
/// set -- the functional L2 does -- use two sets from two threads. Nothing
/// about the search consults a host address or uninitialised memory, so two
/// runs over the same access sequence leave the same lines resident.
///
/// Copying copies every entry, which for a large array is a large copy.
///
/// @tparam Entry Per-way tag type; see TagEntry for the fields it must have.
template <typename Entry> class BasicTagArray {
public:
  /// @brief Index of one way in the array, as find() and fill() return it.
  using Index = std::size_t;

  /// @brief What fill() did.
  struct Fill {
    Index index;              ///< The way the new line now occupies.
    Entry evicted;            ///< The line displaced; `evicted.valid` is false if the way was free.
    uint64_t evicted_address; ///< Line-aligned byte address of the displaced line, if any.
  };

  BasicTagArray() = default;

  /// @brief Construct and configure in one step.
  /// @param sets Number of sets; must be a power of two.
  /// @param ways Associativity; must be positive.
  /// @param line_bytes Line size in bytes; must be a power of two.
  /// @throws std::invalid_argument or std::length_error, as configure() does.
  BasicTagArray(uint64_t sets, uint64_t ways, uint64_t line_bytes) {
    configure(sets, ways, line_bytes);
  }

  BasicTagArray(const BasicTagArray &) = default;

  /// @brief Copy-assign with the strong guarantee.
  ///
  /// @details Copy, then swap. Memberwise assignment would write the scalar
  /// geometry before copying the entries, so an allocation failure part way
  /// through would leave the destination with its old, smaller storage and the
  /// source's geometry -- and the next lookup would index past the end of it.
  BasicTagArray &operator=(const BasicTagArray &other) {
    BasicTagArray copy(other);
    swap(copy);
    return *this;
  }

  /// @brief Move, leaving the source with no geometry rather than a stale one.
  BasicTagArray(BasicTagArray &&other) noexcept { swap(other); }

  /// @brief Move-assign, leaving the source with no geometry.
  BasicTagArray &operator=(BasicTagArray &&other) noexcept {
    BasicTagArray moved(std::move(other));
    swap(moved);
    return *this;
  }

  /// @brief Exchange every field with @p other.
  void swap(BasicTagArray &other) noexcept {
    std::swap(sets_, other.sets_);
    std::swap(ways_, other.ways_);
    std::swap(line_bytes_, other.line_bytes_);
    std::swap(line_shift_, other.line_shift_);
    std::swap(set_shift_, other.set_shift_);
    slots_.swap(other.slots_);
    set_clocks_.swap(other.set_clocks_);
  }

  /// @brief Most entries a tag array may hold.
  ///
  /// @details A host allocation limit, not a hardware one. Geometry comes from
  /// a configuration file, where a units mistake -- sets given in bytes, say --
  /// asks for hundreds of terabytes; the request passes every arithmetic check,
  /// and the operating system then hands out pages until it kills the process,
  /// with nothing to point at.
  static constexpr uint64_t kMaxEntries = 1ULL << 26;

  /// @brief Largest cache, in bytes, a tag array may model.
  ///
  /// @details kMaxEntries bounds what the array costs the host; this bounds
  /// what it claims to be, and the two catch different mistakes. Without it,
  /// sets * ways stays small while a line size given in the wrong units -- bits,
  /// or a stray shift -- collapses a normal address range onto one or two lines,
  /// so the array reports a hit for nearly everything and throws nothing. Four
  /// gigabytes is larger than any cache being modelled.
  static constexpr uint64_t kMaxCapacityBytes = 1ULL << 32;

  /// @brief Size the array and invalidate everything in it.
  ///
  /// @details Sets and line size must be powers of two, and are rejected
  /// rather than rounded. A rounded geometry is a different cache from the one
  /// the configuration asked for; the constraint itself is the hardware's,
  /// which indexes with a mask because no cache has a divider in its address
  /// path. A rejected geometry, or a failed allocation, leaves the array as it
  /// was.
  /// @param sets Number of sets; must be a power of two.
  /// @param ways Associativity; must be positive.
  /// @param line_bytes Line size in bytes; must be a power of two.
  /// @throws std::invalid_argument if a dimension is zero, or if sets or the
  ///         line size is not a power of two.
  /// @throws std::length_error if sets * ways exceeds kMaxEntries, or if
  ///         sets * ways * line_bytes exceeds kMaxCapacityBytes.
  void configure(uint64_t sets, uint64_t ways, uint64_t line_bytes);

  /// @brief Find the way holding the line of @p byte_address, without
  ///        changing anything.
  ///
  /// @details A probe: replacement order is left alone, so a model asking
  /// about an access that does not allocate -- a non-temporal load, a probe
  /// from another agent, a miss it will fill later -- does not disturb it.
  /// Follow with touch() to count the probe as a use.
  /// @param byte_address Byte address to look up.
  /// @param vmid Address space the address belongs to. Part of the tag: two
  ///        guests can hold the same virtual address, and a shared array that
  ///        ignored this would report a hit for one on the other's line.
  /// @returns The way's index, or npos() when the line is not resident.
  /// @throws std::logic_error if the array has no geometry yet.
  Index find(uint64_t byte_address, uint32_t vmid = 0) const;

  /// @brief Record a use of the way at @p index, making it the most recently
  ///        used way of its set.
  /// @param index A resident way, as find() or fill() returned it.
  void touch(Index index) {
    assert(index < slots_.size() && slots_[index].entry.valid);
    slots_[index].stamp = ++set_clocks_[index / ways_];
  }

  /// @brief Install the line of @p byte_address, evicting the least recently
  ///        used way of its set if every way is occupied.
  ///
  /// @details Does not look for the line first; a caller that might already
  /// hold it probes with find(). The new entry is a default-constructed Entry
  /// with its tag fields set, so any caller-owned state in it starts from its
  /// default, and it becomes the most recently used way of its set.
  /// @param byte_address Byte address of the line to install.
  /// @param vmid Address space the line belongs to.
  /// @returns The way filled, and the line it displaced.
  /// @throws std::logic_error if the array has no geometry yet.
  Fill fill(uint64_t byte_address, uint32_t vmid = 0);

  /// @brief Probe for the line holding @p byte_address, filling it on a miss.
  ///
  /// @details find(), then touch() on a hit or fill() on a miss.
  /// @param byte_address Byte address to look up.
  /// @param vmid Address space the address belongs to.
  /// @retval true The line was already resident.
  /// @retval false The line was not resident and has now been allocated.
  /// @throws std::logic_error if the array has no geometry yet.
  bool access(uint64_t byte_address, uint32_t vmid = 0);

  /// @brief Whether the line holding @p byte_address is resident, without
  ///        allocating it or disturbing replacement order.
  /// @param byte_address Byte address to look up.
  /// @param vmid Address space the address belongs to.
  /// @retval true The line is resident.
  /// @retval false The line is not resident.
  /// @throws std::logic_error if the array has no geometry yet.
  bool contains(uint64_t byte_address, uint32_t vmid = 0) const {
    return find(byte_address, vmid) != npos();
  }

  /// @brief Drop the line holding @p byte_address, if it is resident.
  /// @param byte_address Byte address to invalidate.
  /// @param vmid Address space the address belongs to.
  /// @retval true A resident line was dropped.
  /// @retval false Nothing was resident to drop.
  /// @throws std::logic_error if the array has no geometry yet.
  bool invalidate(uint64_t byte_address, uint32_t vmid = 0);

  /// @brief Drop the line holding @p byte_address in every address space.
  /// @param byte_address Byte address to invalidate.
  /// @throws std::logic_error if the array has no geometry yet.
  void invalidate_all_vmids(uint64_t byte_address);

  /// @brief Drop every line, keeping the geometry.
  ///
  /// @details The recency counters are deliberately never reset, by any
  /// operation. A reset while any line remained resident -- which invalidate()
  /// leaves and this does not -- would make that line look newer than every
  /// line filled afterwards and invert the replacement order.
  /// @throws std::logic_error if the array has no geometry yet.
  void invalidate_all();

  /// @brief The entry at @p index.
  Entry &entry(Index index) { return slots_[index].entry; }
  /// @brief The entry at @p index.
  const Entry &entry(Index index) const { return slots_[index].entry; }

  /// @brief Line-aligned byte address of the line held at @p index.
  /// @details Meaningful only while the entry is valid.
  uint64_t line_address(Index index) const {
    const uint64_t set = index / ways_;
    return ((slots_[index].entry.tag << set_shift_) | set) << line_shift_;
  }

  /// @brief Call @p fn(entry, line_address, index) for every resident line.
  template <typename F> void for_each_valid(F &&fn) {
    for (Index index = 0; index < slots_.size(); ++index)
      if (slots_[index].entry.valid)
        fn(slots_[index].entry, line_address(index), index);
  }

  /// @brief The index find() returns for a line that is not resident.
  Index npos() const { return slots_.size(); }

  /// @brief Number of sets.
  uint64_t sets() const { return sets_; }
  /// @brief Associativity.
  uint64_t ways() const { return ways_; }
  /// @brief Line size in bytes.
  uint64_t line_bytes() const { return line_bytes_; }
  /// @brief Log2 of the line size, for turning a byte address into a line number.
  uint32_t line_shift() const { return line_shift_; }
  /// @brief Whether configure() has been called with a usable geometry.
  bool configured() const { return !slots_.empty(); }

private:
  /// @brief One way of one set: the caller-visible entry, and its recency.
  ///
  /// @details A free way is a default-constructed one, and every path that
  /// frees a way assigns a whole Slot rather than clearing the flag. That keeps
  /// the invariant the victim search relies on: a free way's stamp is zero,
  /// and a resident way's is a positive value of its set's counter.
  struct Slot {
    Entry entry{};
    uint64_t stamp = 0; ///< Set counter at the last use or fill; 0 if free.
  };

  /// @brief A decoded address: its tag, and where its set starts.
  struct Location {
    uint64_t tag;     ///< Line number shifted past the set index.
    std::size_t base; ///< Index of way zero of the set.
  };

  /// @brief Decode @p byte_address into its tag and set.
  Location locate(uint64_t byte_address) const {
    const uint64_t line = byte_address >> line_shift_;
    return {line >> set_shift_, static_cast<std::size_t>((line & (sets_ - 1)) * ways_)};
  }

  /// @brief Whether @p entry is the resident way holding @p tag of @p vmid.
  ///
  /// @details One definition of a tag match, so the probing paths and the
  /// allocating path cannot come to disagree about what is resident.
  static bool matches(const Entry &entry, uint64_t tag, uint32_t vmid) {
    return entry.valid && entry.tag == tag && entry.vmid == vmid;
  }

  /// @brief Throw unless a geometry has been set.
  void require_configured() const {
    if (slots_.empty())
      throw std::logic_error("TagArray must be configured before it is used");
  }

  /// @brief The least recently used way of the set based at @p base: a free
  ///        way if there is one, and the lowest-indexed of several.
  Index victim(std::size_t base) const {
    Index oldest = base;
    for (uint64_t way = 1; way < ways_; ++way) {
      const Index index = base + static_cast<std::size_t>(way);
      if (slots_[index].stamp < slots_[oldest].stamp)
        oldest = index;
    }
    return oldest;
  }

  uint64_t sets_ = 0;
  uint64_t ways_ = 0;
  uint64_t line_bytes_ = 0;
  uint32_t line_shift_ = 0;
  uint32_t set_shift_ = 0;
  std::vector<Slot> slots_;
  /// @brief Per-set recency source. One increment per use or fill.
  std::vector<uint64_t> set_clocks_;
};

/// @brief A tag array with nothing behind the tags: residency only.
///
/// @details For a model that just wants to know whether a line *would* have
/// hit, and should not pay for the data behind the answer.
using TagArray = BasicTagArray<TagEntry>;

// Defined inline, like every other simdojo component: simdojo_headers is an
// INTERFACE target that several binaries link without the object library, and
// these are per-memory-access hot paths in an LTO-off default build.

template <typename Entry>
inline void BasicTagArray<Entry>::configure(uint64_t sets, uint64_t ways, uint64_t line_bytes) {
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

  // Built first and swapped in, so a rejected geometry -- or a throwing
  // allocation -- leaves the array as it was rather than half-replaced.
  std::vector<Slot> slots(static_cast<std::size_t>(*entry_count));
  std::vector<uint64_t> clocks(static_cast<std::size_t>(sets));

  slots_.swap(slots);
  set_clocks_.swap(clocks);
  sets_ = sets;
  ways_ = ways;
  line_bytes_ = line_bytes;
  line_shift_ = static_cast<uint32_t>(std::countr_zero(line_bytes));
  set_shift_ = static_cast<uint32_t>(std::countr_zero(sets));
}

template <typename Entry>
inline typename BasicTagArray<Entry>::Index BasicTagArray<Entry>::find(uint64_t byte_address,
                                                                       uint32_t vmid) const {
  require_configured();
  const auto [tag, base] = locate(byte_address);
  for (uint64_t way = 0; way < ways_; ++way) {
    const Index index = base + static_cast<std::size_t>(way);
    if (matches(slots_[index].entry, tag, vmid))
      return index;
  }
  return npos();
}

template <typename Entry>
inline typename BasicTagArray<Entry>::Fill BasicTagArray<Entry>::fill(uint64_t byte_address,
                                                                      uint32_t vmid) {
  require_configured();
  const auto [tag, base] = locate(byte_address);
  const Index index = victim(base);
  Slot &slot = slots_[index];

  Fill result{index, slot.entry, slot.entry.valid ? line_address(index) : 0};
  slot.entry = Entry{};
  slot.entry.tag = tag;
  slot.entry.vmid = vmid;
  slot.entry.valid = true;
  slot.stamp = ++set_clocks_[index / ways_];
  return result;
}

template <typename Entry>
inline bool BasicTagArray<Entry>::access(uint64_t byte_address, uint32_t vmid) {
  const Index index = find(byte_address, vmid);
  if (index != npos()) {
    touch(index);
    return true;
  }
  fill(byte_address, vmid);
  return false;
}

template <typename Entry>
inline bool BasicTagArray<Entry>::invalidate(uint64_t byte_address, uint32_t vmid) {
  const Index index = find(byte_address, vmid);
  if (index == npos())
    return false;
  // A whole Slot, not just the flag: the victim search reads a free way's
  // stamp as zero.
  slots_[index] = Slot{};
  return true;
}

template <typename Entry>
inline void BasicTagArray<Entry>::invalidate_all_vmids(uint64_t byte_address) {
  require_configured();
  const auto [tag, base] = locate(byte_address);
  for (uint64_t way = 0; way < ways_; ++way) {
    Slot &slot = slots_[base + static_cast<std::size_t>(way)];
    if (slot.entry.valid && slot.entry.tag == tag)
      slot = Slot{};
  }
}

template <typename Entry> inline void BasicTagArray<Entry>::invalidate_all() {
  require_configured();
  slots_.assign(slots_.size(), Slot{});
}

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_TAG_ARRAY_H_
