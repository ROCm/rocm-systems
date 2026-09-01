// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tag_array.h
/// @brief A run-time configured set-associative tag array: residency without data.

#ifndef SIMDOJO_COMPONENTS_TAG_ARRAY_H_
#define SIMDOJO_COMPONENTS_TAG_ARRAY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace simdojo {

/// @brief A set-associative tag array with least-recently-used replacement.
///        Tags only, no data and no coherence state.
///
/// @details Deliberately not simdojo::Cache, and both exist for different
/// jobs. Cache takes its geometry as template parameters and allocates a real
/// data array; this takes geometry at run time and allocates none, because a
/// model that only wants to know whether a line *would* have hit does not want
/// the megabytes of storage that answering it with real data would cost, and
/// wants hundreds of instances whose geometry comes from a configuration file
/// rather than from the source. Cache's replacement policy shuffles a per-set
/// recency list on every hit; this writes one stamp.
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

  /// @brief Most entries a tag array may hold.
  ///
  /// @details An allocation limit, not a hardware one. Geometry comes from a
  /// configuration file, where a units mistake -- sets given in bytes, say --
  /// asks for hundreds of terabytes; the request passes every arithmetic check,
  /// and the operating system then hands out pages until it kills the process,
  /// with nothing to point at. At sixty-four-byte lines this is a four-gigabyte
  /// cache, which is larger than anything being modelled.
  static constexpr uint64_t kMaxEntries = 1ULL << 26;

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
  /// @throws std::length_error if sets * ways exceeds kMaxEntries.
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
  /// @details The recency counter is deliberately not reset. It is
  /// sixty-four bits and nothing needs it to be small, whereas a reset while
  /// any line remained resident would make that line look newer than every
  /// line filled afterwards and invert the replacement order.
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

  /// @brief Index of way zero of the set @p line indexes into.
  std::size_t set_base(uint64_t line) const {
    return static_cast<std::size_t>((line & (sets_ - 1)) * ways_);
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

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_TAG_ARRAY_H_
