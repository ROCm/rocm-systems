// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cache.h
/// @brief Set-associative cache data structure with MOESI coherence tags, built
///        on the shared tag store in tag_array.h.

#ifndef SIMDOJO_COMPONENTS_CACHE_H_
#define SIMDOJO_COMPONENTS_CACHE_H_

#include "simdojo/components/tag_array.h"
#include "util/bit.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace simdojo {

/// @brief Coherence state for a cache line (MOESI protocol).
enum class CoherenceState : uint8_t {
  INVALID,
  SHARED,
  EXCLUSIVE,
  MODIFIED,
  OWNED,
};

/// @brief Tag and metadata for a single cache line.
///
/// @details @c vmid identifies the owning process address space. Lines with the
/// same line address but different vmids are distinct entries, because guest VAs
/// are per-process and may alias across processes. @c tag, @c vmid and @c valid
/// are the fields the tag store matches on; the rest are the controller's, and
/// start from their defaults whenever a line is allocated.
struct CacheTag {
  uint64_t tag = 0;
  uint64_t coherence_epoch = 0; ///< Controller-defined lazy-invalidation generation.
  uint32_t vmid = 0;
  bool valid = false;
  bool dirty = false;
  CoherenceState coherence = CoherenceState::INVALID;
};

/// @brief Set-associative cache data structure with configurable geometry.
///
/// @details Pure data structure, not a simulation Component. Cache controllers wrap
/// this to implement protocol-specific behavior (write-back, write-through,
/// coherence transitions).
///
/// Tag matching and LRU replacement are BasicTagArray's, shared with the
/// timing model's TagArray; this adds a data array indexed by the same ways.
/// The geometry is fixed at compile time, which is what lets controllers use
/// set_index() and tag_bits() as constants.
///
/// @tparam LineSizeBits Log2 of the cache line size in bytes.
/// @tparam NumSets Number of sets in the cache.
/// @tparam Associativity Number of ways per set.
template <uint32_t LineSizeBits, uint32_t NumSets, uint32_t Associativity> class Cache {
public:
  static constexpr uint32_t LINE_SIZE = 1u << LineSizeBits;
  static constexpr uint32_t LINE_MASK = LINE_SIZE - 1;
  static constexpr uint64_t TAG_SHIFT = LineSizeBits;
  static constexpr uint64_t SET_MASK = NumSets - 1;
  static constexpr uint32_t TOTAL_SIZE = LINE_SIZE * NumSets * Associativity;

  struct Allocation {
    CacheTag *tag = nullptr;
    uint8_t *data = nullptr;
  };

  Cache()
      : tags_(NumSets, Associativity, LINE_SIZE),
        data_(static_cast<size_t>(LINE_SIZE) * NumSets * Associativity, 0) {
    static_assert(util::is_power_of_2(NumSets), "NumSets must be a power of 2");
  }

  /// @brief Look up an address in the cache, counting a hit as a use.
  ///
  /// @param addr The memory address to look up.
  /// @param tag_out If non-null and hit, set to point at the matching tag.
  /// @param vmid Owning process address space (lines are tagged by (vmid, addr)).
  /// @retval true Cache hit.
  /// @retval false Cache miss.
  bool lookup(uint64_t addr, CacheTag **tag_out = nullptr, uint32_t vmid = 0) {
    const auto way = tags_.find(addr, vmid);
    if (way == tags_.npos())
      return false;
    tags_.touch(way);
    if (tag_out)
      *tag_out = &tags_.entry(way);
    return true;
  }

  /// @brief Allocate a cache line for an address, evicting the LRU victim if needed.
  ///
  /// @param addr The memory address to allocate for.
  /// @param vmid Owning process address space recorded in the new line's tag.
  /// @param evicted_tag If non-null, filled with the evicted tag (caller checks dirty,
  ///        and uses @c evicted_tag->vmid for the writeback address space).
  /// @param evicted_data If non-null, the evicted line data is copied here.
  /// @returns Pointer to the allocated tag entry.
  CacheTag *allocate(uint64_t addr, uint32_t vmid = 0, CacheTag *evicted_tag = nullptr,
                     uint8_t *evicted_data = nullptr) {
    return allocate_with_data(addr, vmid, evicted_tag, evicted_data).tag;
  }

  /// @brief Allocate a cache line and return both tag and data pointers.
  ///
  /// @details Used by cache controllers that fill a newly allocated line
  /// directly from a backing level, avoiding a second tag scan in fill_line().
  Allocation allocate_with_data(uint64_t addr, uint32_t vmid = 0, CacheTag *evicted_tag = nullptr,
                                uint8_t *evicted_data = nullptr) {
    const auto fill = tags_.fill(addr, vmid);
    if (evicted_tag)
      *evicted_tag = fill.evicted.valid ? fill.evicted : CacheTag{};
    if (evicted_data && fill.evicted.valid)
      std::memcpy(evicted_data, line_data(fill.index), LINE_SIZE);
    return {&tags_.entry(fill.index), line_data(fill.index)};
  }

  /// @brief Invalidate the cache line for an address (if present).
  /// @param addr The memory address whose cache line to invalidate.
  /// @param vmid Owning process address space.
  void invalidate(uint64_t addr, uint32_t vmid = 0) { tags_.invalidate(addr, vmid); }

  /// @brief Invalidate all cache lines for an address across all address spaces.
  /// @param addr The memory address whose cache line to invalidate.
  void invalidate_all_vmids(uint64_t addr) { tags_.invalidate_all_vmids(addr); }

  /// @brief Invalidate all cache lines.
  void invalidate_all() { tags_.invalidate_all(); }

  /// @brief Read from a cache line (must be a hit - caller ensures via lookup).
  /// @param addr The memory address identifying the cache line.
  /// @param dst Destination buffer for the read data.
  /// @param offset Byte offset within the cache line.
  /// @param size Number of bytes to read.
  void read_line(uint64_t addr, uint8_t *dst, uint32_t offset, uint32_t size,
                 uint32_t vmid = 0) const {
    const auto way = tags_.find(addr, vmid);
    assert(way != tags_.npos() && "read_line called on a miss");
    assert(offset + size <= LINE_SIZE);
    std::memcpy(dst, line_data(way) + offset, size);
  }

  /// @brief Return a const pointer to the data for a cache line, counting a
  ///        hit as a use.
  ///
  /// @param addr The memory address identifying the cache line.
  /// @returns Pointer to the line data, or nullptr if not found.
  const uint8_t *line_data_for_read(uint64_t addr, uint32_t vmid = 0) {
    return line_data_for_write(addr, vmid);
  }

  /// @brief Write to a cache line (must be a hit - caller ensures via lookup/allocate).
  /// @param addr The memory address identifying the cache line.
  /// @param src Source buffer containing data to write.
  /// @param offset Byte offset within the cache line.
  /// @param size Number of bytes to write.
  void write_line(uint64_t addr, const uint8_t *src, uint32_t offset, uint32_t size,
                  uint32_t vmid = 0) {
    const auto way = tags_.find(addr, vmid);
    assert(way != tags_.npos() && "write_line called on a miss");
    assert(offset + size <= LINE_SIZE);
    std::memcpy(line_data(way) + offset, src, size);
  }

  /// @brief Fill an entire cache line with data (used after allocate on a miss).
  /// @param addr The memory address identifying the cache line.
  /// @param data Source buffer containing a full cache line of data.
  void fill_line(uint64_t addr, const uint8_t *data, uint32_t vmid = 0) {
    const auto way = tags_.find(addr, vmid);
    assert(way != tags_.npos() && "fill_line called on a miss");
    std::memcpy(line_data(way), data, LINE_SIZE);
  }

  /// @brief Return a mutable pointer to the data for a cache line, counting a
  ///        hit as a use.
  ///
  /// Used by atomic RMW operations that need to read-modify-write in place.
  /// @param addr The memory address identifying the cache line.
  /// @returns Pointer to the line data, or nullptr if not found.
  uint8_t *line_data_for_write(uint64_t addr, uint32_t vmid = 0) {
    const auto way = tags_.find(addr, vmid);
    if (way == tags_.npos())
      return nullptr;
    tags_.touch(way);
    return line_data(way);
  }

  /// @brief Iterate over all dirty lines, calling fn(tag, line_addr, data_ptr) for each.
  /// @tparam F Callable with signature void(CacheTag&, uint64_t, uint8_t*).
  /// @param fn Callback invoked for each dirty cache line.
  template <typename F> void for_each_dirty(F &&fn) {
    tags_.for_each_valid([&](CacheTag &tag, uint64_t line_addr, size_t way) {
      if (tag.dirty)
        fn(tag, line_addr, line_data(way));
    });
  }

  /// @brief Reconstruct the line-aligned address from a tag entry and set index.
  /// @param addr The memory address.
  /// @returns Line-aligned address with offset bits cleared.
  static uint64_t line_address(uint64_t addr) { return addr & ~static_cast<uint64_t>(LINE_MASK); }

  /// @brief Return the byte offset within a cache line.
  /// @param addr The memory address.
  /// @returns Offset within the cache line.
  static uint32_t line_offset(uint64_t addr) { return static_cast<uint32_t>(addr & LINE_MASK); }

  /// @brief Return the set index for an address.
  /// @param addr The memory address.
  /// @returns Set index.
  static uint32_t set_index(uint64_t addr) {
    return static_cast<uint32_t>((addr >> LineSizeBits) & SET_MASK);
  }

  /// @brief Return the tag bits for an address.
  /// @param addr The memory address.
  /// @returns Tag bits. The same value the tag store keeps in CacheTag::tag.
  static uint64_t tag_bits(uint64_t addr) {
    return addr >> (LineSizeBits + std::countr_zero(NumSets));
  }

private:
  uint8_t *line_data(size_t way) { return &data_[way * LINE_SIZE]; }
  const uint8_t *line_data(size_t way) const { return &data_[way * LINE_SIZE]; }

  BasicTagArray<CacheTag> tags_;
  std::vector<uint8_t> data_;
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_CACHE_H_
