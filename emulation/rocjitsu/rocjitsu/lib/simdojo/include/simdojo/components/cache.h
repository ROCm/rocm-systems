// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cache.h
/// @brief Set-associative cache data structure with LRU replacement and MOESI coherence tags.

#ifndef SIMDOJO_COMPONENTS_CACHE_H_
#define SIMDOJO_COMPONENTS_CACHE_H_

#include "util/bit.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace simdojo {

/// @brief Least Recently Used (LRU) replacement policy for a set-associative cache.
///
/// @details Maintains per-set recency order using a compact array of way indices.
/// access() promotes a way to Most Recently Used (MRU) position; victim()
/// returns the LRU way for eviction.
///
/// @tparam NumSets Number of sets in the cache (must be a power of 2).
/// @tparam Associativity Number of ways per set.
template <uint32_t NumSets, uint32_t Associativity> class LRUPolicy {
public:
  LRUPolicy() : order_(NumSets * Associativity) {
    for (uint32_t s = 0; s < NumSets; ++s)
      for (uint32_t w = 0; w < Associativity; ++w)
        order_[s * Associativity + w] = w;
  }

  /// @brief Promote a way to MRU position within a set.
  /// @param set Set index.
  /// @param way Way index within the set.
  void access(uint32_t set, uint32_t way) {
    uint32_t *o = &order_[set * Associativity];
    uint32_t pos = 0;
    for (; pos < Associativity; ++pos)
      if (o[pos] == way)
        break;
    for (uint32_t i = pos; i + 1 < Associativity; ++i)
      o[i] = o[i + 1];
    o[Associativity - 1] = way;
  }

  /// @brief Return the LRU way index for a set (eviction candidate).
  /// @param set Set index.
  /// @returns Way index of the least recently used entry.
  uint32_t victim(uint32_t set) const { return order_[set * Associativity]; }

private:
  std::vector<uint32_t> order_;
};

/// @brief Coherence state for a cache line (MOESI protocol).
enum class CoherenceState : uint8_t {
  INVALID,
  SHARED,
  EXCLUSIVE,
  MODIFIED,
  OWNED,
};

/// @brief Tag and metadata for a single cache line.
struct CacheTag {
  uint64_t tag = 0;
  bool valid = false;
  bool dirty = false;
  CoherenceState coherence = CoherenceState::INVALID;
};

/// @brief Set-associative cache data structure with configurable geometry and replacement policy.
///
/// @details Pure data structure, not a simulation Component. Cache controllers wrap
/// this to implement protocol-specific behavior (write-back, write-through,
/// coherence transitions).
///
/// @tparam LineSizeBits Log2 of the cache line size in bytes.
/// @tparam NumSets Number of sets in the cache.
/// @tparam Associativity Number of ways per set.
/// @tparam Policy Replacement policy (default: LRU).
template <uint32_t LineSizeBits, uint32_t NumSets, uint32_t Associativity,
          typename Policy = LRUPolicy<NumSets, Associativity>>
class Cache {
public:
  static constexpr uint32_t LINE_SIZE = 1u << LineSizeBits;
  static constexpr uint32_t LINE_MASK = LINE_SIZE - 1;
  static constexpr uint64_t TAG_SHIFT = LineSizeBits;
  static constexpr uint64_t SET_MASK = NumSets - 1;
  static constexpr uint32_t TOTAL_SIZE = LINE_SIZE * NumSets * Associativity;

  Cache()
      : tags_(static_cast<size_t>(NumSets) * Associativity),
        data_(static_cast<size_t>(LINE_SIZE) * NumSets * Associativity, 0) {
    static_assert(util::is_power_of_2(NumSets), "NumSets must be a power of 2");
  }

  /// @brief Look up an address in the cache.
  ///
  /// @param addr The memory address to look up.
  /// @param tag_out If non-null and hit, set to point at the matching tag.
  /// @retval true Cache hit.
  /// @retval false Cache miss.
  bool lookup(uint64_t addr, CacheTag **tag_out = nullptr) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag) {
        policy_.access(set, w);
        if (tag_out)
          *tag_out = &t;
        return true;
      }
    }
    return false;
  }

  /// @brief Allocate a cache line for an address, evicting the LRU victim if needed.
  ///
  /// @param addr The memory address to allocate for.
  /// @param evicted_tag If non-null, filled with the evicted tag (caller checks dirty).
  /// @param evicted_data If non-null, the evicted line data is copied here.
  /// @returns Pointer to the allocated tag entry.
  CacheTag *allocate(uint64_t addr, CacheTag *evicted_tag = nullptr,
                     uint8_t *evicted_data = nullptr) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);

    // Check for an invalid way first.
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (!t.valid) {
        t.tag = tag;
        t.valid = true;
        t.dirty = false;
        t.coherence = CoherenceState::INVALID;
        policy_.access(set, w);
        return &t;
      }
    }

    // Evict LRU victim.
    uint32_t victim_way = policy_.victim(set);
    auto &vt = tag_at(set, victim_way);
    if (evicted_tag)
      *evicted_tag = vt;
    if (evicted_data)
      std::memcpy(evicted_data, line_data(set, victim_way), LINE_SIZE);

    vt.tag = tag;
    vt.valid = true;
    vt.dirty = false;
    vt.coherence = CoherenceState::INVALID;
    policy_.access(set, victim_way);
    return &vt;
  }

  /// @brief Invalidate the cache line for an address (if present).
  /// @param addr The memory address whose cache line to invalidate.
  void invalidate(uint64_t addr) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag) {
        t.valid = false;
        t.dirty = false;
        t.coherence = CoherenceState::INVALID;
        return;
      }
    }
  }

  /// @brief Invalidate all cache lines.
  void invalidate_all() {
    for (auto &t : tags_) {
      t.valid = false;
      t.dirty = false;
      t.coherence = CoherenceState::INVALID;
    }
  }

  /// @brief Read from a cache line (must be a hit - caller ensures via lookup).
  /// @param addr The memory address identifying the cache line.
  /// @param dst Destination buffer for the read data.
  /// @param offset Byte offset within the cache line.
  /// @param size Number of bytes to read.
  void read_line(uint64_t addr, uint8_t *dst, uint32_t offset, uint32_t size) const {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      const auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag) {
        assert(offset + size <= LINE_SIZE);
        std::memcpy(dst, line_data(set, w) + offset, size);
        return;
      }
    }
    assert(false && "read_line called on a miss");
  }

  /// @brief Write to a cache line (must be a hit - caller ensures via lookup/allocate).
  /// @param addr The memory address identifying the cache line.
  /// @param src Source buffer containing data to write.
  /// @param offset Byte offset within the cache line.
  /// @param size Number of bytes to write.
  void write_line(uint64_t addr, const uint8_t *src, uint32_t offset, uint32_t size) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag) {
        assert(offset + size <= LINE_SIZE);
        std::memcpy(line_data(set, w) + offset, src, size);
        return;
      }
    }
    assert(false && "write_line called on a miss");
  }

  /// @brief Fill an entire cache line with data (used after allocate on a miss).
  /// @param addr The memory address identifying the cache line.
  /// @param data Source buffer containing a full cache line of data.
  void fill_line(uint64_t addr, const uint8_t *data) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag) {
        std::memcpy(line_data(set, w), data, LINE_SIZE);
        return;
      }
    }
    assert(false && "fill_line called on a miss");
  }

  /// @brief Return a mutable pointer to the data for a cache line (must be a hit).
  ///
  /// Used by atomic RMW operations that need to read-modify-write in place.
  /// @param addr The memory address identifying the cache line.
  /// @returns Pointer to the line data, or nullptr if not found.
  uint8_t *line_data_for_write(uint64_t addr) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag) {
        policy_.access(set, w);
        return line_data(set, w);
      }
    }
    return nullptr;
  }

  /// @brief Iterate over all dirty lines, calling fn(tag, line_addr, data_ptr) for each.
  /// @tparam F Callable with signature void(CacheTag&, uint64_t, uint8_t*).
  /// @param fn Callback invoked for each dirty cache line.
  template <typename F> void for_each_dirty(F &&fn) {
    for (uint32_t s = 0; s < NumSets; ++s)
      for (uint32_t w = 0; w < Associativity; ++w) {
        auto &t = tag_at(s, w);
        if (t.valid && t.dirty) {
          uint64_t line_addr =
              (t.tag << (LineSizeBits + log2_sets())) | (static_cast<uint64_t>(s) << LineSizeBits);
          fn(t, line_addr, line_data(s, w));
        }
      }
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
  /// @returns Tag bits.
  static uint64_t tag_bits(uint64_t addr) { return addr >> (LineSizeBits + log2_sets()); }

private:
  static constexpr uint32_t log2_sets() {
    uint32_t n = NumSets, bits = 0;
    while (n > 1) {
      n >>= 1;
      ++bits;
    }
    return bits;
  }

  CacheTag &tag_at(uint32_t set, uint32_t way) {
    return tags_[static_cast<size_t>(set) * Associativity + way];
  }

  const CacheTag &tag_at(uint32_t set, uint32_t way) const {
    return tags_[static_cast<size_t>(set) * Associativity + way];
  }

  uint8_t *line_data(uint32_t set, uint32_t way) {
    return &data_[(static_cast<size_t>(set) * Associativity + way) * LINE_SIZE];
  }

  const uint8_t *line_data(uint32_t set, uint32_t way) const {
    return &data_[(static_cast<size_t>(set) * Associativity + way) * LINE_SIZE];
  }

  std::vector<CacheTag> tags_;
  std::vector<uint8_t> data_;
  Policy policy_;
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_CACHE_H_
