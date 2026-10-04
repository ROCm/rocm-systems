// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cache.h
/// @brief Set-associative cache data structure with LRU replacement and MOESI coherence tags.

#ifndef SIMDOJO_COMPONENTS_CACHE_H_
#define SIMDOJO_COMPONENTS_CACHE_H_

#include "util/bit.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
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
  LRUPolicy() : order_(static_cast<size_t>(NumSets) * Associativity) {
    for (uint32_t s = 0; s < NumSets; ++s)
      for (uint32_t w = 0; w < Associativity; ++w)
        order_[static_cast<size_t>(s) * Associativity + w] = static_cast<WayIndex>(w);
  }

  /// @brief Promote a way to MRU position within a set.
  /// @param set Set index.
  /// @param way Way index within the set.
  void access(uint32_t set, uint32_t way) {
    WayIndex *o = &order_[static_cast<size_t>(set) * Associativity];
    uint32_t pos = 0;
    for (; pos < Associativity; ++pos)
      if (o[pos] == way)
        break;
    for (uint32_t i = pos; i + 1 < Associativity; ++i)
      o[i] = o[i + 1];
    o[Associativity - 1] = static_cast<WayIndex>(way);
  }

  /// @brief Return the LRU way index for a set (eviction candidate).
  /// @param set Set index.
  /// @returns Way index of the least recently used entry.
  uint32_t victim(uint32_t set) const { return order_[static_cast<size_t>(set) * Associativity]; }

private:
  // Compact indices reduce construction traffic; larger ways retain 32-bit indices.
  using WayIndex = std::conditional_t<(Associativity <= std::numeric_limits<uint8_t>::max() + 1u),
                                      uint8_t, uint32_t>;
  std::vector<WayIndex> order_;
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
///
/// @details @c vmid identifies the owning process address space. Lines with the
/// same line address but different vmids are distinct entries, because guest VAs
/// are per-process and may alias across processes.
struct CacheTag {
  uint64_t tag = 0;
  uint64_t coherence_epoch = 0; ///< Controller-defined lazy-invalidation generation.
  /// Byte-validity mask for partial-line fills. Current cache geometries use
  /// at most 128-byte lines, so two words cover every byte.
  std::array<uint64_t, 2> valid_bytes = {~uint64_t{0}, ~uint64_t{0}};
  uint32_t vmid = 0;
  bool valid = false;
  bool dirty = false;
  CoherenceState coherence = CoherenceState::INVALID;
};

/// @brief Set-associative cache data structure with configurable geometry and replacement policy.
///
/// @details Pure data structure, not a simulation Component. Cache controllers wrap
/// this to implement protocol-specific behavior (write-back, write-through,
/// coherence transitions).
/// Only allocation methods may make a line valid. Controllers may update dirty
/// and coherence metadata on resident lines returned by the cache.
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
  static constexpr size_t TOTAL_SIZE = static_cast<size_t>(LINE_SIZE) * NumSets * Associativity;

  struct Allocation {
    CacheTag *tag = nullptr;
    uint8_t *data = nullptr;
  };

  Cache() : touched_sets_(NumSets, 0) {
    static_assert(util::is_power_of_2(NumSets), "NumSets must be a power of 2");
    static_assert(LINE_SIZE <= 128, "Cache byte-validity mask supports lines up to 128 bytes");
  }

  /// @brief Return whether every byte in one line-relative range is present.
  static bool bytes_valid(const CacheTag &tag, uint32_t offset, uint32_t size) {
    assert(offset <= LINE_SIZE && size <= LINE_SIZE - offset);
    for (uint32_t byte = offset; byte < offset + size; ++byte)
      if ((tag.valid_bytes[byte / 64] & (uint64_t{1} << (byte % 64))) == 0)
        return false;
    return true;
  }

  /// @brief Mark every byte of a newly allocated line absent.
  static void clear_valid_bytes(CacheTag &tag) { tag.valid_bytes = {}; }

  /// @brief Mark one line-relative byte range present.
  static void mark_valid_bytes(CacheTag &tag, uint32_t offset, uint32_t size) {
    assert(offset <= LINE_SIZE && size <= LINE_SIZE - offset);
    for (uint32_t byte = offset; byte < offset + size; ++byte)
      tag.valid_bytes[byte / 64] |= uint64_t{1} << (byte % 64);
  }

  /// @brief Look up an address in the cache.
  ///
  /// @param addr The memory address to look up.
  /// @param tag_out If non-null and hit, set to point at the matching tag.
  /// @param vmid Owning process address space (lines are tagged by (vmid, addr)).
  /// @retval true Cache hit.
  /// @retval false Cache miss.
  bool lookup(uint64_t addr, CacheTag **tag_out = nullptr, uint32_t vmid = 0) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
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
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);

    // Check for an invalid way first.
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (!t.valid) {
        // Adjacent markers can share a host cache line. Write only on first use.
        if (!touched_sets_[set])
          touched_sets_[set] = 1;
        t.tag = tag;
        t.coherence_epoch = 0;
        t.valid_bytes = {~uint64_t{0}, ~uint64_t{0}};
        t.vmid = vmid;
        t.valid = true;
        t.dirty = false;
        t.coherence = CoherenceState::INVALID;
        policy_.access(set, w);
        if (evicted_tag)
          *evicted_tag = {};
        return {&t, line_data(set, w)};
      }
    }

    // A full set is already marked, so eviction need not write its shared marker.
    uint32_t victim_way = policy_.victim(set);
    auto &vt = tag_at(set, victim_way);
    if (evicted_tag)
      *evicted_tag = vt;
    if (evicted_data)
      std::memcpy(evicted_data, line_data(set, victim_way), LINE_SIZE);

    vt.tag = tag;
    vt.coherence_epoch = 0;
    vt.valid_bytes = {~uint64_t{0}, ~uint64_t{0}};
    vt.vmid = vmid;
    vt.valid = true;
    vt.dirty = false;
    vt.coherence = CoherenceState::INVALID;
    policy_.access(set, victim_way);
    return {&vt, line_data(set, victim_way)};
  }

  /// @brief Invalidate the cache line for an address (if present).
  /// @param addr The memory address whose cache line to invalidate.
  /// @param vmid Owning process address space.
  void invalidate(uint64_t addr, uint32_t vmid = 0) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
        t.coherence_epoch = 0;
        t.valid = false;
        t.dirty = false;
        t.coherence = CoherenceState::INVALID;
        return;
      }
    }
  }

  /// @brief Invalidate all cache lines for an address across all address spaces.
  /// @param addr The memory address whose cache line to invalidate.
  void invalidate_all_vmids(uint64_t addr) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag) {
        t.coherence_epoch = 0;
        t.valid = false;
        t.dirty = false;
        t.coherence = CoherenceState::INVALID;
      }
    }
  }

  /// @brief Invalidate all cache lines.
  void invalidate_all() {
    for (uint32_t s = 0; s < NumSets; ++s) {
      if (!touched_sets_[s])
        continue;
      for (uint32_t w = 0; w < Associativity; ++w) {
        auto &t = tag_at(s, w);
        t.coherence_epoch = 0;
        t.valid = false;
        t.dirty = false;
        t.coherence = CoherenceState::INVALID;
      }
      touched_sets_[s] = 0;
    }
  }

  /// @brief Read from a cache line (must be a hit - caller ensures via lookup).
  /// @param addr The memory address identifying the cache line.
  /// @param dst Destination buffer for the read data.
  /// @param offset Byte offset within the cache line.
  /// @param size Number of bytes to read.
  void read_line(uint64_t addr, uint8_t *dst, uint32_t offset, uint32_t size,
                 uint32_t vmid = 0) const {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      const auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
        assert(offset + size <= LINE_SIZE);
        assert(bytes_valid(t, offset, size) && "read_line called for absent bytes");
        std::memcpy(dst, line_data(set, w) + offset, size);
        return;
      }
    }
    assert(false && "read_line called on a miss");
  }

  /// @brief Return a const pointer to the data for a cache line.
  ///
  /// @param addr The memory address identifying the cache line.
  /// @returns Pointer to the line data, or nullptr if not found.
  const uint8_t *line_data_for_read(uint64_t addr, uint32_t vmid = 0) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
        policy_.access(set, w);
        return line_data(set, w);
      }
    }
    return nullptr;
  }

  /// @brief Write to a cache line (must be a hit - caller ensures via lookup/allocate).
  /// @param addr The memory address identifying the cache line.
  /// @param src Source buffer containing data to write.
  /// @param offset Byte offset within the cache line.
  /// @param size Number of bytes to write.
  void write_line(uint64_t addr, const uint8_t *src, uint32_t offset, uint32_t size,
                  uint32_t vmid = 0) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
        assert(offset + size <= LINE_SIZE);
        std::memcpy(line_data(set, w) + offset, src, size);
        mark_valid_bytes(t, offset, size);
        return;
      }
    }
    assert(false && "write_line called on a miss");
  }

  /// @brief Fill an entire cache line with data (used after allocate on a miss).
  /// @param addr The memory address identifying the cache line.
  /// @param data Source buffer containing a full cache line of data.
  void fill_line(uint64_t addr, const uint8_t *data, uint32_t vmid = 0) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
        std::memcpy(line_data(set, w), data, LINE_SIZE);
        t.valid_bytes = {~uint64_t{0}, ~uint64_t{0}};
        return;
      }
    }
    assert(false && "fill_line called on a miss");
  }

  /// @brief Fill only bytes that are absent from a resident partial line.
  /// @details Preserves bytes already supplied by stores or earlier partial
  /// reads while completing the line from a backing-level snapshot.
  void fill_missing_bytes(uint64_t addr, const uint8_t *data, uint32_t vmid = 0) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
        uint8_t *line = line_data(set, w);
        for (uint32_t byte = 0; byte < LINE_SIZE; ++byte)
          if (!bytes_valid(t, byte, 1))
            line[byte] = data[byte];
        t.valid_bytes = {~uint64_t{0}, ~uint64_t{0}};
        return;
      }
    }
    assert(false && "fill_missing_bytes called on a miss");
  }

  /// @brief Return a mutable pointer to the data for a cache line (must be a hit).
  ///
  /// Used by atomic RMW operations that need to read-modify-write in place.
  /// @param addr The memory address identifying the cache line.
  /// @returns Pointer to the line data, or nullptr if not found.
  uint8_t *line_data_for_write(uint64_t addr, uint32_t vmid = 0) {
    uint32_t set = set_index(addr);
    uint64_t tag = tag_bits(addr);
    for (uint32_t w = 0; w < Associativity; ++w) {
      auto &t = tag_at(set, w);
      if (t.valid && t.tag == tag && t.vmid == vmid) {
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
    for (uint32_t s = 0; s < NumSets; ++s) {
      if (!touched_sets_[s])
        continue;
      for (uint32_t w = 0; w < Associativity; ++w) {
        auto &t = tag_at(s, w);
        if (t.valid && t.dirty) {
          uint64_t line_addr =
              (t.tag << (LineSizeBits + log2_sets())) | (static_cast<uint64_t>(s) << LineSizeBits);
          fn(t, line_addr, line_data(s, w));
        }
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
  // Zero bytes represent unused data and invalid tags. Allocation initializes
  // valid_bytes before making a tag valid, so its nonzero default is unnecessary
  // for unused tags. Keep calloc storage restricted to these types.
  template <typename T, size_t Count> class ZeroStorage {
  public:
    static_assert(std::is_trivially_copyable_v<T>, "Cache storage requires trivial copies");
    static_assert(std::is_trivially_destructible_v<T>,
                  "Cache storage requires trivial destruction");
    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, CacheTag>,
                  "Cache storage only supports byte data and CacheTag");
    // Bind every field so additions require checking the unused-tag representation.
    static_assert(
        [] {
          const auto [tag, coherence_epoch, valid_bytes, vmid, valid, dirty, coherence] =
              CacheTag{};
          return tag == 0 && coherence_epoch == 0 && vmid == 0 && !valid && !dirty &&
                 static_cast<uint8_t>(coherence) == 0 &&
                 valid_bytes == std::array<uint64_t, 2>{~uint64_t{0}, ~uint64_t{0}};
        }(),
        "CacheTag fields must permit zero-initialized invalid entries");

    ZeroStorage() : elements_(allocate()) {}
    ZeroStorage(const ZeroStorage &other) : elements_(other.elements_ ? allocate() : nullptr) {
      if (elements_)
        std::memcpy(elements_.get(), other.elements_.get(), Count * sizeof(T));
    }
    ZeroStorage &operator=(const ZeroStorage &other) {
      if (this != &other) {
        if (other.elements_) {
          if (!elements_)
            elements_.reset(allocate());
          std::memcpy(elements_.get(), other.elements_.get(), Count * sizeof(T));
        } else {
          elements_.reset();
        }
      }
      return *this;
    }
    ZeroStorage(ZeroStorage &&) noexcept = default;
    ZeroStorage &operator=(ZeroStorage &&) noexcept = default;

    T &operator[](size_t index) { return elements_[index]; }
    const T &operator[](size_t index) const { return elements_[index]; }

  private:
    class Free {
    public:
      void operator()(T *ptr) const { std::free(ptr); }
    };
    static T *allocate() {
      while (true) {
        auto *ptr = static_cast<T *>(std::calloc(Count, sizeof(T)));
        if (ptr || Count == 0)
          return ptr;
        std::new_handler handler = std::get_new_handler();
        if (!handler)
          throw std::bad_alloc();
        handler();
      }
    }
    std::unique_ptr<T[], Free> elements_;
  };

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

  ZeroStorage<CacheTag, static_cast<size_t>(NumSets) * Associativity> tags_;
  ZeroStorage<uint8_t, TOTAL_SIZE> data_;
  Policy policy_;
  // Keep this allocation after existing storage to preserve its allocator layout.
  // Track sets allocated since full invalidation, even if their individual lines
  // have since been invalidated. Full invalidation clears each marker after
  // visiting every way. Separate bytes preserve the existing contract allowing
  // different sets to be accessed concurrently under controller-provided set locks.
  // Full-cache operations still require exclusive maintenance access.
  std::vector<uint8_t> touched_sets_;
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_CACHE_H_
