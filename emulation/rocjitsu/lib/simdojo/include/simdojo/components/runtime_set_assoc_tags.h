// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file runtime_set_assoc_tags.h
/// @brief Runtime-sized, dataless set-associative tag store with LRU replacement.

#ifndef SIMDOJO_COMPONENTS_RUNTIME_SET_ASSOC_TAGS_H_
#define SIMDOJO_COMPONENTS_RUNTIME_SET_ASSOC_TAGS_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace simdojo {

/// @brief Runtime-configured set-associative tag array with LRU replacement.
///
/// @details Unlike Cache<>, this is runtime-sized (geometry from a config rather
/// than template parameters), carries NO line data and NO coherence state, and uses
/// modulo set-indexing (sets need not be a power of two). It is a pure
/// residency/eviction mechanism: callers layer their own per-slot side state (e.g.
/// a timing model's fill-readiness cycle) keyed by the stable slot index returned
/// here. Slot index = set * ways + way, stable for the lifetime of a configuration.
///
/// LRU contract (matches the cycle-model's prior hand-rolled cache exactly):
///   - lookup() and present() do NOT touch LRU; the hit path promotes explicitly
///     via touch(). (A caller that probes without intending to promote -- e.g. a
///     deferred fill resolution -- must see LRU order unchanged.)
///   - install() sets the tag valid and promotes the slot to MRU.
///   - victim() returns the lowest-index invalid way, else the true-LRU way; it
///     neither installs nor touches.
class RuntimeSetAssocTags {
public:
  /// @brief (Re)configure geometry. sets/ways/line_bytes == 0 => unconfigured.
  void configure(uint32_t sets, uint32_t ways, uint32_t line_bytes) {
    sets_ = sets; ways_ = ways; line_bytes_ = line_bytes; lru_tick_ = 0;
    if (!sets_ || !ways_ || !line_bytes_) { sets_ = ways_ = line_bytes_ = 0; tags_.clear(); lru_.clear(); return; }
    tags_.assign(static_cast<size_t>(sets_) * ways_, Tag{});
    lru_.assign(static_cast<size_t>(sets_) * ways_, 0);
  }

  bool configured() const { return sets_ != 0; }
  size_t slot_count() const { return static_cast<size_t>(sets_) * ways_; }

  /// @brief Look up a line. Hit: slot index (no LRU touch). Miss: -1.
  int lookup(uint64_t line_base) const {
    if (!sets_) return -1;
    uint32_t s = set_of(line_base);
    for (uint32_t w = 0; w < ways_; ++w) {
      size_t slot = static_cast<size_t>(s) * ways_ + w;
      if (tags_[slot].valid && tags_[slot].tag == line_base) return static_cast<int>(slot);
    }
    return -1;
  }

  /// @brief Const residency probe (no LRU touch).
  bool present(uint64_t line_base) const { return lookup(line_base) >= 0; }

  /// @brief Eviction candidate slot: lowest-index invalid way, else LRU way. No install/touch.
  uint32_t victim(uint64_t line_base) const {
    assert(configured());
    uint32_t s = set_of(line_base);
    size_t base = static_cast<size_t>(s) * ways_;
    size_t best = base;
    for (uint32_t w = 0; w < ways_; ++w) {
      size_t slot = base + w;
      if (!tags_[slot].valid) return static_cast<uint32_t>(slot);
      if (lru_[slot] < lru_[best]) best = slot;
    }
    return static_cast<uint32_t>(best);
  }

  /// @brief Install a tag at a slot (valid=true) and promote it to MRU.
  void install(uint32_t slot, uint64_t line_base) {
    tags_[slot].tag = line_base; tags_[slot].valid = true; lru_[slot] = ++lru_tick_;
  }

  /// @brief Promote a slot to MRU (called by the hit path).
  void touch(uint32_t slot) { lru_[slot] = ++lru_tick_; }

  /// @brief Invalidate all lines and reset LRU.
  void reset() {
    for (auto& t : tags_) t = Tag{};
    std::fill(lru_.begin(), lru_.end(), 0);
    lru_tick_ = 0;
  }

private:
  struct Tag { uint64_t tag = 0; bool valid = false; };
  uint32_t set_of(uint64_t line_base) const {
    return static_cast<uint32_t>((line_base / line_bytes_) % sets_);
  }

  std::vector<Tag>      tags_;     // sets_*ways_, slot = set*ways_+way
  std::vector<uint64_t> lru_;      // monotonic recency stamp per slot
  uint64_t              lru_tick_ = 0;
  uint32_t              sets_ = 0, ways_ = 0, line_bytes_ = 0;
};

}  // namespace simdojo

#endif  // SIMDOJO_COMPONENTS_RUNTIME_SET_ASSOC_TAGS_H_
