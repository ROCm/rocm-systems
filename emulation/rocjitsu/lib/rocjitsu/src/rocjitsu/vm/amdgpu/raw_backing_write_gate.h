// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file raw_backing_write_gate.h
/// @brief Serialize raw backing writes with GPU cache admission.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Quiesce raw backing effects for one VA span before publishing a
/// controlled grant.
/// @details A permit spans admission and the backing effect without holding a
/// page-table or allocation lock across callbacks that can reenter KFD.
class VmRawBackingWriteGate {
public:
  class Permit {
  public:
    Permit() = default;
    Permit(Permit &&other) noexcept
        : gate_(std::exchange(other.gate_, nullptr)), shard_(other.shard_), id_(other.id_) {}
    Permit &operator=(Permit &&other) noexcept {
      if (this != &other) {
        release();
        gate_ = std::exchange(other.gate_, nullptr);
        shard_ = other.shard_;
        id_ = other.id_;
      }
      return *this;
    }
    Permit(const Permit &) = delete;
    Permit &operator=(const Permit &) = delete;
    ~Permit() { release(); }

  private:
    friend class VmRawBackingWriteGate;
    Permit(VmRawBackingWriteGate *gate, size_t shard, uint64_t id)
        : gate_(gate), shard_(shard), id_(id) {}
    void release() noexcept {
      if (gate_) {
        auto &shard = gate_->shards_[shard_];
        std::lock_guard lock(shard.mutex);
        erase_entry(shard.active, id_);
      }
      gate_ = nullptr;
    }
    VmRawBackingWriteGate *gate_ = nullptr;
    size_t shard_ = 0;
    uint64_t id_ = 0;
  };

  class Closed {
  public:
    Closed(Closed &&other) noexcept : gate_(std::exchange(other.gate_, nullptr)), id_(other.id_) {}
    Closed &operator=(Closed &&other) noexcept {
      if (this != &other) {
        reopen();
        gate_ = std::exchange(other.gate_, nullptr);
        id_ = other.id_;
      }
      return *this;
    }
    Closed(const Closed &) = delete;
    Closed &operator=(const Closed &) = delete;
    ~Closed() { reopen(); }

  private:
    friend class VmRawBackingWriteGate;
    Closed(VmRawBackingWriteGate *gate, uint64_t id) : gate_(gate), id_(id) {}
    void reopen() noexcept {
      if (gate_) {
        auto locks = gate_->lock_all_shards();
        erase_entry(gate_->closed_, id_);
      }
      gate_ = nullptr;
    }
    VmRawBackingWriteGate *gate_ = nullptr;
    uint64_t id_ = 0;
  };

  [[nodiscard]] std::optional<Permit> try_acquire(uint64_t begin, uint64_t size) noexcept {
    if (!valid(begin, size))
      return std::nullopt;
    static std::atomic<size_t> next_shard{0};
    static thread_local const size_t index =
        next_shard.fetch_add(1, std::memory_order_relaxed) % kShards;
    auto &shard = shards_[index];
    std::lock_guard lock(shard.mutex);
    if (shard.next_id == 0)
      return std::nullopt;
    const Span requested{begin, size};
    for (const Entry &closed : closed_)
      if (overlaps(requested, closed.span))
        return std::nullopt;
    try {
      const uint64_t id = shard.next_id;
      shard.active.push_back({requested, id});
      ++shard.next_id;
      return Permit(this, index, id);
    } catch (const std::bad_alloc &) {
      return std::nullopt;
    }
  }

  [[nodiscard]] std::optional<Closed> try_close(uint64_t begin, uint64_t size) noexcept {
    if (!valid(begin, size))
      return std::nullopt;
    auto locks = lock_all_shards();
    if (next_id_ == 0)
      return std::nullopt;
    const Span requested{begin, size};
    for (const auto &shard : shards_)
      for (const Entry &active : shard.active)
        if (overlaps(requested, active.span))
          return std::nullopt;
    try {
      const uint64_t id = next_id_;
      closed_.push_back({requested, id});
      ++next_id_;
      return Closed(this, id);
    } catch (const std::bad_alloc &) {
      return std::nullopt;
    }
  }

private:
  struct Span {
    uint64_t begin;
    uint64_t size;
  };

  struct Entry {
    Span span;
    uint64_t id;
  };

  static bool valid(uint64_t begin, uint64_t size) noexcept {
    return size != 0 && size - 1 <= std::numeric_limits<uint64_t>::max() - begin;
  }

  static bool overlaps(const Span &left, const Span &right) noexcept {
    return left.begin <= right.begin ? right.begin - left.begin < left.size
                                     : left.begin - right.begin < right.size;
  }

  static void erase_entry(std::vector<Entry> &entries, uint64_t id) noexcept {
    const auto found = std::find_if(entries.begin(), entries.end(),
                                    [id](const Entry &entry) { return entry.id == id; });
    assert(found != entries.end());
    if (found != entries.end()) {
      *found = entries.back();
      entries.pop_back();
    }
  }

  // Raw writes on independent CU workers only touch their own shard. Closing
  // a controlled range takes every shard in order so the overlap check and
  // publication are atomic with respect to all outstanding permits.
  static constexpr size_t kShards = 16;
  struct alignas(64) Shard {
    std::mutex mutex;
    std::vector<Entry> active;
    uint64_t next_id = 1;
  };
  std::array<std::unique_lock<std::mutex>, kShards> lock_all_shards() {
    std::array<std::unique_lock<std::mutex>, kShards> locks;
    for (size_t i = 0; i < kShards; ++i)
      locks[i] = std::unique_lock(shards_[i].mutex);
    return locks;
  }

  std::array<Shard, kShards> shards_;
  std::vector<Entry> closed_;
  uint64_t next_id_ = 1;
};

} // namespace rocjitsu::amdgpu
