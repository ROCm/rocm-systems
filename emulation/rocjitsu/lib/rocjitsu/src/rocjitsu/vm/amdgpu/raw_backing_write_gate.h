// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file raw_backing_write_gate.h
/// @brief Serialize raw backing writes with GPU cache admission.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <utility>

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
        shard.active.erase(id_);
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
        gate_->closed_.erase(id_);
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
    for (const Entry &closed : closed_.entries())
      if (overlaps(requested, closed.span))
        return std::nullopt;
    const uint64_t id = shard.next_id;
    if (!shard.active.append({requested, id}))
      return std::nullopt;
    ++shard.next_id;
    return Permit(this, index, id);
  }

  [[nodiscard]] std::optional<Closed> try_close(uint64_t begin, uint64_t size) noexcept {
    if (!valid(begin, size))
      return std::nullopt;
    auto locks = lock_all_shards();
    if (next_id_ == 0)
      return std::nullopt;
    const Span requested{begin, size};
    for (const auto &shard : shards_)
      for (const Entry &active : shard.active.entries())
        if (overlaps(requested, active.span))
          return std::nullopt;
    const uint64_t id = next_id_;
    if (!closed_.append({requested, id}))
      return std::nullopt;
    ++next_id_;
    return Closed(this, id);
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

  // Admission failure must leave existing tokens intact, including when
  // growing their bookkeeping runs out of memory.
  class EntryStorage {
  public:
    bool append(Entry entry) noexcept {
      if (size_ == capacity_) {
        constexpr size_t max_capacity = std::numeric_limits<std::ptrdiff_t>::max() / sizeof(Entry);
        if (capacity_ == max_capacity)
          return false;
        const size_t capacity =
            capacity_ == 0 ? 8 : (capacity_ > max_capacity / 2 ? max_capacity : capacity_ * 2);
        std::unique_ptr<Entry[]> storage(new (std::nothrow) Entry[capacity]);
        if (!storage)
          return false;
        if (size_ != 0)
          std::copy_n(storage_.get(), size_, storage.get());
        storage_ = std::move(storage);
        capacity_ = capacity;
      }
      storage_[size_++] = entry;
      return true;
    }

    std::span<const Entry> entries() const noexcept { return {storage_.get(), size_}; }

    void erase(uint64_t id) noexcept {
      for (size_t index = 0; index < size_; ++index) {
        if (storage_[index].id == id) {
          storage_[index] = storage_[--size_];
          return;
        }
      }
      assert(false && "raw backing gate token is no longer registered");
    }

  private:
    std::unique_ptr<Entry[]> storage_;
    size_t size_ = 0;
    size_t capacity_ = 0;
  };

  // Raw writes on independent CU workers only touch their own shard. Closing
  // a controlled range takes every shard in order so the overlap check and
  // publication are atomic with respect to all outstanding permits.
  // A large CU pool should not share its ordinary write locks. Under TSan,
  // leave space in the held-lock tracker for the caller's other locks.
#if defined(__SANITIZE_THREAD__)
  static constexpr size_t kShards = 16;
#elif defined(__has_feature)
  static constexpr size_t kShards = __has_feature(thread_sanitizer) ? 16 : 64;
#else
  static constexpr size_t kShards = 64;
#endif
  struct alignas(64) Shard {
    std::mutex mutex;
    EntryStorage active;
    uint64_t next_id = 1;
  };
  std::array<std::unique_lock<std::mutex>, kShards> lock_all_shards() {
    std::array<std::unique_lock<std::mutex>, kShards> locks;
    for (size_t i = 0; i < kShards; ++i)
      locks[i] = std::unique_lock(shards_[i].mutex);
    return locks;
  }

  std::array<Shard, kShards> shards_;
  EntryStorage closed_;
  uint64_t next_id_ = 1;
};

} // namespace rocjitsu::amdgpu
