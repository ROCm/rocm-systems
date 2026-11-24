// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file backing_line_versions.h
/// @brief Track writes to shared storage while a cache fill is pending.

#pragma once

#include "simdojo/components/backing_store_id.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simdojo {

/// Records the versions of the storage lines touched by a completed read.
/// A cache can fill from the reply while those versions still match. A write
/// through any alias of the same backing ID and line offset stales the stamp.
/// Eviction or a maintenance clear also rejects the old stamp if the same
/// line later enters the table again.
class BackingLineVersions {
  struct LineKey {
    BackingStoreId backing_id;
    uint64_t line_offset;

    bool operator==(const LineKey &) const = default;
  };

  struct LineKeyHash {
    size_t operator()(const LineKey &key) const noexcept {
      const size_t a = std::hash<uint64_t>{}(key.backing_id);
      const size_t b = std::hash<uint64_t>{}(key.line_offset);
      return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    }
  };

  static bool bounded_lines(uint64_t line_bytes, BackingStoreId backing_id, uint64_t offset,
                            size_t size, uint64_t &first, uint64_t &last) noexcept {
    if (backing_id == 0 || size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - offset) {
      return false;
    }
    first = offset & ~(line_bytes - 1);
    last = (offset + size - 1) & ~(line_bytes - 1);
    return true;
  }

  class State {
  public:
    State(size_t line_bytes, size_t capacity)
        : line_bytes(line_bytes), capacity(capacity),
          disabled(!std::has_single_bit(line_bytes) || capacity == 0) {}

    void invalidate_all_locked() noexcept {
      versions.clear();
      if (epoch == std::numeric_limits<uint64_t>::max()) {
        disabled = true;
      } else {
        ++epoch;
      }
    }

    std::shared_mutex mutex;
    std::unordered_map<LineKey, uint64_t, LineKeyHash> versions;
    const size_t line_bytes;
    const size_t capacity;
    uint64_t next_version = 1;
    uint64_t epoch = 1;
    bool disabled = false;
  };

  struct Snapshot {
    BackingStoreId backing_id;
    uint64_t offset;
    size_t size;
    uint64_t first_line;
    uint64_t epoch;
    std::vector<uint64_t> versions;
  };

public:
  class FillAuthority {
  public:
    FillAuthority() = default;
    FillAuthority(FillAuthority &&) noexcept = default;
    FillAuthority &operator=(FillAuthority &&other) noexcept {
      if (this != &other) {
        lock_ = std::move(other.lock_);
        state_ = std::move(other.state_);
      }
      return *this;
    }
    FillAuthority(const FillAuthority &) = delete;
    FillAuthority &operator=(const FillAuthority &) = delete;

    [[nodiscard]] explicit operator bool() const noexcept { return lock_.owns_lock(); }

  private:
    friend class BackingLineVersions;
    friend class ReadStamp;
    explicit FillAuthority(std::shared_ptr<State> state,
                           std::shared_lock<std::shared_mutex> lock) noexcept
        : state_(std::move(state)), lock_(std::move(lock)) {}

    std::shared_ptr<State> state_;
    std::shared_lock<std::shared_mutex> lock_;
  };

  class ReadStamp {
  public:
    ReadStamp() = default;

    [[nodiscard]] explicit operator bool() const noexcept { return snapshot_ != nullptr; }
    [[nodiscard]] BackingStoreId backing_id() const noexcept {
      return snapshot_ ? snapshot_->backing_id : 0;
    }
    [[nodiscard]] uint64_t offset() const noexcept { return snapshot_ ? snapshot_->offset : 0; }
    [[nodiscard]] size_t size() const noexcept { return snapshot_ ? snapshot_->size : 0; }
    [[nodiscard]] bool covers(BackingStoreId backing_id, uint64_t offset,
                              size_t size) const noexcept {
      if (!snapshot_ || backing_id != snapshot_->backing_id || size == 0 ||
          offset < snapshot_->offset) {
        return false;
      }
      const uint64_t relative = offset - snapshot_->offset;
      return relative < snapshot_->size && size <= snapshot_->size - relative;
    }

    /// A failed try-lock or changed version skips the optional cache fill.
    /// The caller holds the returned authority only through the short fill.
    [[nodiscard]] FillAuthority try_acquire_fill(BackingStoreId backing_id, uint64_t offset,
                                                 size_t size) const noexcept {
      if (!state_ || !covers(backing_id, offset, size)) {
        return {};
      }
      try {
        std::shared_lock lock(state_->mutex, std::try_to_lock);
        if (!lock.owns_lock() || state_->disabled || state_->epoch != snapshot_->epoch) {
          return {};
        }
        const uint64_t first = offset & ~(state_->line_bytes - 1);
        const uint64_t last = (offset + size - 1) & ~(state_->line_bytes - 1);
        const size_t begin =
            static_cast<size_t>((first - snapshot_->first_line) / state_->line_bytes);
        const size_t end = static_cast<size_t>((last - snapshot_->first_line) / state_->line_bytes);
        if (end >= snapshot_->versions.size()) {
          return {};
        }
        for (size_t i = begin; i <= end; ++i) {
          const LineKey key{backing_id, snapshot_->first_line + i * state_->line_bytes};
          const auto found = state_->versions.find(key);
          if (found == state_->versions.end() || found->second != snapshot_->versions[i]) {
            return {};
          }
        }
        return FillAuthority(state_, std::move(lock));
      } catch (...) {
        return {};
      }
    }

  private:
    friend class BackingLineVersions;
    ReadStamp(std::shared_ptr<State> state, std::shared_ptr<const Snapshot> snapshot) noexcept
        : state_(std::move(state)), snapshot_(std::move(snapshot)) {}

    std::shared_ptr<State> state_;
    std::shared_ptr<const Snapshot> snapshot_;
  };

  class WriterAuthority {
  public:
    WriterAuthority() = default;
    WriterAuthority(WriterAuthority &&) noexcept = default;
    WriterAuthority &operator=(WriterAuthority &&other) noexcept {
      if (this != &other) {
        lock_ = std::move(other.lock_);
        state_ = std::move(other.state_);
      }
      return *this;
    }
    WriterAuthority(const WriterAuthority &) = delete;
    WriterAuthority &operator=(const WriterAuthority &) = delete;

    [[nodiscard]] explicit operator bool() const noexcept { return lock_.owns_lock(); }

    /// Call once after a successful backing write, before releasing authority.
    /// Unknown identity or invalid range conservatively invalidates all stamps.
    void advance(BackingStoreId backing_id, uint64_t offset, size_t size) noexcept {
      if (!lock_.owns_lock() || !state_ || state_->disabled) {
        return;
      }
      uint64_t first = 0;
      uint64_t last = 0;
      if (!bounded_lines(state_->line_bytes, backing_id, offset, size, first, last)) {
        state_->invalidate_all_locked();
        return;
      }
      if (state_->next_version == std::numeric_limits<uint64_t>::max()) {
        state_->invalidate_all_locked();
        state_->disabled = true;
        return;
      }
      const uint64_t version = state_->next_version++;
      for (auto &[key, tracked_version] : state_->versions) {
        if (key.backing_id == backing_id && key.line_offset >= first && key.line_offset <= last) {
          tracked_version = version;
        }
      }
    }

    /// Invalidate stamps that raced maintenance preparation before this
    /// authority was taken. Keep authority through the following effect.
    void invalidate_all() noexcept {
      if (lock_.owns_lock() && state_) {
        state_->invalidate_all_locked();
      }
    }

  private:
    friend class BackingLineVersions;
    explicit WriterAuthority(std::shared_ptr<State> state,
                             std::unique_lock<std::shared_mutex> lock) noexcept
        : state_(std::move(state)), lock_(std::move(lock)) {}

    std::shared_ptr<State> state_;
    std::unique_lock<std::shared_mutex> lock_;
  };

  /// A zero capacity or non-power-of-two line size disables tracking.
  BackingLineVersions(size_t line_bytes, size_t capacity)
      : state_(std::make_shared<State>(line_bytes, capacity)) {}

  /// Capture after maintenance and a successful backing read, before the VM
  /// effect publishes completion. A busy lock or allocation failure disables
  /// this read's optional fill without changing the committed read result.
  [[nodiscard]] std::optional<ReadStamp> try_capture_read(BackingStoreId backing_id,
                                                          uint64_t offset, size_t size) noexcept {
    WriterAuthority writer = try_acquire_writer();
    return capture_read_under_writer(writer, backing_id, offset, size);
  }

  /// Capture a successful backing read while retaining the effect's exclusive
  /// authority, so a concurrent writer cannot intervene before publication.
  [[nodiscard]] std::optional<ReadStamp> capture_read_under_writer(const WriterAuthority &writer,
                                                                   BackingStoreId backing_id,
                                                                   uint64_t offset,
                                                                   size_t size) noexcept {
    if (!writer || writer.state_.get() != state_.get() || state_->disabled) {
      return std::nullopt;
    }
    uint64_t first = 0;
    uint64_t last = 0;
    if (!bounded_lines(state_->line_bytes, backing_id, offset, size, first, last)) {
      return std::nullopt;
    }
    const uint64_t line_count = (last - first) / state_->line_bytes + 1;
    if (line_count > state_->capacity) {
      return std::nullopt;
    }
    try {
      auto snapshot = std::make_shared<Snapshot>();
      snapshot->backing_id = backing_id;
      snapshot->offset = offset;
      snapshot->size = size;
      snapshot->first_line = first;
      snapshot->epoch = state_->epoch;
      snapshot->versions.reserve(static_cast<size_t>(line_count));
      for (uint64_t i = 0; i < line_count; ++i) {
        const LineKey key{backing_id, first + i * state_->line_bytes};
        auto found = state_->versions.find(key);
        if (found == state_->versions.end()) {
          if (state_->next_version == std::numeric_limits<uint64_t>::max()) {
            state_->invalidate_all_locked();
            state_->disabled = true;
            return std::nullopt;
          }
          if (state_->versions.size() == state_->capacity) {
            // Keep the current read's lines resident. An evicted older stamp
            // simply fails its later freshness check.
            auto victim = state_->versions.begin();
            while (victim != state_->versions.end() && victim->first.backing_id == backing_id &&
                   victim->first.line_offset >= first && victim->first.line_offset <= last) {
              ++victim;
            }
            if (victim == state_->versions.end()) {
              return std::nullopt;
            }
            state_->versions.erase(victim);
          }
          found = state_->versions.emplace(key, state_->next_version++).first;
        }
        snapshot->versions.push_back(found->second);
      }
      return ReadStamp(state_, std::move(snapshot));
    } catch (...) {
      return std::nullopt;
    }
  }

  /// Acquire only after dirty resolver callbacks and cache maintenance have
  /// prepared the effect. On contention, release maintenance and retry the
  /// same request; never wait here while cache locks are held.
  [[nodiscard]] WriterAuthority try_acquire_writer() noexcept {
    try {
      std::unique_lock lock(state_->mutex, std::try_to_lock);
      if (!lock.owns_lock()) {
        return {};
      }
      return WriterAuthority(state_, std::move(lock));
    } catch (...) {
      return {};
    }
  }

  /// Invalidate older in-flight read stamps before cache maintenance may
  /// publish dirty backing bytes. This lock is released before any resolver
  /// callback or cache lock is entered.
  [[nodiscard]] bool try_invalidate_all() noexcept {
    try {
      std::unique_lock lock(state_->mutex, std::try_to_lock);
      if (!lock.owns_lock()) {
        return false;
      }
      state_->invalidate_all_locked();
      return true;
    } catch (...) {
      return false;
    }
  }
  void invalidate_all() {
    std::unique_lock lock(state_->mutex);
    state_->invalidate_all_locked();
  }

private:
  std::shared_ptr<State> state_;
};

} // namespace simdojo
