// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file canonical_line_freshness.h
/// @brief Bounded canonical backing-line versions for deferred peer fills.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <utility>

namespace rocjitsu {

/// Tracks up to kMaxEntries versions of 128-byte lines in canonical backing
/// coordinates. Eviction only rejects an old stamp; it can never make one
/// fresh again. A monotonic epoch also rejects stamps after maintenance clears
/// the table and later inserts the same canonical line.
class CanonicalLineFreshness {
  class State;
  struct Snapshot;

public:
  static constexpr size_t kLineSize = 128;
  static constexpr size_t kMaxEntries = 4096;
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
    friend class CanonicalLineFreshness;
    friend class ReadStamp;
    explicit FillAuthority(std::shared_ptr<State> state,
                           std::shared_lock<std::shared_mutex> lock) noexcept;

    std::shared_ptr<State> state_;
    std::shared_lock<std::shared_mutex> lock_;
  };

  class ReadStamp {
  public:
    ReadStamp() = default;

    [[nodiscard]] explicit operator bool() const noexcept { return snapshot_ != nullptr; }
    [[nodiscard]] uint64_t backing_id() const noexcept;
    [[nodiscard]] uint64_t offset() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] bool covers(uint64_t backing_id, uint64_t offset, size_t size) const noexcept;

    /// A failed try-lock or changed version skips the optional cache fill.
    /// The caller holds the returned authority only through the short fill.
    [[nodiscard]] FillAuthority try_acquire_fill(uint64_t backing_id, uint64_t offset,
                                                 size_t size) const noexcept;

  private:
    friend class CanonicalLineFreshness;
    ReadStamp(std::shared_ptr<State> state, std::shared_ptr<const Snapshot> snapshot) noexcept;

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
    void advance(uint64_t backing_id, uint64_t offset, size_t size) noexcept;

    /// Invalidate stamps that raced maintenance preparation before this
    /// authority was taken. Keep authority through the following effect.
    void invalidate_all() noexcept;

  private:
    friend class CanonicalLineFreshness;
    explicit WriterAuthority(std::shared_ptr<State> state,
                             std::unique_lock<std::shared_mutex> lock) noexcept;

    std::shared_ptr<State> state_;
    std::unique_lock<std::shared_mutex> lock_;
  };

  explicit CanonicalLineFreshness(size_t capacity = kMaxEntries);

  /// Capture after maintenance and a successful backing read, before the VM
  /// effect publishes completion. A busy lock or allocation failure disables
  /// this read's optional fill without changing the committed read result.
  [[nodiscard]] std::optional<ReadStamp> try_capture_read(uint64_t backing_id, uint64_t offset,
                                                          size_t size) noexcept;

  /// Capture a successful backing read while retaining the effect's exclusive
  /// authority, so a concurrent writer cannot intervene before publication.
  [[nodiscard]] std::optional<ReadStamp> capture_read_under_writer(const WriterAuthority &writer,
                                                                   uint64_t backing_id,
                                                                   uint64_t offset,
                                                                   size_t size) noexcept;

  /// Acquire only after dirty resolver callbacks and cache maintenance have
  /// prepared the effect. On contention, release maintenance and retry the
  /// same request; never wait here while cache locks are held.
  [[nodiscard]] WriterAuthority try_acquire_writer() noexcept;

  /// Invalidate older in-flight read stamps before cache maintenance may
  /// publish dirty backing bytes. This lock is released before any resolver
  /// callback or cache lock is entered.
  [[nodiscard]] bool try_invalidate_all() noexcept;
  void invalidate_all();

private:
  std::shared_ptr<State> state_;
};

} // namespace rocjitsu
