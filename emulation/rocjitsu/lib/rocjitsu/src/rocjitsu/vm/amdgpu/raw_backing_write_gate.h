// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
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
    Permit(Permit &&other) noexcept : gate_(std::exchange(other.gate_, nullptr)), id_(other.id_) {}
    Permit &operator=(Permit &&other) noexcept {
      if (this != &other) {
        release();
        gate_ = std::exchange(other.gate_, nullptr);
        id_ = other.id_;
      }
      return *this;
    }
    Permit(const Permit &) = delete;
    Permit &operator=(const Permit &) = delete;
    ~Permit() { release(); }

  private:
    friend class VmRawBackingWriteGate;
    Permit(VmRawBackingWriteGate *gate, uint64_t id) : gate_(gate), id_(id) {}
    void release() noexcept {
      if (gate_) {
        std::lock_guard lock(gate_->mutex_);
        erase_entry(gate_->active_, id_);
      }
      gate_ = nullptr;
    }
    VmRawBackingWriteGate *gate_ = nullptr;
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
        std::lock_guard lock(gate_->mutex_);
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
    std::lock_guard lock(mutex_);
    if (next_id_ == 0)
      return std::nullopt;
    const Span requested{begin, size};
    for (const Entry &closed : closed_)
      if (overlaps(requested, closed.span))
        return std::nullopt;
    try {
      const uint64_t id = next_id_;
      active_.push_back({requested, id});
      ++next_id_;
      return Permit(this, id);
    } catch (const std::bad_alloc &) {
      return std::nullopt;
    }
  }

  [[nodiscard]] std::optional<Closed> try_close(uint64_t begin, uint64_t size) noexcept {
    if (!valid(begin, size))
      return std::nullopt;
    std::lock_guard lock(mutex_);
    if (next_id_ == 0)
      return std::nullopt;
    const Span requested{begin, size};
    for (const Entry &active : active_)
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

  std::mutex mutex_;
  std::vector<Entry> active_;
  std::vector<Entry> closed_;
  uint64_t next_id_ = 1;
};

} // namespace rocjitsu::amdgpu
