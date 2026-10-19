// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/canonical_line_freshness.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

struct LineKey {
  uint64_t backing_id;
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

bool bounded_lines(uint64_t backing_id, uint64_t offset, size_t size, uint64_t &first,
                   uint64_t &last) noexcept {
  if (backing_id == 0 || size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - offset)
    return false;
  first = offset & ~(CanonicalLineFreshness::kLineSize - 1);
  last = (offset + size - 1) & ~(CanonicalLineFreshness::kLineSize - 1);
  return true;
}

} // namespace

struct CanonicalLineFreshness::State {
  explicit State(size_t capacity) : capacity(capacity) {}

  void invalidate_all_locked() noexcept {
    versions.clear();
    if (epoch == std::numeric_limits<uint64_t>::max())
      disabled = true;
    else
      ++epoch;
  }

  std::shared_mutex mutex;
  std::unordered_map<LineKey, uint64_t, LineKeyHash> versions;
  size_t capacity;
  uint64_t next_version = 1;
  uint64_t epoch = 1;
  bool disabled = false;
};

struct CanonicalLineFreshness::Snapshot {
  uint64_t backing_id;
  uint64_t offset;
  size_t size;
  uint64_t first_line;
  uint64_t epoch;
  std::vector<uint64_t> versions;
};

CanonicalLineFreshness::FillAuthority::FillAuthority(
    std::shared_ptr<State> state, std::shared_lock<std::shared_mutex> lock) noexcept
    : state_(std::move(state)), lock_(std::move(lock)) {}

CanonicalLineFreshness::ReadStamp::ReadStamp(std::shared_ptr<State> state,
                                             std::shared_ptr<const Snapshot> snapshot) noexcept
    : state_(std::move(state)), snapshot_(std::move(snapshot)) {}

uint64_t CanonicalLineFreshness::ReadStamp::backing_id() const noexcept {
  return snapshot_ ? snapshot_->backing_id : 0;
}

uint64_t CanonicalLineFreshness::ReadStamp::offset() const noexcept {
  return snapshot_ ? snapshot_->offset : 0;
}

size_t CanonicalLineFreshness::ReadStamp::size() const noexcept {
  return snapshot_ ? snapshot_->size : 0;
}

bool CanonicalLineFreshness::ReadStamp::covers(uint64_t backing_id, uint64_t offset,
                                               size_t size) const noexcept {
  if (!snapshot_ || backing_id != snapshot_->backing_id || size == 0 || offset < snapshot_->offset)
    return false;
  const uint64_t relative = offset - snapshot_->offset;
  return relative < snapshot_->size && size <= snapshot_->size - relative;
}

CanonicalLineFreshness::FillAuthority
CanonicalLineFreshness::ReadStamp::try_acquire_fill(uint64_t backing_id, uint64_t offset,
                                                    size_t size) const noexcept {
  if (!state_ || !covers(backing_id, offset, size))
    return {};
  try {
    std::shared_lock lock(state_->mutex, std::try_to_lock);
    if (!lock.owns_lock() || state_->disabled || state_->epoch != snapshot_->epoch)
      return {};
    const uint64_t first = offset & ~(kLineSize - 1);
    const uint64_t last = (offset + size - 1) & ~(kLineSize - 1);
    const size_t begin = static_cast<size_t>((first - snapshot_->first_line) / kLineSize);
    const size_t end = static_cast<size_t>((last - snapshot_->first_line) / kLineSize);
    if (end >= snapshot_->versions.size())
      return {};
    for (size_t i = begin; i <= end; ++i) {
      const LineKey key{backing_id, snapshot_->first_line + i * kLineSize};
      const auto found = state_->versions.find(key);
      if (found == state_->versions.end() || found->second != snapshot_->versions[i])
        return {};
    }
    return FillAuthority(state_, std::move(lock));
  } catch (...) {
    return {};
  }
}

CanonicalLineFreshness::WriterAuthority::WriterAuthority(
    std::shared_ptr<State> state, std::unique_lock<std::shared_mutex> lock) noexcept
    : state_(std::move(state)), lock_(std::move(lock)) {}

void CanonicalLineFreshness::WriterAuthority::advance(uint64_t backing_id, uint64_t offset,
                                                      size_t size) noexcept {
  if (!lock_.owns_lock() || !state_ || state_->disabled)
    return;
  uint64_t first = 0;
  uint64_t last = 0;
  if (!bounded_lines(backing_id, offset, size, first, last)) {
    state_->invalidate_all_locked();
    return;
  }
  if (state_->next_version == std::numeric_limits<uint64_t>::max()) {
    state_->invalidate_all_locked();
    state_->disabled = true;
    return;
  }
  const uint64_t version = state_->next_version++;
  for (auto &[key, tracked_version] : state_->versions)
    if (key.backing_id == backing_id && key.line_offset >= first && key.line_offset <= last)
      tracked_version = version;
}

void CanonicalLineFreshness::WriterAuthority::invalidate_all() noexcept {
  if (lock_.owns_lock() && state_)
    state_->invalidate_all_locked();
}

CanonicalLineFreshness::CanonicalLineFreshness(size_t capacity)
    : state_(std::make_shared<State>(capacity == 0 ? 1 : std::min(capacity, kMaxEntries))) {}

std::optional<CanonicalLineFreshness::ReadStamp>
CanonicalLineFreshness::try_capture_read(uint64_t backing_id, uint64_t offset,
                                         size_t size) noexcept {
  WriterAuthority writer = try_acquire_writer();
  return capture_read_under_writer(writer, backing_id, offset, size);
}

std::optional<CanonicalLineFreshness::ReadStamp> CanonicalLineFreshness::capture_read_under_writer(
    const WriterAuthority &writer, uint64_t backing_id, uint64_t offset, size_t size) noexcept {
  if (!writer || writer.state_.get() != state_.get() || state_->disabled)
    return std::nullopt;
  uint64_t first = 0;
  uint64_t last = 0;
  if (!bounded_lines(backing_id, offset, size, first, last))
    return std::nullopt;
  const uint64_t line_count = (last - first) / kLineSize + 1;
  if (line_count > state_->capacity)
    return std::nullopt;
  try {
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->backing_id = backing_id;
    snapshot->offset = offset;
    snapshot->size = size;
    snapshot->first_line = first;
    snapshot->epoch = state_->epoch;
    snapshot->versions.reserve(static_cast<size_t>(line_count));
    for (uint64_t i = 0; i < line_count; ++i) {
      const LineKey key{backing_id, first + i * kLineSize};
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
                 victim->first.line_offset >= first && victim->first.line_offset <= last)
            ++victim;
          if (victim == state_->versions.end())
            return std::nullopt;
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

CanonicalLineFreshness::WriterAuthority CanonicalLineFreshness::try_acquire_writer() noexcept {
  try {
    std::unique_lock lock(state_->mutex, std::try_to_lock);
    if (!lock.owns_lock())
      return {};
    return WriterAuthority(state_, std::move(lock));
  } catch (...) {
    return {};
  }
}

void CanonicalLineFreshness::invalidate_all() {
  std::unique_lock lock(state_->mutex);
  state_->invalidate_all_locked();
}

bool CanonicalLineFreshness::try_invalidate_all() noexcept {
  try {
    std::unique_lock lock(state_->mutex, std::try_to_lock);
    if (!lock.owns_lock())
      return false;
    state_->invalidate_all_locked();
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace rocjitsu
