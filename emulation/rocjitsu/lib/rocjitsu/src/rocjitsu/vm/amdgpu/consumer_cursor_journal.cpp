// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/consumer_cursor_journal.h"

#include <cassert>
#include <stdexcept>
#include <utility>

namespace rocjitsu::amdgpu {

ConsumerCursorJournal::ConsumerCursorJournal(uint64_t address,
                                             std::optional<uint64_t> initial_cursor, uint32_t width)
    : address_(address), width_(width) {
  if (width_ != sizeof(uint32_t) && width_ != sizeof(uint64_t))
    throw std::invalid_argument("consumer cursor width must be 32 or 64 bits");
  reset(initial_cursor);
}

VmAccessOutcome ConsumerCursorJournal::initialize(const GpuVmAccess &access) {
  if (initialized_)
    return VmAccessOutcome::Complete;
  const AtomicLoadResult loaded = access.atomic_load(address_, width_);
  if (loaded.outcome == VmAccessOutcome::Complete) {
    cursor_ = loaded.value;
    initialized_ = true;
  }
  return loaded.outcome;
}

void ConsumerCursorJournal::retire(uint64_t cursor, GpuVmAccess access) {
  assert(!publication_access_);
  cursor_ = cursor;
  initialized_ = true;
  publication_access_.emplace(std::move(access));
}

VmAccessOutcome ConsumerCursorJournal::publish() {
  if (!publication_access_)
    return VmAccessOutcome::Complete;
  const VmAccessOutcome outcome = publication_access_->atomic_store(address_, width_, cursor_);
  if (outcome == VmAccessOutcome::Complete)
    publication_access_.reset();
  return outcome;
}

void ConsumerCursorJournal::reset(std::optional<uint64_t> initial_cursor) {
  cursor_ = initial_cursor.value_or(0);
  initialized_ = initial_cursor.has_value();
  publication_access_.reset();
}

} // namespace rocjitsu::amdgpu
