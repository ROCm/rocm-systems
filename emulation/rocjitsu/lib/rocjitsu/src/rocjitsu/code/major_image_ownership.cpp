// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/major_image_ownership.h"

#include "util/bit.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>

namespace rocjitsu::major_image_ownership {
namespace {

struct Owner {
  OwnerKind kind = OwnerKind::InputImage;
  uint64_t fixed_bytes = 0;
  const void *context = nullptr;
  MeasureBytes measure = nullptr;
  const void *shared_storage_identity = nullptr;

  [[nodiscard]] uint64_t bytes() const noexcept {
    return measure == nullptr ? fixed_bytes : measure(context);
  }
};

struct State {
  std::unordered_map<const void *, Owner> owners;
  Measurement measurement;
  Phase phase = Phase::None;
};

thread_local State *g_state = nullptr;

void add_kind_bytes(State &state, std::array<uint64_t, kOwnerKindCount> &by_kind, OwnerKind kind,
                    uint64_t bytes) noexcept {
  const size_t index = static_cast<size_t>(kind);
  const auto next = util::checked_add(by_kind[index], bytes);
  if (!next) {
    state.measurement.overflowed = true;
    by_kind[index] = std::numeric_limits<uint64_t>::max();
  } else {
    by_kind[index] = *next;
  }
}

void observe(State &state) noexcept {
  std::array<uint64_t, kOwnerKindCount> by_kind{};
  uint64_t total = 0;
  uint64_t observed_mask = 0;
  for (const auto &[identity, owner] : state.owners) {
    (void)identity;
    observed_mask |= uint64_t{1} << static_cast<size_t>(owner.kind);
    if (owner.kind != OwnerKind::InputImage)
      add_kind_bytes(state, by_kind, owner.kind, owner.bytes());
  }
  // Public transform entry points can nest while referring to the same
  // staging image. Charge aliased input views once while keeping distinct
  // input buffers additive. The owner set is deliberately tiny, so a
  // no-allocation scan keeps observation safe inside noexcept transform paths.
  const std::less<const void *> identity_less;
  for (const auto &[identity, owner] : state.owners) {
    if (owner.kind != OwnerKind::InputImage)
      continue;
    const void *storage =
        owner.shared_storage_identity == nullptr ? identity : owner.shared_storage_identity;
    bool first_registration = true;
    uint64_t shared_bytes = owner.bytes();
    for (const auto &[other_identity, other] : state.owners) {
      if (other.kind != OwnerKind::InputImage)
        continue;
      const void *other_storage =
          other.shared_storage_identity == nullptr ? other_identity : other.shared_storage_identity;
      if (other_storage != storage)
        continue;
      if (identity_less(other_identity, identity))
        first_registration = false;
      shared_bytes = std::max(shared_bytes, other.bytes());
    }
    if (first_registration)
      add_kind_bytes(state, by_kind, OwnerKind::InputImage, shared_bytes);
  }
  for (const uint64_t bytes : by_kind) {
    const auto next_total = util::checked_add(total, bytes);
    if (!next_total) {
      state.measurement.overflowed = true;
      total = std::numeric_limits<uint64_t>::max();
      break;
    }
    total = *next_total;
  }
  state.measurement.overall_peak_bytes = std::max(state.measurement.overall_peak_bytes, total);
  if (state.phase == Phase::None)
    return;
  PhaseMeasurement &phase = state.measurement.phases[static_cast<size_t>(state.phase) - 1u];
  phase.observed_owner_mask |= observed_mask;
  if (total > phase.peak_bytes) {
    phase.peak_bytes = total;
    phase.bytes_at_peak = by_kind;
  }
}

} // namespace

ScopedMeasurement::ScopedMeasurement() {
  if (g_state != nullptr) {
    state_ = g_state;
    return;
  }
  state_ = new State;
  g_state = static_cast<State *>(state_);
  owns_state_ = true;
}

ScopedMeasurement::~ScopedMeasurement() {
  if (!owns_state_)
    return;
  assert(g_state == state_ && "major-image measurement scopes must be destroyed in stack order");
  g_state = nullptr;
  delete static_cast<State *>(state_);
}

Measurement ScopedMeasurement::snapshot() const {
  auto *state = static_cast<State *>(state_);
  observe(*state);
  return state->measurement;
}

ScopedPhase::ScopedPhase(Phase phase, bool only_if_none) {
  if (g_state == nullptr || (only_if_none && g_state->phase != Phase::None))
    return;
  previous_ = g_state->phase;
  g_state->phase = phase;
  changed_ = true;
  observe(*g_state);
}

ScopedPhase::~ScopedPhase() {
  if (!changed_ || g_state == nullptr)
    return;
  g_state->phase = previous_;
  observe(*g_state);
}

ScopedOwner::ScopedOwner(OwnerKind kind, uint64_t bytes) noexcept
    : registered_(register_owner(this, kind, bytes)) {}

ScopedOwner::ScopedOwner(OwnerKind kind, const void *shared_storage_identity,
                         uint64_t bytes) noexcept
    : registered_(register_owner(this, kind, shared_storage_identity, bytes)) {}

ScopedOwner::ScopedOwner(OwnerKind kind, const void *context, MeasureBytes measure) noexcept
    : registered_(register_owner(this, kind, context, measure)) {}

ScopedOwner::~ScopedOwner() {
  if (registered_)
    unregister_owner(this);
}

void ScopedOwner::checkpoint() const noexcept {
  if (registered_)
    major_image_ownership::checkpoint();
}

bool register_impl(const void *identity, Owner owner) noexcept {
  if (g_state == nullptr)
    return false;
  try {
    const auto [it, inserted] = g_state->owners.emplace(identity, owner);
    (void)it;
    if (!inserted) {
      g_state->measurement.bookkeeping_error = true;
      return false;
    }
    observe(*g_state);
    return true;
  } catch (...) {
    g_state->measurement.bookkeeping_error = true;
    return false;
  }
}

bool register_owner(const void *identity, OwnerKind kind, uint64_t bytes) noexcept {
  return register_impl(identity, Owner{.kind = kind, .fixed_bytes = bytes});
}

bool register_owner(const void *identity, OwnerKind kind, const void *shared_storage_identity,
                    uint64_t bytes) noexcept {
  return register_impl(identity, Owner{.kind = kind,
                                       .fixed_bytes = bytes,
                                       .shared_storage_identity = shared_storage_identity});
}

bool register_owner(const void *identity, OwnerKind kind, const void *context,
                    MeasureBytes measure) noexcept {
  if (measure == nullptr) {
    if (g_state != nullptr)
      g_state->measurement.bookkeeping_error = true;
    return false;
  }
  return register_impl(identity, Owner{.kind = kind, .context = context, .measure = measure});
}

void add_owner_bytes(const void *identity, uint64_t bytes) noexcept {
  if (g_state == nullptr)
    return;
  const auto owner = g_state->owners.find(identity);
  if (owner == g_state->owners.end() || owner->second.measure != nullptr) {
    g_state->measurement.bookkeeping_error = true;
    return;
  }
  const auto total = util::checked_add(owner->second.fixed_bytes, bytes);
  if (!total) {
    owner->second.fixed_bytes = std::numeric_limits<uint64_t>::max();
    g_state->measurement.overflowed = true;
  } else {
    owner->second.fixed_bytes = *total;
  }
  observe(*g_state);
}

void transfer_owner_impl(const void *old_identity, const void *new_identity,
                         const std::optional<const void *> &new_context) noexcept {
  if (g_state == nullptr)
    return;
  const auto old = g_state->owners.find(old_identity);
  if (old == g_state->owners.end()) {
    g_state->measurement.bookkeeping_error = true;
    return;
  }
  if (g_state->owners.contains(new_identity) || (old->second.measure != nullptr && !new_context)) {
    g_state->measurement.bookkeeping_error = true;
    return;
  }
  try {
    auto owner = g_state->owners.extract(old);
    owner.key() = new_identity;
    if (new_context)
      owner.mapped().context = *new_context;
    const auto inserted = g_state->owners.insert(std::move(owner));
    if (!inserted.inserted)
      g_state->measurement.bookkeeping_error = true;
  } catch (...) {
    g_state->measurement.bookkeeping_error = true;
  }
  observe(*g_state);
}

void transfer_owner(const void *old_identity, const void *new_identity) noexcept {
  transfer_owner_impl(old_identity, new_identity, std::nullopt);
}

void transfer_owner(const void *old_identity, const void *new_identity,
                    const void *new_context) noexcept {
  transfer_owner_impl(old_identity, new_identity, new_context);
}

void unregister_owner(const void *identity) noexcept {
  if (g_state == nullptr)
    return;
  observe(*g_state);
  g_state->owners.erase(identity);
  observe(*g_state);
}

void checkpoint() noexcept {
  if (g_state != nullptr)
    observe(*g_state);
}

} // namespace rocjitsu::major_image_ownership
