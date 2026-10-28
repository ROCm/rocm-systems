// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_handles.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace rocjitsu::amdgpu {
class ScratchReservation;

/// @brief Device-wide ownership of scratch backing ranges, isolated by address-space generation.
/// @details Does not allocate memory or choose execution placement. Its mutex is a leaf lock:
/// neither reserve nor release enters a CU. Each returned reservation releases its own range.
class ScratchReservations {
public:
  /// @brief Atomically reserve an exact, nonempty range, or return null without mutation.
  /// @details Adjacent ranges are allowed. backing retains any underlying allocation, such as
  /// a PM4 pool slot, until this reservation is released. Reservations may outlive this manager.
  [[nodiscard]] std::shared_ptr<ScratchReservation> try_reserve(AddressSpaceHandle space,
                                                                uint64_t base, uint64_t bytes,
                                                                std::shared_ptr<void> backing = {});

private:
  friend class ScratchReservation;
  using Key = std::tuple<uint32_t, uint64_t, uint64_t>; // Address-space slot, generation, base.
  struct State {
    std::mutex mutex;
    std::map<Key, uint64_t> occupied; // Lengths; keys and lengths never change while reserved.
  };
  std::shared_ptr<State> state_ = std::make_shared<State>();
};

/// @brief One wave's scratch allocation, independent of its mutable SCRATCH_BASE register.
class ScratchReservation {
public:
  ~ScratchReservation() {
    std::lock_guard lock(state_->mutex);
    state_->occupied.erase(region_);
  }
  ScratchReservation(const ScratchReservation &) = delete;
  ScratchReservation &operator=(const ScratchReservation &) = delete;

  uint64_t base() const { return std::get<2>(region_->first); }
  uint64_t size() const { return region_->second; }

private:
  friend class ScratchReservations;
  ScratchReservation(std::shared_ptr<ScratchReservations::State> state,
                     std::map<ScratchReservations::Key, uint64_t>::iterator region,
                     std::shared_ptr<void> backing)
      : state_(std::move(state)), region_(region), backing_(std::move(backing)) {}

  std::shared_ptr<ScratchReservations::State> state_;
  std::map<ScratchReservations::Key, uint64_t>::iterator region_;
  // Released after the destructor body drops the reservation mutex.
  std::shared_ptr<void> backing_;
};

inline std::shared_ptr<ScratchReservation>
ScratchReservations::try_reserve(AddressSpaceHandle space, uint64_t base, uint64_t bytes,
                                 std::shared_ptr<void> backing) {
  if (!space || bytes == 0 || bytes - 1 > std::numeric_limits<uint64_t>::max() - base)
    return {};
  const Key key{space.slot, space.generation, base};
  const auto same_space = [&](const auto &entry) {
    return std::get<0>(entry.first) == space.slot && std::get<1>(entry.first) == space.generation;
  };
  std::map<Key, uint64_t>::iterator region;
  {
    std::lock_guard lock(state_->mutex);
    const auto next = state_->occupied.lower_bound(key);
    if (next != state_->occupied.end() && same_space(*next) &&
        std::get<2>(next->first) - base < bytes)
      return {};
    if (next != state_->occupied.begin()) {
      const auto previous = std::prev(next);
      if (same_space(*previous) && base - std::get<2>(previous->first) < previous->second)
        return {};
    }
    region = state_->occupied.emplace_hint(next, key, bytes);
  }
  ScratchReservation *reservation;
  try {
    reservation = new ScratchReservation(state_, region, std::move(backing));
  } catch (...) {
    std::lock_guard lock(state_->mutex);
    state_->occupied.erase(region);
    throw;
  }
  // If control-block allocation fails, shared_ptr deletes the reservation and releases its range.
  return std::shared_ptr<ScratchReservation>(reservation);
}
} // namespace rocjitsu::amdgpu
