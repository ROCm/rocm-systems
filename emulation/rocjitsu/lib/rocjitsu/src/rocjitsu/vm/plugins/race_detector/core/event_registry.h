// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/instruction_snapshot.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

/// Append-only event store with amortized prefix trimming.
///
/// Events are allocated with monotonically increasing IDs. Callers access
/// event data through typed accessors indexed by EventId. The registry
/// periodically trims completed entries from the front. Completed LDS events
/// that still participate in cross-wave checks move to a sparse side table,
/// preserving their IDs and captured instructions until barrier retirement.
///
/// An event is trimmable when it can never be accessed again:
///   - RETIRED events (explicitly retired at s_barrier via retireEvent).
///   - WAVE_COMPLETE events without LDS intervals: non-LDS events, and LDS
///     events that touch no LDS bytes, such as a tensor transfer that moves no
///     data. These have left all per-wave and per-register lists and were never
///     added to ldsWriteEvents/ldsReadEvents; retireEvent is a no-op for them,
///     so they don't need to wait for s_barrier.
///   - WAVE_COMPLETE LDS events with intervals are NOT trimmable until
///     retired, because retireEvent must still remove them from
///     ldsWriteEvents/ldsReadEvents and adjust byte counts.
///
/// Trimming is attempted every kTrimAttemptInterval allocations. The
/// trimmable prefix is scanned and erased only if it covers at least half
/// the dense entries, amortizing the O(n) vector shift.
class EventRegistry {
  struct SupersededRegister {
    uint32_t reg;
    std::array<uint64_t, 4> lanes;
  };
  std::unordered_map<int, std::vector<SupersededRegister>> supersededFlatBytes_;
  struct EventInfo {
    WaveId waveId;
    InstructionSnapshot instruction;
    MemoryEventType type;
    std::array<amdgpu::MemoryCounterObligation, amdgpu::MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS>
        counterObligations;
    uint8_t numCounterObligations;
    uint8_t satisfiedCounterMask;
    VgprWriteOrder writeOrder;
    LdsAccess ldsAccess;
    bool ldsPublished;
    EventStatus status;
    uint8_t byteMask;
    uint8_t lastRegisterByteMask;
    bool hasLdsSourceReads;
    uint64_t execMask;
    std::vector<uint32_t> registers;
    IntervalSet ldsIntervals;
    // Younger counter-only units in this FIFO, before its next tracked event.
    std::array<uint8_t, amdgpu::MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS> followingUnits{};
    ConditionalVgprWrite feedback{};
  };

public:
  // A younger routed issue can extend an older event's counter distance or
  // supersede its FLAT result bytes. Save only entries the issue changes.
  class RoutedChanges {
  public:
    struct Distance {
      EventId id;
      uint8_t slot;
      uint8_t previous;
    };
    struct Register {
      EventId id;
      uint32_t reg;
      std::array<uint64_t, 4> previous;
    };
    std::vector<Distance> distances;
    std::vector<Register> registers;
    void clear() {
      distances.clear();
      registers.clear();
    }
  };

  void restoreRoutedChanges(const RoutedChanges &changes) {
    for (const auto &change : changes.distances | std::views::reverse)
      if (contains(change.id) && status(change.id) == EventStatus::ACTIVE)
        get(change.id).followingUnits[change.slot] = change.previous;
    for (const auto &change : changes.registers | std::views::reverse) {
      const auto found = supersededFlatBytes_.find(change.id.value);
      if (found == supersededFlatBytes_.end())
        continue;
      auto &masks = found->second;
      auto reg = std::ranges::find(masks, change.reg, &SupersededRegister::reg);
      if (reg == masks.end())
        continue;
      reg->lanes = change.previous;
      if (reg->lanes == std::array<uint64_t, 4>{})
        masks.erase(reg);
      if (masks.empty())
        supersededFlatBytes_.erase(found);
    }
  }

  /// Allocate an event with its producing instruction and every counter obligation.
  EventId add(WaveId waveId, InstructionSnapshot instruction, MemoryEventType type,
              std::vector<uint32_t> registers, uint64_t execMask, uint8_t byteMask,
              IntervalSet ldsIntervals,
              std::span<const amdgpu::MemoryCounterObligation> counterObligations,
              VgprWriteOrder writeOrder, uint8_t lastRegisterByteMask = 0,
              LdsAccess ldsAccess = {}) {
    int id = base_offset_ + static_cast<int>(entries_.size());
    assert(!counterObligations.empty() ||
           (isFromLds(type) && registers.empty() && !ldsIntervals.empty()));
    assert(counterObligations.size() <= amdgpu::MemoryIssueInfo::MAX_COUNTER_OBLIGATIONS);
    EventInfo event{waveId,
                    instruction,
                    type,
                    {},
                    static_cast<uint8_t>(counterObligations.size()),
                    0,
                    writeOrder,
                    ldsAccess,
                    false,
                    counterObligations.empty() ? EventStatus::WAVE_COMPLETE : EventStatus::ACTIVE,
                    byteMask,
                    lastRegisterByteMask ? lastRegisterByteMask : byteMask,
                    false,
                    execMask,
                    std::move(registers),
                    std::move(ldsIntervals)};
    std::ranges::copy(counterObligations, event.counterObligations.begin());
    entries_.push_back(std::move(event));

    // To prevent the number of events recorded growing indefinitely, we try to
    // remove retired events from time to time.
    if (id % kTrimAttemptInterval == kTrimAttemptInterval - 1)
      tryTrimEvents();
    return EventId{id};
  }

  /// Transition ACTIVE → WAVE_COMPLETE after every counter obligation is satisfied.
  void markComplete(EventId id) {
    assert(get(id).status == EventStatus::ACTIVE);
    assert(allWaitCountersSatisfied(id));
    auto &e = get(id);
    if (e.writeOrder.flat_lds_lanes)
      supersededFlatBytes_.erase(id.value);
    e.status = e.ldsPublished ? EventStatus::RETIRED : EventStatus::WAVE_COMPLETE;
  }

  /// Transition → RETIRED (s_barrier flushed this event).
  void markRetired(EventId id) {
    if (id.value < base_offset_) {
      completed_lds_.erase(id.value);
      return;
    }
    get(id).status = EventStatus::RETIRED;
  }

  // -- Typed accessors (all inline) --

  MemoryEventType type(EventId id) const { return get(id).type; }
  std::span<const amdgpu::MemoryCounterObligation> counterObligations(EventId id) const {
    const auto &event = get(id);
    return {event.counterObligations.data(), event.numCounterObligations};
  }
  bool hasPendingWaitCounter(EventId id, amdgpu::WaitCounterType waitType) const {
    const auto &event = get(id);
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if ((event.satisfiedCounterMask & (uint8_t{1} << i)) == 0 &&
          amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        return true;
    }
    return false;
  }
  std::optional<MemoryOrderClass> pendingCompletionClass(EventId id,
                                                         amdgpu::WaitCounterType waitType) const {
    const auto &event = get(id);
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if ((event.satisfiedCounterMask & (uint8_t{1} << i)) == 0 &&
          amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        return event.counterObligations[i].completion_class();
    }
    return std::nullopt;
  }
  // No identity is needed for a younger issue that owns no registers or LDS
  // bytes. Its distance still establishes when a partial wait covers this event.
  bool appendCounterOnly(EventId id, amdgpu::MemoryCounterObligation issue,
                         RoutedChanges *changes = nullptr) {
    auto &event = get(id);
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      const auto prior = event.counterObligations[i];
      if ((event.satisfiedCounterMask & (uint8_t{1} << i)) == 0 &&
          prior.wait_counter_type() == issue.wait_counter_type() &&
          prior.completion_class() == issue.completion_class()) {
        auto &distance = event.followingUnits[i];
        if (changes)
          changes->distances.push_back({id, i, distance});
        distance = static_cast<uint8_t>(std::min<int>(std::numeric_limits<uint8_t>::max(),
                                                      distance + issue.counter_increment()));
        return true;
      }
    }
    return false;
  }
  int pendingCounterIncrement(EventId id, amdgpu::WaitCounterType waitType) const {
    const auto &event = get(id);
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if ((event.satisfiedCounterMask & (uint8_t{1} << i)) == 0 &&
          amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        return event.counterObligations[i].counter_increment() + event.followingUnits[i];
    }
    return 0;
  }
  bool satisfyWaitCounter(EventId id, amdgpu::WaitCounterType waitType) {
    auto &event = get(id);
    assert(event.status == EventStatus::ACTIVE);
    for (uint8_t i = 0; i < event.numCounterObligations; ++i) {
      if (amdgpu::wait_counter_covers(waitType, event.counterObligations[i].wait_counter_type()))
        event.satisfiedCounterMask |= uint8_t{1} << i;
    }
    return allWaitCountersSatisfied(id);
  }
  bool allWaitCountersSatisfied(EventId id) const {
    const auto &event = get(id);
    return event.satisfiedCounterMask ==
           static_cast<uint8_t>((uint8_t{1} << event.numCounterObligations) - 1);
  }
  VgprWriteOrder writeOrder(EventId id) const {
    const auto order = get(id).writeOrder;
    // A generic GFX10 FLAT cannot overwrite in order. As a producer, once
    // its LGKM position completes, only an ordinary VMEM writeback remains.
    if (order.type == amdgpu::VmemWriteback::GenericFlat &&
        !hasPendingWaitCounter(id, amdgpu::WaitCounterType::LGKMCNT))
      return MemoryOrderClass::VMEM;
    return order;
  }
  /// Return the lanes whose outstanding write can be overtaken by this write.
  uint64_t conflictingWriteLanes(EventId id, VgprWriteOrder incoming, uint64_t lanes) const {
    const auto prior = writeOrder(id);
    const auto conflicts = [&](VgprWriteOrder fifo, uint64_t active) -> uint64_t {
      if (fifo.completion == MemoryOrderClass::UNORDERED)
        return active;
      if (!prior.flat_lds_lanes)
        return prior == fifo ? 0 : active;
      uint64_t pending = 0;
      if (fifo != MemoryOrderClass::LDS &&
          hasPendingWaitCounter(id, amdgpu::WaitCounterType::LGKMCNT))
        pending |= active & *prior.flat_lds_lanes;
      if (fifo != MemoryOrderClass::VMEM &&
          hasPendingWaitCounter(id, amdgpu::WaitCounterType::VMCNT))
        pending |= active & ~*prior.flat_lds_lanes;
      return pending;
    };
    if (!incoming.flat_lds_lanes)
      return conflicts(incoming, lanes);
    return conflicts(MemoryOrderClass::LDS, lanes & *incoming.flat_lds_lanes) |
           conflicts(MemoryOrderClass::VMEM, lanes & ~*incoming.flat_lds_lanes);
  }
  MemoryOrderClass memoryOrder(EventId id) const { return get(id).writeOrder.completion; }
  bool readsLds(EventId id) const {
    const auto &e = get(id);
    return e.ldsAccess.read_modify_write || isFromLds(e.type);
  }
  bool ldsAtomic(EventId id) const { return get(id).ldsAccess.read_modify_write; }
  bool atomicLdsElementMatches(EventId id, uint8_t bytes) const {
    return bytes != 0 && get(id).ldsAccess.aligned_atomic_bytes == bytes;
  }
  bool writesLds(EventId id) const {
    const auto &e = get(id);
    return e.ldsAccess.read_modify_write || isToLds(e.type);
  }
  MemoryOrderClass ldsOrder(EventId id) const {
    const auto &e = get(id);
    return e.ldsAccess.order(e.writeOrder.completion);
  }
  bool ldsAccessComplete(EventId id) const {
    const auto &e = get(id);
    return e.status != EventStatus::ACTIVE ||
           (e.ldsAccess.uses_flat_lds &&
            !hasPendingWaitCounter(id, amdgpu::WaitCounterType::LGKMCNT) &&
            !hasPendingWaitCounter(id, amdgpu::WaitCounterType::DSCNT));
  }
  bool hasLiveLdsIntervals(EventId id) const {
    const auto &e = get(id);
    return !e.ldsPublished && !e.ldsIntervals.empty();
  }
  void markLdsPublished(EventId id) { get(id).ldsPublished = true; }
  EventStatus status(EventId id) const { return get(id).status; }
  bool isTrimmable(EventId id) const { return isEntryTrimmable(get(id)); }
  uint64_t pc(EventId id) const { return instruction(id).pc(); }
  const InstructionSnapshot &instruction(EventId id) const { return get(id).instruction; }
  uint8_t byteMask(EventId id) const { return get(id).byteMask; }
  /// Packed loads can write fewer bytes in the last destination without
  /// creating a second event or consuming another wait-counter increment.
  uint8_t registerByteMask(EventId id, uint32_t reg) const {
    if (const auto feedback = get(id).feedback; feedback.lanes && reg == feedback.reg)
      return 0xf;
    const auto &event = get(id);
    return !event.registers.empty() && reg == event.registers.back() ? event.lastRegisterByteMask
                                                                     : event.byteMask;
  }
  void setConditionalVgprWrite(EventId id, ConditionalVgprWrite feedback) {
    get(id).feedback = feedback;
  }
  // An ordered native LDS load replaces only the FLAT bytes supplied by the
  // same LDS pipeline. The old event still owns its original counter positions.
  void supersedeFlatLdsResult(EventId id, uint32_t reg, uint64_t lanes, uint8_t bytes,
                              RoutedChanges *changes = nullptr) {
    const auto &event = get(id);
    lanes &= event.writeOrder.flat_lds_lanes.value_or(0) & event.execMask;
    bytes &= registerByteMask(id, reg);
    if (!lanes || !bytes)
      return;
    auto &masks = supersededFlatBytes_[id.value];
    auto it = std::ranges::find(masks, reg, &SupersededRegister::reg);
    if (it == masks.end())
      it = masks.insert(it, {reg, {}});
    if (changes)
      changes->registers.push_back({id, reg, it->lanes});
    for (unsigned byte = 0; byte < it->lanes.size(); ++byte)
      if (bytes & (1u << byte))
        it->lanes[byte] |= lanes;
  }

  uint64_t registerLanes(EventId id, uint32_t reg, uint8_t bytes = 0xf) const {
    const auto &event = get(id);
    const uint64_t lanes =
        event.feedback.lanes && reg == event.feedback.reg ? event.feedback.lanes : event.execMask;
    if (!event.writeOrder.flat_lds_lanes)
      return lanes;
    const auto found = supersededFlatBytes_.find(id.value);
    if (found == supersededFlatBytes_.end())
      return lanes;
    const auto it = std::ranges::find(found->second, reg, &SupersededRegister::reg);
    if (it == found->second.end())
      return lanes;
    bytes &= registerByteMask(id, reg);
    uint64_t pending = 0;
    for (unsigned byte = 0; byte < it->lanes.size(); ++byte)
      if (bytes & (1u << byte))
        pending |= lanes & ~it->lanes[byte];
    return pending;
  }
  uint64_t execMask(EventId id) const { return get(id).execMask; }
  WaveId waveId(EventId id) const { return get(id).waveId; }

  std::span<const uint32_t> registers(EventId id) const { return get(id).registers; }

  const IntervalSet &ldsIntervals(EventId id) const { return get(id).ldsIntervals; }

  void markLdsSourceReads(EventId id) { get(id).hasLdsSourceReads = true; }

  bool isActiveForLane(EventId id, int lane) const { return (get(id).execMask >> lane) & 1; }

  // -- Test helpers --

  /// Number of events currently stored (live + not-yet-trimmed dead).
  int size() const { return static_cast<int>(entries_.size() + completed_lds_.size()); }

  /// Total events ever allocated (including trimmed).
  int totalAllocated() const { return base_offset_ + static_cast<int>(entries_.size()); }

  /// Whether @p id names an event still present in this registry.
  bool contains(EventId id) const {
    return id.value >= base_offset_ ? id.value < totalAllocated()
                                    : completed_lds_.contains(id.value);
  }

  /// Number of events trimmed so far.
  int trimmedCount() const { return totalAllocated() - size(); }

  static constexpr int kTrimAttemptInterval = 1000;

private:
  const EventInfo &get(EventId id) const {
    if (id.value < base_offset_)
      return completed_lds_.at(id.value);
    return entries_[id.value - base_offset_];
  }
  EventInfo &get(EventId id) { return const_cast<EventInfo &>(std::as_const(*this).get(id)); }

  static bool isEntryTrimmable(const EventInfo &e) {
    if (e.status == EventStatus::RETIRED)
      return true;
    return e.status == EventStatus::WAVE_COMPLETE &&
           (isWaveLocal(e.type) || (e.ldsIntervals.empty() && !e.hasLdsSourceReads));
  }

  void tryTrimEvents() {
    int trimCount = 0;
    int size = static_cast<int>(entries_.size());
    while (trimCount < size && entries_[trimCount].status != EventStatus::ACTIVE)
      trimCount++;

    if (trimCount < size / 2)
      return;

    for (int i = 0; i < trimCount; ++i) {
      if (!isEntryTrimmable(entries_[i]))
        completed_lds_.emplace(base_offset_ + i, std::move(entries_[i]));
    }
    entries_.erase(entries_.begin(), entries_.begin() + trimCount);
    base_offset_ += trimCount;
  }

  std::vector<EventInfo> entries_;
  std::unordered_map<int, EventInfo> completed_lds_;
  int base_offset_ = 0;
};

} // namespace rocjitsu::plugins::race_detector
