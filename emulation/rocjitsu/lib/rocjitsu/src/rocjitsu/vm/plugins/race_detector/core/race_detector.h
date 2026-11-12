// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/event_registry.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

/// Workgroup-level race detection state. Owns the event registry, live LDS
/// event lists, per-chunk byte counts, and per-wave WaveRaceStates.
///
/// Event lifecycle:
///   1. allocateEventId() — registers a new event (ACTIVE).
///   2. markEventWaveComplete() — transitions to WAVE_COMPLETE after a wait or
///      ordered counter-capacity backpressure satisfies all of the event's
///      wait-counter obligations.
///   3. retireEvent() — removes from live lists, decrements byte counts,
///      marks RETIRED. A completed LDS event is retired once every other live
///      wave has acquired it through a barrier (see Barrier ordering). In a
///      single-wave workgroup there is no other wave, so an LDS event is
///      retired as soon as it completes.
///
/// Barrier ordering: a barrier orders an access before it only if the issuing
/// wave completed the access before its own signal, and orders an access after
/// it only if the accessing wave issued it after its own wait. Each round of a
/// barrier is a release/acquire channel, tracked with a vector clock over the
/// workgroup's waves:
///   - Each wave has an epoch, advanced by each of its signals. An event that
///     completes records its wave's current epoch (EventRegistry::releaseEpoch).
///   - A signal releases into the round everything the wave has completed or
///     acquired; passing the wait of a round acquires everything released into
///     it. A wave that ends releases into every later workgroup round, which
///     cannot complete without it (CDNA5 ISA 5.6).
///   - Another wave's completed event is ordered for a wave that has acquired
///     its epoch; it is retired once every live wave has.
/// A single s_barrier is reported as a round whose signals and waits coincide.
///
/// LDS race validation uses a two-level approach:
///   - Fast path: per-chunk byte counts (byteWriteCounts / byteReadCounts)
///     skip empty chunks and chunks owned by the accessing wave's
///     ordered LDS events.
///   - Slow path: when a chunk may contain a conflict, scans live event intervals
///     with binary search (IntervalSet::overlapsRange).
class RaceDetector {
public:
  /// ldsSizeBytes is the CU backing capacity, independent of workgroup
  /// allocation metadata and the lazily materialized byte count.
  RaceDetector(int nWaves, int vgprCount, int sgprCount, Dim3d workgroupId,
               std::function<void(RaceViolation)> raceHandler, CounterCapacities,
               size_t ldsSizeBytes);

  /// Allocate an event using the counter and ordering implied by its type.
  EventId allocateEventId(WaveId wave, InstructionSnapshot instruction, MemoryEventType type,
                          std::vector<uint32_t> registers, uint64_t execMask) {
    return allocateEventId(wave, instruction, type, std::move(registers), execMask, 0xF, {},
                           CounterObligations(type), defaultMemoryOrder(type));
  }

  /// Allocate a workgroup-global event ID and record its metadata.
  EventId allocateEventId(WaveId, InstructionSnapshot instruction, MemoryEventType,
                          std::vector<uint32_t> registers, uint64_t execMask, uint8_t byteMask,
                          IntervalSet ldsIntervals,
                          std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                          MemoryOrderClass memoryOrder, uint8_t lastRegisterByteMask = 0,
                          LdsAccess ldsAccess = {});

  /// Transition an event from ACTIVE to WAVE_COMPLETE.
  void markEventWaveComplete(EventId);

  /// Satisfy the event's obligations covered by one wait-counter update.
  /// Returns true once every obligation for the event has been satisfied.
  bool satisfyEventWaitCounter(EventId, amdgpu::WaitCounterType);

  /// Retire an event. For LDS-touching events, removes from live lists
  /// and decrements per-byte counts. All events are marked RETIRED in the
  /// registry, enabling prefix trimming.
  void retireEvent(EventId);

  /// Check for RAW hazards: no outstanding LDS writes overlap the range.
  /// Same-wave operations are ordered only when both belong to the same
  /// non-UNORDERED completion class.
  void validateRead(int addr, WaveId, int lane, int nBytes,
                    MemoryOrderClass currentMemoryOrder = MemoryOrderClass::LDS,
                    uint8_t atomicBytes = 0) const;

  /// Check for WAR hazards: no outstanding LDS reads overlap the range.
  /// Same-wave operations are ordered only when both belong to the same
  /// non-UNORDERED completion class.
  /// TODO(newling): WAW detection (write vs outstanding writes) is not
  /// implemented.
  void validateWrite(int addr, WaveId, int lane, int nBytes,
                     MemoryOrderClass currentMemoryOrder = MemoryOrderClass::LDS,
                     uint8_t atomicBytes = 0) const;

  /// @brief Conservatively check an instruction's LDS intervals for conflicts.
  /// @details The intervals must describe valid LDS byte ranges. False proves
  /// that validateRead (or validateWrite, when isWrite) would report nothing
  /// for any access by `wave` within `accessed`, so per-lane validation can be
  /// skipped. Visits the 16-byte chunks of each interval, without scanning gaps
  /// between intervals. Separate intervals can visit the same chunk.
  bool mayConflict(const IntervalSet &accessed, WaveId wave, bool isWrite,
                   MemoryOrderClass currentMemoryOrder) const;

  /// True only when the complete access lies in the CU's logical LDS backing.
  bool validLdsRange(uint64_t address, uint64_t bytes) const {
    return bytes != 0 && address < ldsSizeBytes_ && bytes <= ldsSizeBytes_ - address;
  }

  /// Bytes in complete elements that fit the LDS backing. Vector transfers
  /// can retain a valid prefix when later elements are dropped as out of bounds.
  uint64_t ldsBytesInRange(uint64_t address, uint64_t bytes, uint64_t elementBytes) const {
    if (elementBytes == 0 || !validLdsRange(address, elementBytes))
      return 0;
    return std::min(bytes, ldsSizeBytes_ - address) / elementBytes * elementBytes;
  }

  const EventRegistry &events() const { return events_; }

  const std::vector<EventId> &getLdsWriteEvents() const { return ldsWriteEvents; }
  const std::vector<EventId> &getLdsReadEvents() const { return ldsReadEvents; }

  Dim3d getWorkgroupId() const { return workgroupId; }
  const std::function<void(RaceViolation)> &getRaceHandler() const { return raceHandler; }

  WaveRaceState &getWaveRaceState(int waveIndex);

  /// Whether the workgroup has a single wave, so that no other wave can race
  /// with an event its own wave has completed.
  bool hasSingleWave() const { return waveRaceStates.size() == 1; }

  // -- Barrier ordering --

  /// `wave`'s signal was counted toward `round`: release into the round
  /// everything the wave has completed or acquired.
  void barrierSignaled(WaveId wave, BarrierRoundId round);

  /// `round` completed, notifying `members`. `wholeWorkgroup` is true for the
  /// workgroup barriers, which cannot complete until every wave has signalled
  /// or ended, so they also release the events of waves that ended.
  void barrierRoundCompleted(BarrierRoundId round, std::span<const WaveId> members,
                             bool wholeWorkgroup);

  /// `wave` passed its wait for `round`: acquire everything released into it.
  void barrierWaitPassed(WaveId wave, BarrierRoundId round);

  /// `wave` ended. Its implicit end-of-program wait has already been applied.
  void waveEnded(WaveId wave);

  /// Rounds that have not completed, or that a member has yet to wait for.
  size_t openBarrierRounds() const { return rounds_.size(); }

private:
  struct BarrierRoundState {
    /// Per wave, the highest epoch released into the round.
    std::vector<uint32_t> clock;
    /// Members that have not yet passed the round's wait.
    std::vector<bool> awaiting;
    bool completed = false;
  };

  int waveCount() const { return static_cast<int>(waveRaceStates.size()); }

  /// Highest epoch of `owner` that `accessor` has acquired. A wave's own
  /// entry is its current epoch.
  uint32_t &acquired(WaveId accessor, WaveId owner) {
    return acquired_[static_cast<size_t>(accessor.value) * waveCount() + owner.value];
  }
  uint32_t acquired(WaveId accessor, WaveId owner) const {
    return acquired_[static_cast<size_t>(accessor.value) * waveCount() + owner.value];
  }

  /// Whether another wave's completed event is ordered before `accessor`'s
  /// access.
  bool ordered(WaveId accessor, EventId) const;

  BarrierRoundState &roundState(BarrierRoundId round);

  /// `wave` will not wait for any completed round of `barrier`, or of any
  /// barrier when `barrier` is empty. Rounds no member waits for are dropped.
  void stopAwaiting(WaveId wave, std::optional<uint8_t> barrier);

  /// Retire each wave's completed events that every other live wave has
  /// acquired.
  void retireOrdered();

  struct LdsChunkCounts {
    int bytes = 0;
    /// When bytes > 0, a nonnegative owner proves that every counted event
    /// is an ordinary LDS access from this wave. -1 makes no ownership claim.
    /// Removing events preserves a known owner; a mixed chunk stays
    /// conservative until its count drains, even if only one wave remains.
    WaveId orderedWave{-1};
  };

  void adjustByteCounts(const IntervalSet &ivs, std::vector<LdsChunkCounts> &counts, int delta,
                        WaveId orderedWave = WaveId{-1});
  static bool hasPotentialLdsConflict(int addr, int nBytes,
                                      const std::vector<LdsChunkCounts> &counts, WaveId wave,
                                      MemoryOrderClass currentMemoryOrder);

  /// Active LDS write events (for scanning during read validation).
  std::vector<EventId> ldsWriteEvents;

  /// Active LDS read events (for scanning during write validation).
  std::vector<EventId> ldsReadEvents;

  /// Outstanding write/read counts at chunk granularity for fast-path
  /// validation. Indexed by byte address / kCountGranularity. Conservative:
  /// may over-report conflicts for sub-chunk accesses. Vectors start empty
  /// and grow only within the CU backing capacity as LDS events arrive.
  static constexpr int kCountGranularity = 16;
  /// Sole-owner tracking avoids rescanning a long loop's own ordered LDS
  /// history, using one wave ID per chunk rather than per-wave counters.
  /// Events remain live for other waves until a barrier retires them.
  std::vector<LdsChunkCounts> byteWriteCounts;
  std::vector<LdsChunkCounts> byteReadCounts;

  size_t ldsSizeBytes_;
  EventRegistry events_;

  Dim3d workgroupId;
  std::function<void(RaceViolation)> raceHandler;

  /// One WaveRaceState per wave, indexed by wave ID.
  std::vector<WaveRaceState> waveRaceStates;

  /// waveCount() x waveCount() vector clock, row-major by acquiring wave.
  std::vector<uint32_t> acquired_;
  /// The join of the clocks of the waves that ended.
  std::vector<uint32_t> exitClock_;
  std::vector<bool> ended_;
  std::map<BarrierRoundId, BarrierRoundState> rounds_;
};

} // namespace rocjitsu::plugins::race_detector
