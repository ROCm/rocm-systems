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
///      wave has acquired it through a barrier (see Barrier ordering). When
///      no other wave of the workgroup is live, an LDS event is retired as
///      soon as it completes.
///
/// Barrier ordering: for a direct handoff, the detector requires the issuing
/// wave to complete its LDS access before signalling, and the accessing wave
/// to pass its wait before accessing those bytes. Each round of a
/// barrier is a release/acquire channel, tracked with a vector clock over the
/// workgroup's waves:
///   - Each wave has an epoch, advanced by each of its signals. An event records
///     that epoch when its LDS access completes (EventRegistry::releaseEpoch),
///     even if its register result remains pending.
///   - A signal releases into the round everything the wave has completed or
///     acquired; passing the wait of a round acquires everything released into
///     it. A wave that ends releases into every later workgroup round it had
///     not signalled, which cannot complete until it ends (CDNA5 ISA 5.6). A
///     round it had signalled can complete without its end, and orders only
///     what that signal released.
///   - Another wave's completed event is ordered for a wave that has acquired
///     its epoch; it is retired once every live wave has.
/// A single s_barrier is reported as a round whose signals and waits coincide.
///
/// LDS race validation uses a two-level approach:
///   - Fast path: per-chunk byte counts (byteWriteCounts / byteReadCounts)
///     skip empty chunks and chunks owned by the accessing wave's
///     ordered LDS events.
///   - Slow path: when a chunk may contain a conflict, scans the intervals
///     of the events whose LDS access is in flight, and of each other wave's
///     completed events that the accessing wave has not acquired through a
///     barrier.
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
                          VgprWriteOrder writeOrder, uint8_t lastRegisterByteMask = 0,
                          LdsAccess ldsAccess = {});

  /// Attach the LDS footprint after a mixed FLAT issue successfully translates its global lanes.
  void attachLdsIntervals(EventId event, IntervalSet intervals);

  /// Transition an event from ACTIVE to WAVE_COMPLETE.
  void markEventWaveComplete(EventId);

  /// Record that an event's LDS access has completed, in its wave's current
  /// barrier epoch. The wave's next barrier signal releases it to other waves.
  void stampReleaseEpoch(EventId);

  /// Satisfy the event's obligations covered by one wait-counter update.
  /// Returns true once every obligation for the event has been satisfied.
  bool satisfyEventWaitCounter(EventId, amdgpu::WaitCounterType);
  bool appendCounterOnly(EventId id, amdgpu::MemoryCounterObligation issue,
                         EventRegistry::RoutedChanges *changes) {
    return events_.appendCounterOnly(id, issue, changes);
  }

  /// Retire an event. For LDS-touching events, removes from live lists
  /// and decrements per-byte counts. All events are marked RETIRED in the
  /// registry, enabling prefix trimming.
  void retireEvent(EventId);
  /// Publish a completed LDS portion at a workgroup barrier. Its VGPR result
  /// and other counter obligations may still be active.
  void retireLdsIntervals(EventId);

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

  /// SRC2 reads a second LDS operand and updates its first address atomically.
  /// Both ranges share the issuing instruction's wait and barrier lifetime.
  void attachSrc2Source(EventId id, IntervalSet sourceReads);

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

  void restoreRoutedChanges(const EventRegistry::RoutedChanges &changes) {
    events_.restoreRoutedChanges(changes);
  }
  void supersedeFlatResult(EventId id, VgprWriteOrder replacement, uint32_t reg, uint64_t lanes,
                           uint8_t bytes, EventRegistry::RoutedChanges *changes) {
    events_.supersedeFlatResult(id, replacement, reg, lanes, bytes, changes);
  }
  void setConditionalVgprWrite(EventId id, ConditionalVgprWrite feedback) {
    events_.setConditionalVgprWrite(id, feedback);
  }
  const EventRegistry &events() const { return events_; }

  /// LDS events whose bytes another wave's access can still race with.
  size_t liveLdsWriteEvents() const { return liveLdsWriteEvents_; }
  size_t liveLdsReadEvents() const { return liveLdsReadEvents_; }

  Dim3d getWorkgroupId() const { return workgroupId; }
  const std::function<void(RaceViolation)> &getRaceHandler() const { return raceHandler; }

  WaveRaceState &getWaveRaceState(int waveIndex);

  /// Whether every other wave of the workgroup has ended, or there is none,
  /// so that no other wave can race with an event `wave` completes.
  bool isOnlyLiveWave(WaveId wave) const { return liveWaves_ == 1 && !ended_[wave.value]; }

  // -- Barrier ordering --

  /// `wave`'s signal was counted toward `round`: release into the round
  /// everything the wave has completed or acquired.
  void barrierSignaled(WaveId wave, BarrierRoundId round);

  /// `round` completed. `armed` lists the waves whose completion bit the
  /// round set, whose next wait on this barrier it releases; not every wave
  /// it notified. A notified wave whose bit an earlier round left set is not
  /// armed: its wait passes for that round, and acquires that round.
  /// `wholeWorkgroup` is true for the workgroup barriers, which cannot
  /// complete until every wave has signalled or ended, so they also release
  /// the events of waves that ended.
  void barrierRoundCompleted(BarrierRoundId round, std::span<const WaveId> armed,
                             bool wholeWorkgroup);

  /// `wave` passed its wait for `round`: acquire everything released into it.
  void barrierWaitPassed(WaveId wave, BarrierRoundId round);

  /// `wave` ended. Its implicit end-of-program wait has already been applied.
  void waveEnded(WaveId wave);

  /// Rounds that have not completed, or that a member has yet to wait for.
  size_t openBarrierRounds() const { return openRounds_; }

  /// The most waves a workgroup can have; barrier rounds keep a bit per wave.
  static constexpr int kMaxWaves = 64;

private:
  void indexLdsIntervals(EventId event);

  /// A slot for one round. Closed slots keep their storage for reuse, since a
  /// workgroup typically has one or two rounds open at a time.
  struct BarrierRoundState {
    BarrierRoundId id;
    bool open = false;
    bool completed = false;
    /// Bit per member that has not yet passed the round's wait.
    uint64_t awaiting = 0;
    /// Bit per wave whose signal was counted toward the round.
    uint64_t signalled = 0;
    /// Per wave, the highest epoch released into the round.
    std::vector<uint32_t> clock;
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

  /// Call `visit` with candidate events of other waves whose LDS access
  /// completed in an epoch `accessor` has not acquired. Histories with bounds
  /// disjoint from [start, end) are skipped. A wave's history is in epoch
  /// order, so these are a suffix of it, and the events a barrier has ordered
  /// before `accessor`'s access are never visited.
  template <typename Visit>
  void forEachUnacquiredLdsEvent(WaveId accessor, int start, int end, Visit visit) const;

  /// Whether a barrier has ordered another wave's completed event before
  /// `accessor`'s access.
  bool ordered(WaveId accessor, EventId) const;

  BarrierRoundState *findRound(BarrierRoundId round);
  /// The open slot for `round`, opening one if needed.
  BarrierRoundState &roundState(BarrierRoundId round);
  void closeRound(BarrierRoundState &state);

  /// `wave` will not wait for any completed round of `barrier`, or of any
  /// barrier when `barrier` is empty. Rounds no member waits for are dropped.
  void stopAwaiting(WaveId wave, std::optional<uint8_t> barrier);

  /// Raise `accessor`'s acquired epoch of each owner to at least `clock`'s,
  /// retiring the events that every other live wave has now acquired.
  void acquire(WaveId accessor, std::span<const uint32_t> clock);

  /// A wave at `owner`'s coverage minimum raised its entry, so `owner`'s
  /// events it was holding back may now be retirable.
  void leaveCoverageMinimum(WaveId owner);

  /// Recompute `owner`'s coverage from the live waves and retire what it
  /// covers.
  void recomputeCoverage(WaveId owner);

  /// Recompute the coverage that waves ending left stale.
  void refreshStaleCoverage();

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

  /// Events that read LDS as a source operand, scanned whole during write
  /// validation: they are few, and stay here until they retire.
  struct Src2Event {
    EventId id;
    IntervalSet sourceReads;
  };
  std::vector<Src2Event> src2Events;

  /// LDS write events whose LDS access is in flight (for scanning during read
  /// validation). Once the access completes the event is in its wave's
  /// history instead, in the order of the barrier epochs that release it.
  std::vector<EventId> ldsWriteEvents;

  /// LDS read events whose LDS access is in flight (for scanning during write
  /// validation).
  std::vector<EventId> ldsReadEvents;

  /// LDS events with live bytes, in flight or in a wave's history.
  size_t liveLdsWriteEvents_ = 0;
  size_t liveLdsReadEvents_ = 0;

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
  /// Per owner, the lowest epoch of it that every other live wave has
  /// acquired; its completed events up to that epoch are retired.
  std::vector<uint32_t> covered_;
  /// Per owner, how many other live waves have acquired exactly covered_.
  std::vector<int> atCoverage_;
  /// Owners whose last wave at covered_ ended. Their coverage is recomputed,
  /// and what it covers retired, at the next barrier wait rather than when
  /// the wave ends: the wave's halt is reported under the plugin group's
  /// lock, and at the end of a workgroup nothing remains to benefit. Until
  /// then the covered events are still checked, and skipped as ordered.
  std::vector<bool> staleCoverage_;
  bool hasStaleCoverage_ = false;
  int liveWaves_ = 0;
  /// The join of the clocks of the waves that ended.
  std::vector<uint32_t> exitClock_;
  /// Bit per wave that ended.
  uint64_t endedWaves_ = 0;
  std::vector<bool> ended_;
  std::vector<BarrierRoundState> rounds_;
  size_t openRounds_ = 0;
};

} // namespace rocjitsu::plugins::race_detector
