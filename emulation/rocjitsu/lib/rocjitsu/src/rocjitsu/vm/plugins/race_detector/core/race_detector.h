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
///      wave has acquired it, through a barrier or an LDS barrier cell (see
///      Barrier ordering). When no other wave of the workgroup is live, an LDS
///      event is retired as soon as it completes.
///
/// Barrier ordering: for a direct handoff, the detector requires the issuing
/// wave to complete its LDS access before signalling, and the accessing wave
/// to pass its wait before accessing those bytes. Each round of a
/// barrier is a release/acquire channel, tracked with a vector clock over the
/// workgroup's waves:
///   - Each wave has an epoch, advanced by each of its signals and each of its
///     arrivals at an LDS barrier cell. An event records that epoch when its
///     LDS access completes (EventRegistry::releaseEpoch), even if its
///     register result remains pending.
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
/// LDS barrier cells (CDNA5 ISA 11.2.2) feed the same clock. A cell is a 64-bit
/// LDS word that waves arrive at and poll instead of blocking at a barrier:
///   - An arrival releases into the cell's phase in progress, as a signal
///     releases into a round.
///   - An arrival also hands off the wave's in-flight LDS accesses that the
///     arrive is ordered after (WaveRaceState::handOffPending): they have
///     executed in LDS by the time the arrival does, although their wave has
///     not waited for them. A barrier signal, and an arrival of another kind,
///     is not ordered after them, so a handoff is not released on the wave's
///     epoch. Each wave has a handoff epoch per kind of arrival, advanced by
///     its arrivals of that kind; a handed-off event records the kind and that
///     epoch, and only the wave's arrivals of that kind release it. A wave
///     that acquired a handoff passes it on in everything it releases, since
///     it read the cell after the arrival. A handed-off event is ordered for
///     the waves that acquire its handoff, while its access is in flight and
///     after it completes; for its own wave nothing changes. It is retired
///     like any other event, once every other live wave has acquired the epoch
///     in which its LDS access completed.
///   - The arrival that rolls the cell's pending count under completes the
///     phase. A wave that reads the cell acquires every completed phase of
///     it, whether or not it then tests the phase.
///   - An arrival updates the cell atomically, so it conflicts only with
///     non-atomic writes of the cell, never with reads of it.
///
/// LDS race validation uses a two-level approach:
///   - Fast path: per-chunk byte counts (byteWriteCounts / byteReadCounts)
///     skip empty chunks and chunks owned by the accessing wave's
///     ordered LDS events.
///   - Slow path: when a chunk may contain a conflict, scans the intervals
///     of the events whose LDS access is in flight, and of each other wave's
///     completed events that the accessing wave has not acquired through a
///     barrier or an LDS barrier cell.
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
  /// barrier epoch. The wave's next barrier signal or barrier-cell arrival
  /// releases it to other waves.
  void stampReleaseEpoch(EventId);

  /// Hand an event whose LDS access is in flight off to other waves, at an LDS
  /// barrier arrival of `kind` in its wave's handoff epoch `epoch` of that
  /// kind.
  void markEventHandedOff(EventId eventId, BarrierCellArrival kind, uint32_t epoch) {
    events_.markHandedOff(eventId, kind, epoch);
  }

  /// The index events use to refer to the barrier instruction `barrier`
  /// (see EventRegistry::barrierIndex()).
  uint16_t barrierIndex(const InstructionSnapshot &barrier) {
    return events_.barrierIndex(barrier);
  }
  /// Record that an event's LDS access was in flight when its wave signalled
  /// the barrier `barrier`, an index from barrierIndex(), in the wave's
  /// barrier epoch `epoch`.
  void markBarrierCrossedInFlight(EventId eventId, uint16_t barrier, uint32_t epoch) {
    events_.markBarrierCrossedInFlight(eventId, barrier, epoch);
  }

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
  /// An arrival at an LDS barrier cell reads the cell it updates, and is
  /// checked here like any other atomic.
  void validateRead(int addr, WaveId, int lane, int nBytes,
                    MemoryOrderClass currentMemoryOrder = MemoryOrderClass::LDS,
                    uint8_t atomicBytes = 0) const;

  /// Check for WAR hazards: no outstanding LDS reads overlap the range.
  /// Same-wave operations are ordered only when both belong to the same
  /// non-UNORDERED completion class.
  /// TODO(newling): WAW detection (write vs outstanding writes) is not
  /// implemented.
  /// An arrival at an LDS barrier cell updates the cell atomically, so it
  /// races with no read and is not checked here.
  void validateWrite(int addr, WaveId, int lane, int nBytes,
                     MemoryOrderClass currentMemoryOrder = MemoryOrderClass::LDS,
                     uint8_t atomicBytes = 0) const;

  /// @brief Check an instruction's LDS intervals for conflicts, once for all
  /// its lanes.
  /// @details The intervals must describe valid LDS byte ranges. False proves
  /// that validateRead (or validateWrite, when isWrite) would report nothing
  /// for any access by `wave` within `accessed`, so per-lane validation can be
  /// skipped. First visits the 16-byte chunks of each interval, without
  /// scanning gaps between intervals; when a chunk has live bytes, scans the
  /// events the per-lane checks would scan, by the same rules, against the
  /// whole of `accessed`.
  /// `atomicBytes` is nonzero only when every lane of the access is a naturally
  /// aligned atomic of that width, as validateRead and validateWrite take it
  /// for one lane. With zero no event is exempted as an atomic of matching
  /// width, so true can be returned for an access whose lanes all pass.
  bool mayConflict(const IntervalSet &accessed, WaveId wave, bool isWrite,
                   MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes = 0) const;

  /// SRC2 reads a second LDS operand and updates its first address atomically.
  /// Both ranges share the issuing instruction's wait and barrier lifetime.
  void attachSrc2Source(EventId id, IntervalSet sourceReads);

  /// A tensor transfer whose descriptor requests it arrives at the barrier
  /// cell at `cell` when the transfer completes (CDNA5 ISA 11.2.2), updating
  /// the cell's 64 bits atomically. The cell shares the transfer's wait and
  /// barrier lifetime, so the arrival takes no TENSORcnt slot of its own. A
  /// write of the cell that is not an atomic of the cell's width races with
  /// it until the transfer is ordered before the write.
  void attachBarrierArrive(EventId transfer, uint32_t cell);

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
  /// everything the wave has completed or acquired. `barrier` is the
  /// instruction that signalled, when the caller knows it. The signal does not
  /// release what the wave still has in flight, so `barrier` and the wave's
  /// epoch are recorded on each of its LDS accesses in flight; a race between
  /// such an access and a wave ordered after this signal names the barrier (see
  /// RaceViolation::crossedBarrier). Nothing is recorded in a single-wave
  /// workgroup, which has no other wave to race with.
  void barrierSignaled(WaveId wave, BarrierRoundId round,
                       const InstructionSnapshot *barrier = nullptr);

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

  // -- LDS barrier cells (CDNA5 ISA 11.2.2) --

  /// `wave` arrived at the barrier cell at LDS byte `address`; `completed` if
  /// the arrival changed the cell's phase. The arrival releases into the
  /// cell's phase in progress everything the wave has completed or acquired,
  /// and hands off the in-flight accesses `kind` is ordered after. A Ds
  /// arrival returns the cell, so it also reads it.
  void barrierCellArrived(WaveId wave, uint32_t address, BarrierCellArrival kind, bool completed);

  /// `wave` read the LDS bytes `accessed`: acquire every completed phase of
  /// each barrier cell with a byte of its state dword among them. Does nothing
  /// while no wave has arrived at a cell.
  void acquireBarrierCells(WaveId wave, const IntervalSet &accessed);

  /// Cells some wave has arrived at. A cell is kept for the life of the
  /// workgroup.
  size_t barrierCellCount() const { return cells_.size(); }

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
    /// The clock released into the round.
    std::vector<uint32_t> clock;
  };

  int waveCount() const { return static_cast<int>(waveRaceStates.size()); }

  /// Highest epoch of `owner` that `accessor` has acquired. A wave's own
  /// entry is its current epoch.
  uint32_t &acquired(WaveId accessor, WaveId owner) {
    return acquired_[static_cast<size_t>(accessor.value) * clockSize() + owner.value];
  }
  uint32_t acquired(WaveId accessor, WaveId owner) const {
    return acquired_[static_cast<size_t>(accessor.value) * clockSize() + owner.value];
  }

  /// Whether `matches` holds for any event an LDS access by `accessor` in
  /// `currentMemoryOrder` has to be checked against: the events whose LDS
  /// access is in flight, except the accessor's own that are ordered with the
  /// access, and each event of another wave whose LDS access completed in an
  /// epoch `accessor` has not acquired. A wave's history is in epoch order, so
  /// those are a suffix of it, and the events a barrier has ordered before the
  /// access are never visited, nor is a history whose bounds are disjoint
  /// from [start, end), which must cover the access. A write is checked
  /// against reads and a read against writes. Stops at the first event for
  /// which `matches` returns true.
  template <typename Matches>
  bool anyCheckedLdsEvent(WaveId accessor, bool isWrite, MemoryOrderClass currentMemoryOrder,
                          int start, int end, Matches matches) const;

  /// Whether `matches` holds for any attached LDS read that a write by
  /// `accessor` has to be checked against: all but the accessor's own that are
  /// ordered with the write, and another wave's that a barrier has ordered
  /// before it or that was handed off to the accessor. Stops at the first read
  /// for which `matches` returns true.
  void attachLdsRead(EventId id, IntervalSet bytes, bool barrierArrive);

  template <typename Matches>
  bool anyCheckedAttachedRead(WaveId accessor, MemoryOrderClass currentMemoryOrder,
                              Matches matches) const;

  /// Whether a barrier has ordered another wave's completed event before
  /// `accessor`'s access.
  bool ordered(WaveId accessor, EventId) const;

  /// `violation`, a race between `accessor` and the LDS access of `eventId`,
  /// with the barrier that access crossed in flight and that `accessor` has
  /// passed, if another wave owns the access and there is such a barrier.
  RaceViolation withCrossedBarrier(RaceViolation violation, WaveId accessor, EventId eventId) const;

  /// Whether `accessor` has acquired the handoff of another wave's event at an
  /// LDS barrier arrival, which orders the event's LDS access before
  /// `accessor`'s whether or not that access has completed.
  bool handedOffTo(WaveId accessor, EventId) const;

  BarrierRoundState *findRound(BarrierRoundId round);
  /// The open slot for `round`, opening one if needed.
  BarrierRoundState &roundState(BarrierRoundId round);
  void closeRound(BarrierRoundState &state);

  /// `wave` will not wait for any completed round of `barrier`, or of any
  /// barrier when `barrier` is empty. Rounds no member waits for are dropped.
  void stopAwaiting(WaveId wave, std::optional<uint8_t> barrier);

  /// Raise `accessor`'s acquired epoch and handoff epochs of each owner to at
  /// least `clock`'s, retiring the events that every other live wave has now
  /// acquired.
  void acquire(WaveId accessor, std::span<const uint32_t> clock);

  /// Join into `clock` what `wave` releases: the epoch and handoff epochs it
  /// has acquired of every wave, and its own epoch. Of its own handoff epochs
  /// it releases only that of `arrival`, the kind of LDS barrier arrival
  /// making the release, if any: a barrier signal, the wave's end and an
  /// arrival of another kind are not ordered after the accesses handed off.
  void joinReleased(std::span<uint32_t> clock, WaveId wave,
                    std::optional<BarrierCellArrival> arrival = std::nullopt) const;

  /// One LDS barrier cell some wave has arrived at.
  struct BarrierCell {
    uint32_t address = 0;
    /// The clock released into the phase in progress.
    std::vector<uint32_t> open;
    /// The clock released into the completed phases. Phases complete in order,
    /// so a reader of one has every earlier one behind it.
    std::vector<uint32_t> done;
  };

  /// The phase is in the cell's low dword: bits [31:WIDTH] (CDNA5 ISA 11.2.2,
  /// Table 70). A read of any byte of that dword is taken to observe it.
  static constexpr int kBarrierCellStateBytes = 4;

  BarrierCell &barrierCell(uint32_t address);

  // A clock has one section of waveCount() entries for the waves' epochs,
  // then one for their handoff epochs of each kind of LDS barrier arrival.
  static constexpr int kArrivalKinds = 3;
  static_assert(static_cast<int>(BarrierCellArrival::Tensor) == kArrivalKinds - 1,
                "a clock has one handoff section per BarrierCellArrival");
  int clockSize() const { return (1 + kArrivalKinds) * waveCount(); }
  /// Index in a clock of `owner`'s handoff epoch of `kind`.
  int handoffIndex(BarrierCellArrival kind, WaveId owner) const {
    return (1 + static_cast<int>(kind)) * waveCount() + owner.value;
  }

  std::span<uint32_t> acquiredRow(WaveId wave) {
    return std::span(acquired_).subspan(static_cast<size_t>(wave.value) * clockSize(), clockSize());
  }
  std::span<const uint32_t> acquiredRow(WaveId wave) const {
    return std::span(acquired_).subspan(static_cast<size_t>(wave.value) * clockSize(), clockSize());
  }

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

  /// LDS bytes an event reads besides its own intervals: the source operand of
  /// an SRC2 instruction, or the barrier cell a tensor transfer arrives at.
  /// They share the event's wait and barrier lifetime. Scanned whole during
  /// write validation: they are few, and stay here until the event retires.
  struct AttachedLdsRead {
    EventId id;
    IntervalSet bytes;
    /// The bytes are a barrier cell, which the event updates atomically.
    bool barrierArrive = false;
  };
  std::vector<AttachedLdsRead> attachedLdsReads;

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

  /// One clock of clockSize() entries per acquiring wave, row-major: what the
  /// wave has acquired of each wave. A wave's own entries are its current
  /// epoch and handoff epochs.
  std::vector<uint32_t> acquired_;
  /// Per owner, the lowest epoch of it that every other live wave has
  /// acquired; its completed events up to that epoch are retired.
  std::vector<uint32_t> covered_;
  /// Per owner, how many other live waves have acquired exactly covered_.
  std::vector<int> atCoverage_;
  /// Owners whose last wave at covered_ ended. Their coverage is recomputed,
  /// and what it covers retired, at the next barrier wait or read of an LDS
  /// barrier cell rather than when the wave ends: the wave's halt is reported under the plugin
  /// group's lock, and at the end of a workgroup nothing remains to benefit. Until then the covered
  /// events are still checked, and skipped as ordered.
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
  /// Kept for the life of the workgroup and searched linearly, by each arrival
  /// and by each LDS load once a cell exists: a kernel has a handful.
  std::vector<BarrierCell> cells_;
};

} // namespace rocjitsu::plugins::race_detector
