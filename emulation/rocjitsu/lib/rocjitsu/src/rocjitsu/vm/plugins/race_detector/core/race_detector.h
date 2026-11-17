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
///      marks RETIRED. LDS events are retired at s_barrier via
///      flushBarrierPendingEvents, since other waves can still race with a
///      completed event until then. In a single-wave workgroup there is no other
///      wave, so an LDS event is retired as soon as it completes.
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
                    uint8_t atomicBytes = 0);

  /// Check for WAR and WAW hazards: no outstanding LDS reads or writes overlap
  /// the range. Same-wave operations are ordered only when both belong to the
  /// same non-UNORDERED completion class.
  ///
  /// A write over an unordered pending write is a race only if the two store
  /// different values and a read observes them. `data` points to the written
  /// bytes, or is null when they are unknown. A write whose bytes equal what the
  /// LDS reader returns for a pending store from VGPRs is reported as
  /// RaceViolation::Kind::RedundantWrite. Any other overlap becomes a conflict:
  /// a later read reports it, and a later write ordered after the pending writes
  /// clears it.
  void validateWrite(int addr, WaveId, int lane, int nBytes,
                     MemoryOrderClass currentMemoryOrder = MemoryOrderClass::LDS,
                     uint8_t atomicBytes = 0, const uint8_t *data = nullptr,
                     const InstructionSnapshot *instruction = nullptr);

  /// Reads LDS bytes as the emulator holds them when a write is validated: a DS
  /// write already applied, the write being validated not yet.
  using LdsReader = std::function<void(int addr, int nBytes, uint8_t *dst)>;
  void setLdsReader(LdsReader reader) { ldsReader = std::move(reader); }

  /// Two unordered writes to LDS bytes [start, end) that no read has observed.
  struct LdsWawConflict {
    int start;
    int end;
    InstructionSnapshot first;
    WaveId firstWave;
    std::optional<InstructionSnapshot> second;
    WaveId secondWave;
    int secondLane;
  };
  /// Disjoint conflicts keyed by start address.
  const std::map<int, LdsWawConflict> &getLdsWawConflicts() const { return wawConflicts; }

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

  /// A workgroup barrier covers all waves, including those that ended before
  /// it resolved. Retire their WAVE_COMPLETE events; ACTIVE events stay pending.
  void flushWorkgroupBarrierEvents() {
    for (auto &wave : waveRaceStates)
      wave.flushBarrierPendingEvents();
  }

private:
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
  /// The hazard between the current access and a pending event.
  enum class LdsHazard { ReadAfterWrite, WriteAfterRead, WriteAfterWrite };

  /// Report each event of `pending` that overlaps the access and is not
  /// ordered before it. For WriteAfterWrite, record a conflict or report a
  /// redundant write instead (see validateWrite). `contested` holds the parts
  /// of conflicts in the write's bytes before the write.
  void reportUnorderedOverlaps(const std::vector<EventId> &pending, LdsHazard hazard, int addr,
                               WaveId wave, int lane, int nBytes,
                               MemoryOrderClass currentMemoryOrder, uint8_t atomicBytes,
                               const uint8_t *data = nullptr,
                               const InstructionSnapshot *instruction = nullptr,
                               std::span<const LdsWawConflict> contested = {});
  /// Report and drop each conflict a read of [addr, addr + nBytes) observes.
  void reportObservedConflicts(int addr, int nBytes, WaveId wave, int lane);
  /// Drop the parts of conflicts inside [start, end).
  void clearConflicts(int start, int end);
  /// Split the conflict that straddles `at`, if any, so that no conflict does.
  void splitConflictAt(int at);
  bool overlapsConflict(int start, int end) const;
  /// The parts of conflicts inside [start, end), in address order.
  std::vector<LdsWawConflict> conflictsIn(int start, int end) const;
  /// Append to `spans` the byte ranges in [start, end) where LDS does not hold
  /// `data`.
  void differingFromLds(int start, int end, const uint8_t *data,
                        std::vector<Interval> &spans) const;

  std::map<int, LdsWawConflict> wawConflicts;
  LdsReader ldsReader;

  /// Active LDS write events (for scanning during read and write validation).
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
};

} // namespace rocjitsu::plugins::race_detector
