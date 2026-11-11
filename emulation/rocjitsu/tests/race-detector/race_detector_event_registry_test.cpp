// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race_test_instruction.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>

using namespace rocjitsu::plugins::race_detector;

TEST(RaceDetector, EventRegistry_TrimmedWaveLocalEventsAreNotBarrierQueued) {
  // Counter-only events can be trimmed after completion; they carry no LDS
  // evidence for peers. Queuing them for a barrier would leave stale EventIds
  // after trimming.
  int N = EventRegistry::kTrimAttemptInterval;
  RaceDetector detector(/*nWaves=*/
                        1, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4),
                        /*ldsSizeBytes=*/64 * 1024);
  auto &wave = detector.getWaveRaceState(0);

  // Complete each counter-only issue immediately and trigger registry trimming.
  for (int i = 0; i < N; ++i) {
    wave.registerCounterEvent(
        testInstruction(static_cast<uint64_t>(i)),
        CounterObligations{rocjitsu::amdgpu::WaitCounterType::VMCNT, MemoryOrderClass::VMEM});
    PendingWaitCount wait;
    wait.add(rocjitsu::amdgpu::WaitCounterType::VMCNT, 0);
    wave.dispatch(wait);
  }

  EXPECT_GT(detector.events().trimmedCount(), 0);
  // If wave-local events were incorrectly queued for barrier retirement, this
  // would be non-empty. Flushing it after trimming would then touch stale ids.
  EXPECT_TRUE(wave.getBarrierPendingEvents().empty());
  wave.flushBarrierPendingEvents();
}

TEST(RaceDetector, EventRegistry_LdsEventsWithoutBytesAreTrimmedOnceComplete) {
  // An LDS event can touch no LDS bytes: an async copy with no active lanes, or
  // a tensor transfer that moves no data. It still holds a counter slot, but no
  // other wave can race with it, so once its wait completes it is trimmable
  // like a wave-local event, and must not be queued for barrier retirement.
  // The workgroup has two waves, so completed LDS events are not retired
  // immediately as in a single-wave workgroup: only the trim rule for events
  // without LDS bytes lets these go.
  int N = EventRegistry::kTrimAttemptInterval;
  RaceDetector detector(/*nWaves=*/
                        2, /*vgprCount=*/2, /*sgprCount=*/2, Dim3d(0), [](RaceViolation) {},
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5),
                        /*ldsSizeBytes=*/64 * 1024);
  auto &wave = detector.getWaveRaceState(0);
  const std::array obligations{rocjitsu::amdgpu::MemoryCounterObligation{
      rocjitsu::amdgpu::WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_STORE}};

  for (int i = 0; i < N; ++i) {
    wave.registerLdsRangeEvent(testInstruction(/*pc=*/static_cast<uint64_t>(i)),
                               LdsAccess::Read, /*ranges=*/{}, obligations,
                               MemoryOrderClass::ASYNC_STORE);
    PendingWaitCount wait;
    wait.add(rocjitsu::amdgpu::WaitCounterType::ASYNCCNT, 0);
    wave.dispatch(wait);
  }

  // The trim attempt runs after event N - 1 is added, when every earlier event
  // is complete and empty.
  EXPECT_EQ(detector.events().trimmedCount(), N - 1);
  EXPECT_TRUE(wave.getBarrierPendingEvents().empty());
  EXPECT_TRUE(detector.getLdsReadEvents().empty());
}

namespace {
using namespace rocjitsu;
using amdgpu::MemoryCounterObligation;
using amdgpu::WaitCounterType;

constexpr std::array<uint32_t, 1> kWords{0xbf800007};
constexpr InstructionSnapshot kInstruction{0x1234, kWords};
constexpr std::array kObligations{
    MemoryCounterObligation{WaitCounterType::LGKMCNT, MemoryOrderClass::UNORDERED}};

TEST(InstructionSnapshotTest, CopiesEveryEncodingWidth) {
  constexpr std::array<uint32_t, 5> expected{1, 2, 3, 4, 5};
  static_assert(expected.size() == InstructionSnapshot::kMaxWords);
  for (size_t count = 1; count <= InstructionSnapshot::kMaxWords; ++count) {
    auto source = expected;
    const InstructionSnapshot instruction(0x10, std::span(source).first(count));
    source.fill(0);
    EXPECT_EQ(instruction.pc(), 0x10u);
    EXPECT_TRUE(std::ranges::equal(instruction.encoding(), std::span(expected).first(count)));
  }
}

TEST(InstructionSnapshotTest, EqualityComparesPcAndEncoding) {
  const std::array<uint32_t, 2> words{0xbf800000, 0};
  const InstructionSnapshot one_word(0x10, std::span(words).first(1));
  EXPECT_EQ(one_word, InstructionSnapshot(0x10, std::span(words).first(1)));
  EXPECT_NE(one_word, InstructionSnapshot(0x14, std::span(words).first(1)));
  EXPECT_NE(one_word, InstructionSnapshot(0x10, std::array{0xbf800007u}));
  // A trailing zero word is part of the encoding, not unused storage.
  EXPECT_NE(one_word, InstructionSnapshot(0x10, words));
}

class RaceEventRegistrationTest : public ::testing::Test {
protected:
  RaceDetector detector{1, Dim3d(0), [this](RaceViolation v) { violations.push_back(v); },
                        counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4), 1024};
  std::vector<RaceViolation> violations;
};

TEST(EventRegistry, RetainsCapturedInstruction) {
  EventRegistry events;
  const auto id = events.add(WaveId{0}, kInstruction, LdsAccess::None, {}, kObligations,
                             MemoryOrderClass::UNORDERED);
  EXPECT_EQ(events.instruction(id), kInstruction);
}

TEST_F(RaceEventRegistrationTest, WaveApisRetainCapturedPcAndEncoding) {
  auto &wave = detector.getWaveRaceState(0);
  const std::array<uint32_t, 1> addresses{8};
  const std::array<uint32_t, 1> secondAddresses{16};
  const std::array ranges{LdsByteRange{24, 4}};
  auto at = [](int offset) { return InstructionSnapshot(kInstruction.pc() + offset, kWords); };
  wave.registerCounterEvent(at(0), kObligations);
  wave.registerLdsEvent(at(1), LdsAccess::Read, 1, 1, addresses, 4, kObligations,
                        MemoryOrderClass::UNORDERED);
  wave.registerLdsEvent(at(2), LdsAccess::Read, 1, 1, addresses, secondAddresses, 4, kObligations,
                        MemoryOrderClass::UNORDERED);
  wave.registerLdsRangeEvent(at(3), LdsAccess::Write, ranges, kObligations,
                             MemoryOrderClass::UNORDERED);
  ASSERT_EQ(detector.events().totalAllocated(), 4);
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(detector.events().instruction(EventId{i}), at(i));
  const auto &intervals = detector.events().ldsIntervals(EventId{2});
  EXPECT_TRUE(intervals.overlapsRange(8, 12));
  EXPECT_TRUE(intervals.overlapsRange(16, 20));
  EXPECT_FALSE(intervals.overlapsRange(12, 16));
  EXPECT_TRUE(detector.events().ldsIntervals(EventId{3}).overlapsRange(24, 28));
}
} // namespace

// ---- Retirement of completed LDS events ----
//
// A completed LDS event stays live until a barrier only so that other waves of
// the workgroup can still race with it. A single-wave workgroup has no other
// wave, so its waits retire LDS events immediately.

namespace {
constexpr uint32_t kTileBytes = 128;
constexpr std::array kAsyncStoreObligations{
    MemoryCounterObligation{WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_STORE}};
constexpr std::array kAsyncLoadObligations{
    MemoryCounterObligation{WaitCounterType::ASYNCCNT, MemoryOrderClass::ASYNC_LOAD}};

std::unique_ptr<RaceDetector> makeDetector(int nWaves, std::vector<RaceViolation> &violations) {
  return std::make_unique<RaceDetector>(
      nWaves, Dim3d(0), [&violations](RaceViolation v) { violations.push_back(v); },
      counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA5), /*ldsSizeBytes=*/64 * 1024);
}

/// An async store reading one tile: an LDS read on ASYNCcnt.
void asyncStoreTile(WaveRaceState &wave, uint64_t pc, uint32_t address) {
  const LdsByteRange range{.address = address, .bytes = kTileBytes};
  wave.registerLdsRangeEvent(testInstruction(pc), LdsAccess::Read, std::span(&range, 1),
                             kAsyncStoreObligations, MemoryOrderClass::ASYNC_STORE);
}

/// An async load writing one tile: an LDS write on ASYNCcnt.
void asyncLoadTile(WaveRaceState &wave, uint64_t pc, uint32_t address) {
  const LdsByteRange range{.address = address, .bytes = kTileBytes};
  wave.registerLdsRangeEvent(testInstruction(pc), LdsAccess::Write, std::span(&range, 1),
                             kAsyncLoadObligations, MemoryOrderClass::ASYNC_LOAD);
}

void waitAsynccnt(WaveRaceState &wave, int count) {
  PendingWaitCount wait;
  wait.add(WaitCounterType::ASYNCCNT, count);
  wave.dispatch(wait);
}

TEST(RaceDetector, SingleWave_CompletedLdsEventsAreRetiredWithoutABarrier) {
  std::vector<RaceViolation> violations;
  auto detector = makeDetector(/*nWaves=*/1, violations);
  auto &wave = detector->getWaveRaceState(0);
  asyncStoreTile(wave, /*pc=*/0, /*address=*/0);
  asyncLoadTile(wave, /*pc=*/1, /*address=*/kTileBytes);
  ASSERT_EQ(detector->getLdsReadEvents().size(), 1u);
  ASSERT_EQ(detector->getLdsWriteEvents().size(), 1u);

  waitAsynccnt(wave, 0);

  EXPECT_TRUE(detector->getLdsReadEvents().empty());
  EXPECT_TRUE(detector->getLdsWriteEvents().empty());
  EXPECT_TRUE(wave.getBarrierPendingEvents().empty());
  EXPECT_EQ(detector->events().status(EventId{0}), EventStatus::RETIRED);
  EXPECT_EQ(detector->events().status(EventId{1}), EventStatus::RETIRED);
  EXPECT_TRUE(violations.empty());
}

TEST(RaceDetector, MultiWave_CompletedLdsEventsStayLiveUntilABarrier) {
  std::vector<RaceViolation> violations;
  auto detector = makeDetector(/*nWaves=*/2, violations);
  auto &wave = detector->getWaveRaceState(0);
  asyncStoreTile(wave, /*pc=*/0, /*address=*/0);
  asyncLoadTile(wave, /*pc=*/1, /*address=*/kTileBytes);

  waitAsynccnt(wave, 0);

  // Another wave can still race with the completed events.
  EXPECT_EQ(detector->events().status(EventId{0}), EventStatus::WAVE_COMPLETE);
  EXPECT_EQ(detector->events().status(EventId{1}), EventStatus::WAVE_COMPLETE);
  EXPECT_EQ(detector->getLdsReadEvents().size(), 1u);
  EXPECT_EQ(detector->getLdsWriteEvents().size(), 1u);
  EXPECT_EQ(wave.getBarrierPendingEvents().size(), 2u);
  detector->validateWrite(/*addr=*/0, WaveId{1}, /*lane=*/0, /*nBytes=*/4, MemoryOrderClass::LDS);
  EXPECT_EQ(violations.size(), 1u);

  wave.flushBarrierPendingEvents();
  EXPECT_EQ(detector->events().status(EventId{0}), EventStatus::RETIRED);
  EXPECT_EQ(detector->events().status(EventId{1}), EventStatus::RETIRED);
  EXPECT_TRUE(detector->getLdsReadEvents().empty());
  EXPECT_TRUE(detector->getLdsWriteEvents().empty());
}

TEST(RaceDetector, SingleWave_DoubleBufferedStoreLoopKeepsOnlyPendingStores) {
  // Each iteration checks a refill of one of two tiles as a DS write would be
  // checked, stores the tile with an async store, and waits until at most one
  // store is outstanding. The wait completes the store of the tile the next
  // iteration refills, so only one store is ever pending, and the read list
  // must not grow with the loop.
  constexpr int kIterations = 8192;
  std::vector<RaceViolation> violations;
  auto detector = makeDetector(/*nWaves=*/1, violations);
  auto &wave = detector->getWaveRaceState(0);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const uint32_t tile = (iteration % 2) * kTileBytes;
    detector->validateWrite(static_cast<int>(tile), WaveId{0}, /*lane=*/0,
                            static_cast<int>(kTileBytes), MemoryOrderClass::LDS);
    asyncStoreTile(wave, static_cast<uint64_t>(iteration), tile);
    waitAsynccnt(wave, 1);
    ASSERT_EQ(detector->getLdsReadEvents().size(), 1u) << "iteration " << iteration;
  }
  EXPECT_TRUE(violations.empty());
  ASSERT_EQ(detector->events().totalAllocated(), kIterations);
  // Every completed store is retired and can be trimmed; the pending one cannot.
  // The registry has not trimmed the last completed store yet: a trim attempt
  // runs while a store is being allocated, when the previous store is still
  // pending, so trimming stops before it.
  EXPECT_TRUE(detector->events().isTrimmable(EventId{kIterations - 2}));
  EXPECT_FALSE(detector->events().isTrimmable(EventId{kIterations - 1}));
}

TEST(RaceDetector, SingleWave_RetiredLdsEventsAreTrimmed) {
  // Each store completes before the next is issued. The registry attempts a
  // trim when it allocates event kTrimAttemptInterval - 1, after adding it:
  // every earlier event is then retired and can be trimmed in a single-wave
  // workgroup, while in a two-wave workgroup each one waits for a barrier.
  for (const int nWaves : {1, 2}) {
    SCOPED_TRACE(nWaves);
    std::vector<RaceViolation> violations;
    auto detector = makeDetector(nWaves, violations);
    auto &wave = detector->getWaveRaceState(0);
    for (int index = 0; index < EventRegistry::kTrimAttemptInterval; ++index) {
      asyncStoreTile(wave, static_cast<uint64_t>(index), /*address=*/0);
      if (index + 1 < EventRegistry::kTrimAttemptInterval)
        waitAsynccnt(wave, 0);
    }
    EXPECT_EQ(detector->events().trimmedCount(),
              nWaves == 1 ? EventRegistry::kTrimAttemptInterval - 1 : 0);
    EXPECT_EQ(detector->getLdsReadEvents().size(),
              nWaves == 1 ? 1u : static_cast<size_t>(EventRegistry::kTrimAttemptInterval));
    EXPECT_TRUE(violations.empty());
  }
}
} // namespace
