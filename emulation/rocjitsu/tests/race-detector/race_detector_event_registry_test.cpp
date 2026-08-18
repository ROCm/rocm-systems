// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include <gtest/gtest.h>
#include <vector>

using namespace rocjitsu::plugins::race_detector;

TEST(RaceDetector, EventRegistry_TrimmedWaveLocalEventsAreNotBarrierQueued) {
  // This test protects the boundary between two event lifetimes:
  //
  //   * Wave-local events, such as GLOBAL_TO_VGPR, only matter to the wave
  //     that issued them. After s_waitcnt says the load is complete, the race
  //     detector no longer needs to keep the event for a later workgroup
  //     barrier. EventRegistry is allowed to trim these WAVE_COMPLETE events.
  //
  //   * LDS events can matter to other waves in the same workgroup. After
  //     s_waitcnt, they must remain live until s_barrier proves every wave has
  //     reached the same synchronization point.
  //
  // The bug this test catches was that all WAVE_COMPLETE events were queued
  // for later barrier retirement, including wave-local GLOBAL_TO_VGPR events.
  // Once EventRegistry trimmed those wave-local events, the barrier queue still
  // held stale EventIds. A later barrier flush could then try to retire an
  // already-trimmed id, causing heap corruption and potentially corrupting LDS
  // race bookkeeping.
  int N = EventRegistry::kTrimAttemptInterval;
  RaceDetector detector(/*nWaves=*/1, /*vgprCount=*/2, /*sgprCount=*/2, Dim3d(0),
                        [](RaceViolation) {});
  auto &wave = detector.getWaveRaceState(0);

  // Allocate enough wave-local loads to trigger EventRegistry trimming. Each
  // load is immediately completed with vmcnt(0), so none should be waiting for
  // a workgroup barrier.
  for (int i = 0; i < N; ++i) {
    wave.registerEvent(/*pc=*/static_cast<uint64_t>(i), MemoryEventType::GLOBAL_TO_VGPR,
                       std::vector<uint32_t>{0}, /*execMask=*/1);
    wave.dispatch(PendingWaitCount{/*vmcnt=*/0, /*lgkmcnt=*/-1});
  }

  EXPECT_GT(detector.events().trimmedCount(), 0);
  // If wave-local events were incorrectly queued for barrier retirement, this
  // would be non-empty. Flushing it after trimming would then touch stale ids.
  EXPECT_TRUE(wave.getBarrierPendingEvents().empty());
  wave.flushBarrierPendingEvents();
}
