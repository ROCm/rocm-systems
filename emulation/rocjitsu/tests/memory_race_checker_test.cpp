// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna5_sim_test_common.h"
#include "rocjitsu/vm/amdgpu/memory_diagnostic_report.h"
#include "rocjitsu/vm/amdgpu/memory_race_checker.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

using Access = MemoryRaceChecker::Access;
using AffineLaneRange = MemoryRaceChecker::AffineLaneRange;
using Finding = MemoryRaceChecker::Finding;
using LaneRange = MemoryRaceChecker::LaneRange;
using WaveKey = MemoryRaceChecker::WaveKey;
using test::cdna5::append_instruction;
using test::cdna5::Gfx1250Sim;
using test::cdna5::S_ENDPGM_GFX12;

Access access(WaveKey wave, uint64_t address, bool read, bool write, uint64_t pc = 0x100) {
  return {.wave = wave,
          .pc = pc,
          .address = address,
          .size = 4,
          .lane = 0,
          .read = read,
          .write = write};
}

Access byte_access(WaveKey wave, uint64_t address, bool read, bool write, uint64_t pc) {
  Access result = access(wave, address, read, write, pc);
  result.size = 1;
  return result;
}

std::vector<LaneRange> dword_lanes(uint64_t base, uint32_t stride = 4) {
  std::vector<LaneRange> lanes;
  for (int32_t lane = 0; lane < 64; ++lane)
    lanes.push_back({base + static_cast<uint64_t>(lane) * stride, 4, lane});
  return lanes;
}

TEST(MemoryRaceChecker, SameWaveRequiresCompletionForOverlappingLds) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey wave{1, 2, 3, 0, 1};
  checker.begin_wave(wave);

  auto producer = access(wave, 0x20, false, true);
  scoreboard.issue(WaitCounterKind::Ds, true);
  producer.completions[0] = scoreboard.latest_token(WaitCounterKind::Ds);
  producer.completion_count = 1;
  checker.lds_access(producer, scoreboard);
  checker.lds_access(access(wave, 0x22, true, false, 0x104), scoreboard);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::RAW);

  scoreboard.wait(WaitCounterKind::Ds, 0);
  checker.lds_access(access(wave, 0x20, true, false, 0x108), scoreboard);
  EXPECT_EQ(findings.size(), 1);
}

TEST(MemoryRaceChecker, OrderedSameWaveLdsIsNotReported) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey wave{1, 2, 3, 0, 1};
  checker.begin_wave(wave);
  scoreboard.issue(WaitCounterKind::Ds, false, 7);
  const auto token = scoreboard.latest_token(WaitCounterKind::Ds);

  auto first = access(wave, 0x20, false, true);
  first.completions[0] = token;
  first.completion_count = 1;
  first.ordered_class = 1 + (static_cast<uint32_t>(token.counter) << 16) + token.order;
  checker.lds_access(first, scoreboard);
  auto second = access(wave, 0x20, true, false, 0x104);
  second.ordered_class = first.ordered_class;
  checker.lds_access(second, scoreboard);
  EXPECT_TRUE(findings.empty());
}

TEST(MemoryRaceChecker, CrossWaveLdsReportsRawWarAndWawAtBytePrecision) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);

  auto a = access(first, 0x20, false, true);
  a.size = 2;
  checker.lds_access(a, scoreboard);
  auto b = access(second, 0x21, true, false, 0x104);
  b.size = 1;
  checker.lds_access(b, scoreboard);
  checker.lds_access(access(first, 0x30, true, false, 0x108), scoreboard);
  checker.lds_access(access(second, 0x30, false, true, 0x10c), scoreboard);
  checker.lds_access(access(first, 0x40, false, true, 0x110), scoreboard);
  checker.lds_access(access(second, 0x40, false, true, 0x114), scoreboard);

  ASSERT_EQ(findings.size(), 3);
  EXPECT_EQ(findings[0].address, 0x21);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::RAW);
  EXPECT_EQ(findings[1].kind, MemoryRaceChecker::Kind::WAR);
  EXPECT_EQ(findings[2].kind, MemoryRaceChecker::Kind::WAW);
}

TEST(MemoryRaceChecker, SameWaveModeSkipsCrossWaveHistory) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);
  constexpr MemoryRaceChecker::LdsCheckMode mode{true, false};

  scoreboard.issue(WaitCounterKind::Load, true);
  auto producer = access(first, 0x20, false, true);
  producer.completions[0] = scoreboard.latest_token(WaitCounterKind::Load);
  producer.completion_count = 1;
  checker.lds_access(producer, scoreboard, nullptr, mode);
  checker.lds_access(access(first, 0x20, true, false, 0x104), scoreboard, nullptr, mode);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_FALSE(findings[0].requires_barrier);

  findings.clear();
  checker.lds_access(access(second, 0x20, true, false, 0x108), scoreboard, nullptr, mode);
  EXPECT_TRUE(findings.empty());
}

TEST(MemoryRaceChecker, CrossWaveModeSkipsSameWavePendingChecks) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);
  constexpr MemoryRaceChecker::LdsCheckMode mode{false, true};

  scoreboard.issue(WaitCounterKind::Load, true);
  auto producer = access(first, 0x20, false, true);
  producer.completions[0] = scoreboard.latest_token(WaitCounterKind::Load);
  producer.completion_count = 1;
  checker.lds_access(producer, scoreboard, nullptr, mode);
  checker.lds_access(access(first, 0x20, true, false, 0x104), scoreboard, nullptr, mode);
  EXPECT_TRUE(findings.empty());

  checker.lds_access(access(second, 0x20, true, false, 0x108), scoreboard, nullptr, mode);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_TRUE(findings[0].requires_barrier);
}

TEST(MemoryRaceChecker, BarrierSynchronizesOnlyItsMembers) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  const WaveKey nonmember{1, 2, 3, 2, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);
  checker.begin_wave(nonmember);

  checker.lds_access(access(first, 0x20, false, true), scoreboard);
  const std::array members{first, second};
  checker.barrier(members);
  checker.lds_access(access(second, 0x20, true, false, 0x104), scoreboard);
  EXPECT_TRUE(findings.empty());
  checker.lds_access(access(nonmember, 0x20, true, false, 0x108), scoreboard);
  EXPECT_EQ(findings.size(), 1);
}

TEST(MemoryRaceChecker, BarrierDoesNotOrderInFlightLdsWrites) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);

  // A multi-buffered async copy stays in flight across the barrier that
  // guards another buffer; only touching its bytes early is a race.
  checker.lds_access(access(first, 0x40, false, true, 0x0f0), scoreboard);
  scoreboard.issue(WaitCounterKind::Load, true);
  auto copy = access(first, 0x20, false, true);
  copy.completions[0] = scoreboard.latest_token(WaitCounterKind::Load);
  copy.completion_count = 1;
  checker.lds_access(copy, scoreboard);
  const std::array members{first, second};
  checker.barrier(members);
  checker.lds_access(access(second, 0x40, true, false, 0x104), scoreboard);
  EXPECT_TRUE(findings.empty());

  checker.lds_access(access(second, 0x20, true, false, 0x108), scoreboard);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::RAW);
  EXPECT_EQ(findings[0].producer.pc, 0x100);
  EXPECT_FALSE(findings[0].requires_barrier);

  // Completion orders the copy for other waves only through a later barrier.
  findings.clear();
  scoreboard.wait(WaitCounterKind::Load, 0);
  checker.retire_completed(first, scoreboard);
  checker.lds_access(access(second, 0x20, true, false, 0x10c), scoreboard);
  EXPECT_EQ(findings.size(), 1);
  checker.barrier(members);
  checker.lds_access(access(second, 0x20, true, false, 0x110), scoreboard);
  EXPECT_EQ(findings.size(), 1);
}

TEST(MemoryRaceChecker, InFlightLdsWriteStaysUnorderedAcrossRepeatedBarriers) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);

  scoreboard.issue(WaitCounterKind::Load, true);
  auto copy = access(first, 0, false, true);
  copy.completions[0] = scoreboard.latest_token(WaitCounterKind::Load);
  copy.completion_count = 1;
  checker.lds_access(copy, AffineLaneRange{0x400, 4, 4, 0, 64}, scoreboard);
  const std::array members{first, second};
  checker.barrier(members);
  checker.barrier(members);
  checker.lds_access(access(second, 0x400 + 9 * 4, false, true, 0x104), scoreboard);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::WAW);
  EXPECT_EQ(findings[0].producer.lane, 9);
}

TEST(MemoryRaceChecker, GlobalConflictsOnlyAcrossWorkgroupsAndIgnoresAtomicPairs) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  const WaveKey wg0{1, 2, 3, 0, 1};
  const WaveKey wg1{1, 2, 4, 0, 1};
  checker.global_access(access(wg0, 0x1000, false, true));
  checker.global_access(access(wg0, 0x1000, true, false, 0x104));
  checker.global_access(access(wg1, 0x1000, true, false, 0x108));
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::RAW);

  auto atomic0 = access(wg0, 0x2000, true, true, 0x10c);
  auto atomic1 = access(wg1, 0x2000, true, true, 0x110);
  atomic0.atomic = atomic1.atomic = true;
  checker.global_access(atomic0);
  checker.global_access(atomic1);
  EXPECT_EQ(findings.size(), 1);
}

TEST(MemoryRaceChecker, DispatchCleanupDropsGlobalHistory) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  const WaveKey wg0{1, 2, 3, 0, 1};
  const WaveKey wg1{1, 2, 4, 0, 1};
  checker.global_access(access(wg0, 0x1000, false, true));
  EXPECT_GT(checker.global_shadow_bytes(), 0);
  checker.end_dispatch(2);
  EXPECT_EQ(checker.global_shadow_bytes(), 0);
  checker.global_access(access(wg1, 0x1000, true, false));
  EXPECT_TRUE(findings.empty());
}

TEST(MemoryRaceChecker, BatchedLdsInstructionChecksEveryLane) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);

  checker.lds_access(access(first, 0, false, true), dword_lanes(0x100), scoreboard);
  checker.lds_access(access(second, 0x100 + 17 * 4, true, false, 0x104), scoreboard);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::RAW);
  EXPECT_EQ(findings[0].address, 0x100 + 17 * 4);
  EXPECT_EQ(findings[0].size, 4);
  EXPECT_EQ(findings[0].producer.lane, 17);
  EXPECT_TRUE(findings[0].requires_barrier);
}

TEST(MemoryRaceChecker, BroadcastLanesStillRaceWithLaterWriter) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);

  checker.lds_access(access(second, 0, true, false, 0x104), dword_lanes(0x200, 0), scoreboard);
  checker.lds_access(access(first, 0x200, false, true, 0x108), scoreboard);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::WAR);
  EXPECT_EQ(findings[0].producer.pc, 0x104);
}

TEST(MemoryRaceChecker, SameWaveBatchedPendingReportsOnlyOverlappingLanes) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey wave{1, 2, 3, 0, 1};
  checker.begin_wave(wave);

  // An unordered async LDS write followed by an ordinary read of two lanes.
  scoreboard.issue(WaitCounterKind::Load, true);
  auto producer = access(wave, 0, false, true);
  producer.completions[0] = scoreboard.latest_token(WaitCounterKind::Load);
  producer.completion_count = 1;
  checker.lds_access(producer, dword_lanes(0x400), scoreboard);
  const std::array<LaneRange, 2> reads{{{0x400 + 3 * 4, 4, 3}, {0x800, 4, 4}}};
  checker.lds_access(access(wave, 0, true, false, 0x104), reads, scoreboard);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].address, 0x400 + 3 * 4);
  EXPECT_EQ(findings[0].producer.lane, 3);
  EXPECT_FALSE(findings[0].requires_barrier);

  scoreboard.wait(WaitCounterKind::Load, 0);
  checker.lds_access(access(wave, 0, true, false, 0x108), reads, scoreboard);
  EXPECT_EQ(findings.size(), 1);
}

TEST(MemoryRaceChecker, BarrierObservesCompletionWithoutRetirement) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);

  scoreboard.issue(WaitCounterKind::Ds, true);
  auto producer = access(first, 0x20, false, true);
  producer.completions[0] = scoreboard.latest_token(WaitCounterKind::Ds);
  producer.completion_count = 1;
  checker.lds_access(producer, scoreboard);
  scoreboard.wait(WaitCounterKind::Ds, 0);
  const std::array members{first, second};
  checker.barrier(members);
  checker.lds_access(access(second, 0x20, true, false, 0x104), scoreboard);
  EXPECT_TRUE(findings.empty());
}

TEST(MemoryRaceChecker, ClusterBarrierSynchronizesEachWorkgroup) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey a0{1, 2, 3, 0, 1};
  const WaveKey a1{1, 2, 3, 1, 1};
  const WaveKey b0{1, 2, 4, 0, 1};
  const WaveKey b1{1, 2, 4, 1, 1};
  for (const WaveKey &wave : {a0, a1, b0, b1})
    checker.begin_wave(wave);

  checker.lds_access(access(a0, 0x20, false, true), scoreboard);
  checker.lds_access(access(b0, 0x20, false, true), scoreboard);
  const std::array members{a0, b0, a1, b1};
  checker.barrier(members);
  checker.lds_access(access(a1, 0x20, true, false, 0x104), scoreboard);
  checker.lds_access(access(b1, 0x20, true, false, 0x104), scoreboard);
  EXPECT_TRUE(findings.empty());
}

TEST(MemoryRaceChecker, EndedWaveWritesSurviveBarrierOfRemainingWaves) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  const WaveKey ended{1, 2, 3, 2, 1};
  for (const WaveKey &wave : {first, second, ended})
    checker.begin_wave(wave);

  // Every active wave joins, but the ended wave never ordered its write.
  checker.lds_access(access(ended, 0x40, false, true, 0x10c), scoreboard);
  checker.end_wave(ended);
  checker.lds_access(access(first, 0x20, false, true), scoreboard);
  const std::array members{first, second};
  checker.barrier(members);
  checker.lds_access(access(second, 0x20, true, false, 0x104), scoreboard);
  EXPECT_TRUE(findings.empty());
  checker.lds_access(access(second, 0x40, true, false, 0x108), scoreboard);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::RAW);
  EXPECT_EQ(findings[0].producer.wave.wave, 2);
}

TEST(MemoryRaceChecker, WaveEndingAfterBarrierReleasesItsHistory) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  const WaveKey leaving{1, 2, 3, 2, 1};
  for (const WaveKey &wave : {first, second, leaving})
    checker.begin_wave(wave);

  checker.lds_access(access(leaving, 0x40, false, true, 0x10c), scoreboard);
  const std::array all{first, second, leaving};
  checker.barrier(all);
  checker.end_wave(leaving);
  checker.lds_access(access(first, 0x20, false, true), scoreboard);
  const std::array remaining{first, second};
  checker.barrier(remaining);
  checker.lds_access(access(second, 0x20, true, false, 0x104), scoreboard);
  checker.lds_access(access(second, 0x40, true, false, 0x108), scoreboard);
  EXPECT_TRUE(findings.empty());
}

TEST(MemoryRaceChecker, GlobalShadowKeepsTwoOwnersPerByte) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  for (uint32_t wg = 0; wg < 8; ++wg)
    checker.global_access(byte_access({1, 2, wg, 0, 1}, 0x1000, true, false, 0x100));
  checker.global_access(byte_access({1, 2, 8, 0, 1}, 0x1001, true, false, 0x104));

  // A retained reader writing byte 0 still races with the other retained one.
  checker.global_access(byte_access({1, 2, 0, 0, 1}, 0x1000, false, true, 0x108));
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].kind, MemoryRaceChecker::Kind::WAR);
  EXPECT_EQ(findings[0].producer.wave.workgroup, 1);

  // Flooding byte 0 does not evict byte 1's lone reader.
  checker.global_access(byte_access({1, 2, 9, 0, 1}, 0x1001, false, true, 0x10c));
  ASSERT_EQ(findings.size(), 2);
  EXPECT_EQ(findings[1].kind, MemoryRaceChecker::Kind::WAR);
  EXPECT_EQ(findings[1].address, 0x1001);
  EXPECT_EQ(findings[1].producer.wave.workgroup, 8);
  EXPECT_FALSE(checker.coverage_degraded());
}

TEST(MemoryRaceChecker, BatchedGlobalInstructionReportsEachConflictingLane) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  const WaveKey wg0{1, 2, 3, 0, 1};
  const WaveKey wg1{1, 2, 4, 0, 1};
  // Lanes span a shadow page boundary.
  checker.global_access(access(wg0, 0, false, true), dword_lanes(0x10000 - 128));
  checker.global_access(access(wg1, 0, true, false, 0x104), dword_lanes(0x10000 - 128));
  EXPECT_EQ(findings.size(), 64);
  for (const Finding &finding : findings) {
    EXPECT_EQ(finding.kind, MemoryRaceChecker::Kind::RAW);
    EXPECT_EQ(finding.producer.lane, finding.consumer.lane);
  }
}

TEST(MemoryRaceChecker, AffineGlobalDescriptorMatchesExpandedRangesAcrossPages) {
  const auto run = [](bool compact) {
    std::vector<Finding> findings;
    MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
    const WaveKey writer{1, 2, 3, 0, 1};
    const WaveKey reader{1, 2, 4, 0, 1};
    constexpr uint64_t base = 0x10000 - 128;
    if (compact) {
      const AffineLaneRange lanes{base, 4, 4, 0, 64};
      checker.global_access(access(writer, 0, false, true), lanes);
      checker.global_access(access(reader, 0, true, false, 0x104), lanes);
    } else {
      const auto lanes = dword_lanes(base);
      checker.global_access(access(writer, 0, false, true), lanes);
      checker.global_access(access(reader, 0, true, false, 0x104), lanes);
    }
    return findings;
  };

  const auto expanded = run(false);
  const auto compact = run(true);
  ASSERT_EQ(compact.size(), expanded.size());
  for (size_t i = 0; i < compact.size(); ++i) {
    EXPECT_EQ(compact[i].space, expanded[i].space);
    EXPECT_EQ(compact[i].kind, expanded[i].kind);
    EXPECT_EQ(compact[i].address, expanded[i].address);
    EXPECT_EQ(compact[i].size, expanded[i].size);
    EXPECT_EQ(compact[i].producer.lane, expanded[i].producer.lane);
    EXPECT_EQ(compact[i].consumer.lane, expanded[i].consumer.lane);
  }
}

TEST(MemoryRaceChecker, UnalignedAffineDescriptorMatchesGenericByteMasks) {
  const auto run = [](bool compact) {
    std::vector<Finding> findings;
    MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
    const WaveKey writer{1, 2, 3, 0, 1};
    const WaveKey reader{1, 2, 4, 0, 1};
    const AffineLaneRange descriptor{0x201, 5, 3, 7, 8};
    std::vector<LaneRange> expanded;
    for (uint32_t i = 0; i < descriptor.count; ++i)
      expanded.push_back({descriptor.address + uint64_t{i} * descriptor.stride, descriptor.size,
                          descriptor.first_lane + static_cast<int32_t>(i)});
    if (compact) {
      checker.global_access(access(writer, 0, false, true), descriptor);
      checker.global_access(access(reader, 0, true, false, 0x104), descriptor);
    } else {
      checker.global_access(access(writer, 0, false, true), expanded);
      checker.global_access(access(reader, 0, true, false, 0x104), expanded);
    }
    return findings;
  };

  const auto expanded = run(false);
  const auto compact = run(true);
  ASSERT_EQ(compact.size(), expanded.size());
  for (size_t i = 0; i < compact.size(); ++i) {
    EXPECT_EQ(compact[i].address, expanded[i].address);
    EXPECT_EQ(compact[i].size, expanded[i].size);
    EXPECT_EQ(compact[i].producer.lane, expanded[i].producer.lane);
    EXPECT_EQ(compact[i].consumer.lane, expanded[i].consumer.lane);
  }
}

TEST(MemoryRaceChecker, AffineOverlapAndBroadcastPreserveFirstLaneAttribution) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey writer{1, 2, 3, 0, 1};
  const WaveKey reader{1, 2, 3, 1, 1};
  checker.begin_wave(writer);
  checker.begin_wave(reader);

  checker.lds_access(access(writer, 0, false, true), AffineLaneRange{0x100, 4, 8, 10, 2},
                     scoreboard);
  checker.lds_access(access(reader, 0x104, true, false, 0x104), scoreboard);
  checker.lds_access(access(writer, 0, false, true, 0x108), AffineLaneRange{0x200, 0, 4, 20, 8},
                     scoreboard);
  checker.lds_access(access(reader, 0x200, true, false, 0x10c), scoreboard);

  ASSERT_EQ(findings.size(), 2);
  EXPECT_EQ(findings[0].producer.lane, 10);
  EXPECT_EQ(findings[1].producer.lane, 20);
}

TEST(MemoryRaceChecker, ConcurrentInstructionsAcrossShadowPagesDoNotDeadlock) {
  auto checker = std::make_shared<MemoryRaceChecker>([](const Finding &) {});
  auto finished = std::make_shared<std::atomic<int>>(0);
  // One lane per 1 KiB shadow page; half the workers visit the pages in reverse.
  const std::vector<LaneRange> forward = dword_lanes(0x100000, 1024);
  const std::vector<LaneRange> backward(forward.rbegin(), forward.rend());
  constexpr int kWorkers = 4;
  for (int worker = 0; worker < kWorkers; ++worker) {
    // Detached so that a deadlock fails the test instead of hanging it.
    std::thread([checker, finished, worker, lanes = worker % 2 ? backward : forward] {
      const WaveKey wave{1, 2, static_cast<uint32_t>(worker), 0, 1};
      for (int i = 0; i < 1000; ++i)
        checker->global_access(access(wave, 0, true, false), lanes);
      finished->fetch_add(1);
    }).detach();
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (finished->load() < kWorkers && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(finished->load(), kWorkers) << "global_access deadlocked across shadow shards";
}

TEST(MemoryRaceChecker, GlobalShadowBudgetDegradesCoverage) {
  MemoryRaceChecker checker([](const Finding &) {});
  EXPECT_EQ(checker.global_shadow_budget(), MemoryRaceChecker::kDefaultGlobalShadowBytes);
  checker.set_global_shadow_budget(0);
  checker.global_access(access({1, 2, 3, 0, 1}, 0x1000, false, true));
  EXPECT_TRUE(checker.coverage_degraded());
  checker.global_access(access({1, 2, 4, 0, 1}, 0x1000, true, false, 0x104));
  EXPECT_EQ(checker.finding_count(), 0);
}

TEST(MemoryRaceChecker, GlobalShadowBudgetIsSharedAcrossShards) {
  constexpr uint64_t budget = 64 << 10;
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_global_shadow_budget(budget);

  checker.global_access(access({1, 2, 3, 0, 1}, 0x1000, false, true));

  EXPECT_GT(checker.global_shadow_bytes(), budget / 256);
  EXPECT_LE(checker.global_shadow_bytes(), budget);
  EXPECT_LE(checker.global_shadow_peak_bytes(), budget);
  EXPECT_FALSE(checker.coverage_degraded());
}

TEST(MemoryRaceChecker, GlobalShadowAccountingHonorsExactFirstRetentionBoundary) {
  // One minimum page slab plus the initial directories, site slab, and record
  // slab is the smallest complete first-retention transaction.
  constexpr uint64_t minimum = 21888;
  MemoryRaceChecker exact([](const Finding &) {});
  exact.set_global_shadow_budget(minimum);
  exact.global_access(access({1, 2, 3, 0, 1}, 0x1000, false, true));
  EXPECT_EQ(exact.global_shadow_bytes(), minimum);
  EXPECT_EQ(exact.global_shadow_peak_bytes(), minimum);
  EXPECT_FALSE(exact.coverage_degraded());

  MemoryRaceChecker short_by_one([](const Finding &) {});
  short_by_one.set_global_shadow_budget(minimum - 1);
  short_by_one.global_access(access({1, 3, 3, 0, 1}, 0x1000, false, true));
  EXPECT_EQ(short_by_one.global_shadow_bytes(), 20864);
  EXPECT_LE(short_by_one.global_shadow_peak_bytes(), minimum - 1);
  EXPECT_TRUE(short_by_one.coverage_degraded());
  short_by_one.end_dispatch(3);
  EXPECT_EQ(short_by_one.global_shadow_bytes(), 0);
}

TEST(MemoryRaceChecker, RecordAdmissionFailureDoesNotChargeAnEmptyPage) {
  constexpr uint64_t site_storage_only = 4224;
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_global_shadow_budget(site_storage_only);
  checker.global_access(access({1, 4, 3, 0, 1}, 0x1000, false, true));

  EXPECT_EQ(checker.global_shadow_bytes(), site_storage_only);
  EXPECT_EQ(checker.global_shadow_peak_bytes(), site_storage_only);
  EXPECT_TRUE(checker.coverage_degraded());
  checker.end_dispatch(4);
  EXPECT_EQ(checker.global_shadow_bytes(), 0);
}

TEST(MemoryRaceChecker, ConcurrentGlobalShadowAdmissionNeverOvershootsBudget) {
  constexpr uint64_t budget = 5 * 21888;
  constexpr uint32_t workers = 16;
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_global_shadow_budget(budget);
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < workers; ++i) {
    threads.emplace_back([&, i] {
      ready.fetch_add(1);
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      checker.global_access(access({1, 100 + i, i, 0, 1}, uint64_t{i} << 20, false, true));
    });
  }
  while (ready.load() != workers)
    std::this_thread::yield();
  start.store(true, std::memory_order_release);
  for (auto &thread : threads)
    thread.join();

  EXPECT_LE(checker.global_shadow_bytes(), budget);
  EXPECT_LE(checker.global_shadow_peak_bytes(), budget);
}

TEST(MemoryRaceChecker, ConcurrentDispatchCleanupAndReportFlushPreserveStatistics) {
  constexpr uint32_t workers = 4;
  constexpr uint32_t dispatches_per_worker = 16;
  test::ScopedTempFile file("rocjitsu-concurrent-race-cleanup-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_report(report);
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (uint32_t worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      ready.fetch_add(1);
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      for (uint32_t iteration = 0; iteration < dispatches_per_worker; ++iteration) {
        const uint32_t dispatch = 1000 + worker * dispatches_per_worker + iteration;
        checker.global_access(
            access({1, dispatch, worker, 0, 1}, uint64_t{dispatch} << 12, false, true));
        checker.end_dispatch(dispatch);
      }
    });
  }
  std::thread writer([&] {
    while (ready.load(std::memory_order_acquire) != workers)
      std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (uint32_t write = 0; write < 32; ++write)
      report->flush();
  });
  for (auto &thread : threads)
    thread.join();
  writer.join();
  report->flush();

  EXPECT_EQ(checker.global_shadow_bytes(), 0);
  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find(std::format("\"completed_dispatches\": {}", workers * dispatches_per_worker)),
            std::string::npos);
  EXPECT_NE(json.find(std::format("\"global_accesses\":{}", workers * dispatches_per_worker)),
            std::string::npos);
}

TEST(MemoryRaceChecker, DispatchCleanupReleasesOnlyItsChargeForImmediateReuse) {
  constexpr uint64_t per_dispatch = 37248;
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_global_shadow_budget(2 * per_dispatch);
  checker.global_access(access({1, 20, 1, 0, 1}, 0x1000, false, true, 0x100));
  checker.global_access(access({1, 21, 1, 0, 1}, 0x2000, false, true, 0x200));
  ASSERT_EQ(checker.global_shadow_bytes(), 2 * per_dispatch);

  checker.end_dispatch(20);
  EXPECT_EQ(checker.global_shadow_bytes(), per_dispatch);
  checker.global_access(access({1, 22, 1, 0, 1}, 0x3000, false, true, 0x300));
  EXPECT_EQ(checker.global_shadow_bytes(), 2 * per_dispatch);

  checker.end_dispatch(22);
  checker.global_access(access({1, 21, 2, 0, 1}, 0x2000, true, false, 0x204));
  EXPECT_EQ(checker.finding_count(), 1);
}

TEST(MemoryRaceChecker, PackedGlobalSitePreservesAndReplacesExactProvenance) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  Access scalar = byte_access({1, 2, 3, 7, 1}, 0x1000, false, true, 0x100000001);
  scalar.lane = -1;
  checker.global_access(scalar);
  Access scalar_consumer = byte_access({1, 2, 4, 8, 1}, 0x1000, true, false, 0x100000002);
  scalar_consumer.lane = 0;
  checker.global_access(scalar_consumer);

  Access producer = byte_access({1, 2, 3, 0xfeedbeef, 1}, 0x1001, false, true, 0x123456789abcdef0);
  producer.lane = -1;
  checker.global_access(producer);

  producer.pc = 0xfedcba9876543210;
  producer.wave.wave = 63;
  producer.lane = 63;
  checker.global_access(producer);

  Access consumer = byte_access({1, 2, 4, 0, 1}, 0x1001, true, false, 0x200000001);
  consumer.lane = 0;
  checker.global_access(consumer);

  Access lane_zero = byte_access({1, 2, 3, 9, 1}, 0x1002, false, true, 0x200000002);
  lane_zero.lane = 0;
  checker.global_access(lane_zero);
  Access lane_63 = byte_access({1, 2, 4, 10, 1}, 0x1002, true, false, 0x200000003);
  lane_63.lane = 63;
  checker.global_access(lane_63);

  ASSERT_EQ(findings.size(), 3);
  EXPECT_EQ(findings[0].producer.lane, -1);
  EXPECT_EQ(findings[0].consumer.lane, 0);
  EXPECT_EQ(findings[1].producer.pc, producer.pc);
  EXPECT_EQ(findings[1].producer.wave.wave, 63u);
  EXPECT_EQ(findings[1].producer.lane, 63);
  EXPECT_EQ(findings[1].consumer.lane, 0);
  EXPECT_EQ(findings[1].address, 0x1001u);
  EXPECT_EQ(findings[1].size, 1u);
  EXPECT_EQ(findings[2].producer.lane, 0);
  EXPECT_EQ(findings[2].consumer.lane, 63);
}

TEST(MemoryRaceChecker, GlobalSiteAndDirectorySlabsGrowWithoutLosingNewestSite) {
  test::ScopedTempFile file("rocjitsu-global-site-growth-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  checker.set_report(report);
  Access producer = access({1, 30, 1, 0, 1}, 0x1000, false, true);
  for (uint64_t i = 0; i < 300; ++i) {
    producer.pc = 0x100000000ULL + i;
    checker.global_access(producer);
  }
  checker.global_access(access({1, 30, 2, 0, 1}, 0x1000, true, false, 0x400));
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].producer.pc, producer.pc);
  checker.end_dispatch(30);
  report->flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"site_slab_allocations\":2"), std::string::npos);
  EXPECT_NE(json.find("\"site_directory_growths\":6"), std::string::npos);
  EXPECT_NE(json.find("\"sites_reused\":"), std::string::npos);
}

TEST(MemoryRaceChecker, FailedSiteGrowthKeepsRetainedProvenanceExact) {
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  Access producer = access({1, 33, 1, 0, 1}, 0x1000, false, true);
  for (uint64_t i = 0; i < 256; ++i) {
    producer.pc = 0x100000000ULL + i;
    checker.global_access(producer);
  }
  const uint64_t retained_pc = producer.pc;
  checker.set_global_shadow_budget(checker.global_shadow_bytes());
  producer.pc = 0x200000000ULL;
  checker.global_access(producer);
  EXPECT_TRUE(checker.coverage_degraded());

  checker.global_access(access({1, 33, 2, 0, 1}, 0x1000, true, false, 0x300000000ULL));
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].producer.pc, retained_pc);
}

TEST(MemoryRaceChecker, SaturationCacheInvalidatesAfterGrowthAndCleanup) {
  constexpr uint64_t minimum = 21888;
  test::ScopedTempFile file("rocjitsu-global-saturation-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_report(report);
  checker.set_global_shadow_budget(minimum - 1);
  const Access producer = access({1, 31, 1, 0, 1}, 0x1000, false, true);
  checker.global_access(producer);
  checker.global_access(producer);
  EXPECT_TRUE(checker.coverage_degraded());

  checker.set_global_shadow_budget(2 * minimum);
  checker.global_access(access({1, 32, 1, 0, 1}, 0x2000, false, true, 0x200));
  checker.global_access(producer);
  checker.end_dispatch(32);
  checker.global_access(producer);
  checker.global_access(access({1, 31, 2, 0, 1}, 0x1000, true, false, 0x104));
  EXPECT_EQ(checker.finding_count(), 1);
  checker.end_dispatch(31);
  report->flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"saturation_cache_hits\":1"), std::string::npos);
  EXPECT_NE(json.find("\"saturation_cache_inserts\":2"), std::string::npos);
  EXPECT_NE(json.find("\"saturation_cache_invalidations\":2"), std::string::npos);
}

TEST(MemoryRaceChecker, SaturationStillChecksRetainedRecordsAndReportsFindings) {
  test::ScopedTempFile file("rocjitsu-global-saturation-findings-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_report(report);
  const WaveKey owner{1, 32, 1, 0, 1};
  for (uint32_t word = 0; word < 256; ++word) {
    Access prior = access(owner, 0x1000 + word * 4, true, false, 0x100);
    checker.global_access(prior);
    prior.read = false;
    prior.write = true;
    checker.global_access(prior);
    prior.read = true;
    checker.global_access(prior);
    prior.read = false;
    prior.atomic = true;
    checker.global_access(prior);
  }
  checker.set_global_shadow_budget(checker.global_shadow_bytes());

  checker.global_access(access({1, 32, 2, 0, 1}, 0x1000, true, false, 0x200));
  const uint64_t after_first = checker.finding_count();
  checker.global_access(access({1, 32, 3, 0, 1}, 0x1000, true, false, 0x204));
  EXPECT_GT(after_first, 0);
  EXPECT_GT(checker.finding_count(), after_first);
  EXPECT_TRUE(checker.coverage_degraded());
  checker.end_dispatch(32);
  report->flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"saturation_cache_hits\":1"), std::string::npos);
}

TEST(MemoryRaceChecker, DiagnosticsAreBoundedAndCountSuppression) {
  MemoryRaceChecker checker([](const Finding &) {});
  const WaveKey wg0{1, 2, 3, 0, 1};
  const WaveKey wg1{1, 2, 4, 0, 1};
  for (uint32_t i = 0; i < 65; ++i) {
    checker.global_access(access(wg0, 0x1000 + i * 4, false, true, 0x100 + i * 8));
    checker.global_access(access(wg1, 0x1000 + i * 4, true, false, 0x104 + i * 8));
  }
  EXPECT_EQ(checker.finding_count(), 65);
  EXPECT_EQ(checker.suppressed_count(), 1);
}

TEST(MemoryRaceChecker, ReportIncludesP0FixedWorkStatistics) {
  test::ScopedTempFile file("rocjitsu-race-statistics-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_report(report);
  const WaveKey writer{1, 2, 3, 0, 1};
  const WaveKey reader{1, 2, 4, 0, 1};
  checker.global_access(access(writer, 0x1000, false, true));
  checker.global_access(access(reader, 0x1000, true, false, 0x104));
  checker.end_dispatch(2);
  // A later dispatch with no checker work must not replace useful detail.
  checker.end_dispatch(99);
  report->flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"race_statistics\""), std::string::npos);
  EXPECT_NE(json.find("\"completed_dispatches\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"last_dispatch\": {\"dispatch\":2"), std::string::npos);
  EXPECT_NE(json.find("\"recent_dispatches\": [{\"dispatch\":2"), std::string::npos);
  EXPECT_NE(json.find("\"global_accesses\":2"), std::string::npos);
  EXPECT_NE(json.find("\"lane_ranges\":2"), std::string::npos);
  EXPECT_NE(json.find("\"dword_candidates\":2"), std::string::npos);
  EXPECT_NE(json.find("\"dword_visits\":2"), std::string::npos);
  EXPECT_NE(json.find("\"explicit_accesses\":2"), std::string::npos);
  EXPECT_NE(json.find("\"shadow_pages_created\":1"), std::string::npos);
  EXPECT_NE(json.find("\"global_records_visited\":1"), std::string::npos);
  EXPECT_NE(json.find("\"records_inserted\":2"), std::string::npos);
  EXPECT_NE(json.find("\"cleanup_pages\":1"), std::string::npos);
  EXPECT_NE(json.find("\"cleanup_records\":2"), std::string::npos);
  EXPECT_NE(json.find("\"global_shards\":[{\"shard\":"), std::string::npos);
}

TEST(MemoryRaceChecker, P0StatisticsCoverLdsBarriersAndLifecycle) {
  test::ScopedTempFile file("rocjitsu-lds-statistics-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_report(report);
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey first{1, 2, 3, 0, 1};
  const WaveKey second{1, 2, 3, 1, 1};
  checker.begin_wave(first);
  checker.begin_wave(second);
  checker.lds_access(access(first, 0x20, false, true), scoreboard);
  checker.lds_access(access(second, 0x20, true, false, 0x104), scoreboard);
  const std::array members{first, second};
  checker.barrier(members);
  checker.end_workgroup(2, 3);
  checker.end_dispatch(2);
  report->flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"lds_accesses\":2"), std::string::npos);
  EXPECT_NE(json.find("\"workgroup_creations\":1"), std::string::npos);
  EXPECT_NE(json.find("\"clock_matrix_growths\":2"), std::string::npos);
  EXPECT_NE(json.find("\"barrier_calls\":1"), std::string::npos);
  EXPECT_NE(json.find("\"barrier_members\":2"), std::string::npos);
  EXPECT_NE(json.find("\"cleanup_workgroups\":1"), std::string::npos);
}

TEST(MemoryRaceChecker, BatchedBindingsAvoidRepeatedLdsLookupAndClockGrowth) {
  test::ScopedTempFile file("rocjitsu-batched-lds-statistics-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  std::vector<Finding> findings;
  MemoryRaceChecker checker([&](const Finding &finding) { findings.push_back(finding); });
  checker.set_report(report);
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const std::array waves{WaveKey{1, 2, 3, 0, 1}, WaveKey{1, 2, 3, 1, 1}, WaveKey{1, 2, 3, 2, 1}};
  auto bindings = checker.begin_workgroup(waves);
  ASSERT_EQ(bindings.size(), waves.size());
  for (size_t i = 0; i < waves.size(); ++i)
    EXPECT_TRUE(bindings[i].valid_for(waves[i]));

  checker.lds_access(access(waves[0], 0x20, false, true), scoreboard, &bindings[0]);
  checker.lds_access(access(waves[1], 0x20, true, false, 0x104), scoreboard, &bindings[1]);
  checker.lds_access(access(waves[2], 0x40, false, true, 0x108), scoreboard, &bindings[2]);
  EXPECT_EQ(findings.size(), 1);

  checker.end_workgroup(2, 3);
  checker.end_dispatch(2);
  report->flush();
  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"workgroup_lookups\":1"), std::string::npos);
  EXPECT_NE(json.find("\"wave_search_steps\":0"), std::string::npos);
  EXPECT_NE(json.find("\"clock_matrix_growths\":1"), std::string::npos);
}

TEST(MemoryRaceChecker, AlignedOverlapIsSkippedBeforeDwordExpansion) {
  test::ScopedTempFile file("rocjitsu-aligned-range-statistics-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_report(report);
  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  const WaveKey wave{1, 2, 3, 0, 1};
  auto binding = checker.begin_wave(wave);
  const std::array<LaneRange, 2> ranges{{{0x100, 8, 0}, {0x104, 8, 1}}};
  checker.lds_access(access(wave, 0, false, true), ranges, scoreboard, &binding);
  checker.end_workgroup(2, 3);
  checker.end_dispatch(2);
  report->flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"dword_candidates\":3"), std::string::npos);
  EXPECT_NE(json.find("\"dword_visits\":3"), std::string::npos);
  EXPECT_NE(json.find("\"dword_duplicates_skipped\":1"), std::string::npos);

  std::vector<Finding> findings;
  MemoryRaceChecker attribution_checker(
      [&](const Finding &finding) { findings.push_back(finding); });
  const std::array waves{wave, WaveKey{1, 2, 3, 1, 1}};
  auto bindings = attribution_checker.begin_workgroup(waves);
  attribution_checker.lds_access(access(waves[0], 0, false, true), ranges, scoreboard,
                                 &bindings[0]);
  attribution_checker.lds_access(access(waves[1], 0x104, true, false, 0x104), scoreboard,
                                 &bindings[1]);
  ASSERT_EQ(findings.size(), 1);
  EXPECT_EQ(findings[0].producer.lane, 0);
}

TEST(MemoryRaceChecker, ReportIncludesAffineDescriptorAndPendingCounters) {
  test::ScopedTempFile file("rocjitsu-affine-race-statistics-");
  auto report = std::make_shared<MemoryDiagnosticReport>(file.path());
  MemoryRaceChecker checker([](const Finding &) {});
  checker.set_report(report);
  const WaveKey writer{1, 2, 3, 0, 1};
  const WaveKey reader{1, 2, 4, 0, 1};
  const AffineLaneRange global_lanes{0x1000, 4, 4, 0, 8};
  checker.global_access(access(writer, 0, false, true), global_lanes);
  checker.global_access(access(reader, 0, true, false, 0x104), global_lanes);

  MemoryWaitShadow shadow;
  MemoryWaitScoreboard scoreboard(shadow);
  checker.begin_wave(writer);
  scoreboard.issue(WaitCounterKind::Load, true);
  auto copy = access(writer, 0, false, true, 0x108);
  copy.completions[0] = scoreboard.latest_token(WaitCounterKind::Load);
  copy.completion_count = 1;
  checker.lds_access(copy, AffineLaneRange{0x200, 4, 4, 0, 4}, scoreboard);
  checker.end_workgroup(2, 3);
  checker.end_dispatch(2);
  report->flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"affine_accesses\":3"), std::string::npos);
  EXPECT_NE(json.find("\"affine_lane_ranges\":20"), std::string::npos);
  EXPECT_NE(json.find("\"affine_dword_runs\":3"), std::string::npos);
  EXPECT_NE(json.find("\"affine_dword_visits\":20"), std::string::npos);
  EXPECT_NE(json.find("\"affine_pending_ops\":1"), std::string::npos);
  EXPECT_NE(json.find("\"explicit_accesses\":0"), std::string::npos);
}

TEST(MemoryDiagnosticReport, CombinesFindingsAndDoesNotClobberWhenUnused) {
  test::ScopedTempFile file("rocjitsu-memory-diagnostics-");
  file.write("validation sentinel");
  { MemoryDiagnosticReport unused(file.path()); }
  {
    std::ifstream input(file.path());
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(input), {}), "validation sentinel");
  }

  MemoryDiagnosticReport report(file.path());
  const WaveKey wave{1, 2, 3, 0, 1};
  MemoryWaitScoreboard::Hazard hazard{
      .producer = {.sequence = 1,
                   .pc = 0x80,
                   .lanes = 1,
                   .reg = {RegClass::VGPR, 4, 1},
                   .counter = WaitCounterKind::Load,
                   .bytes = MemoryWaitScoreboard::kFullDwordByteMask},
      .consumer_pc = 0x84,
      .reg = {RegClass::VGPR, 4, 1},
      .write = false,
      .required_wait = 0};
  report.add_wait(wave, hazard, "loadcnt", 0);
  report.add_race({.space = MemoryRaceChecker::Space::LDS,
                   .kind = MemoryRaceChecker::Kind::RAW,
                   .producer = access(wave, 0x20, false, true, 0x100),
                   .consumer = access({1, 2, 3, 1, 1}, 0x20, true, false, 0x104),
                   .address = 0x20,
                   .size = 4,
                   .requires_barrier = true});
  report.add_suppressed_race();
  report.set_coverage_degraded();
  report.flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"checker\":\"memory-wait\""), std::string::npos);
  EXPECT_NE(json.find("\"checker\":\"memory-race\""), std::string::npos);
  EXPECT_NE(json.find("\"observed\": 3"), std::string::npos);
  EXPECT_NE(json.find("\"suppressed\": 1"), std::string::npos);
  EXPECT_NE(json.find("\"coverage_degraded\": true"), std::string::npos);
}

TEST(MemoryDiagnosticReport, BoundsRecentRaceDispatchStatistics) {
  test::ScopedTempFile file("rocjitsu-bounded-race-statistics-");
  MemoryDiagnosticReport report(file.path());
  for (uint32_t dispatch = 1; dispatch <= 9; ++dispatch) {
    MemoryRaceChecker::DispatchStatistics statistics;
    statistics.dispatch = dispatch;
    statistics.counters.global_accesses = 1;
    report.add_race_statistics(std::move(statistics));
  }
  report.flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"completed_dispatches\": 9"), std::string::npos);
  EXPECT_NE(json.find("\"last_dispatch\": {\"dispatch\":9"), std::string::npos);
  EXPECT_EQ(json.find("\"recent_dispatches\": [{\"dispatch\":1,"), std::string::npos);
  EXPECT_NE(json.find("\"recent_dispatches\": [{\"dispatch\":2,"), std::string::npos);
}

TEST(MemoryDiagnosticReport, ConcurrentFlushPreservesNewerRevisions) {
  test::ScopedTempFile file("rocjitsu-concurrent-memory-diagnostics-");
  MemoryDiagnosticReport report(file.path());
  constexpr uint32_t kProducers = 4;
  constexpr uint32_t kFindingsPerProducer = 1000;
  std::atomic<bool> start{false};
  std::vector<std::thread> producers;
  for (uint32_t producer = 0; producer < kProducers; ++producer) {
    producers.emplace_back([&] {
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      for (uint32_t finding = 0; finding < kFindingsPerProducer; ++finding)
        report.add_suppressed_race();
    });
  }
  std::thread first_writer([&] {
    while (!start.load(std::memory_order_acquire))
      std::this_thread::yield();
    for (uint32_t write = 0; write < 32; ++write)
      report.flush();
  });
  std::thread second_writer([&] {
    while (!start.load(std::memory_order_acquire))
      std::this_thread::yield();
    for (uint32_t write = 0; write < 32; ++write)
      report.flush();
  });

  start.store(true, std::memory_order_release);
  for (auto &producer : producers)
    producer.join();
  first_writer.join();
  second_writer.join();
  report.flush();

  std::ifstream input(file.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  const std::string expected = std::format("\"observed\": {}", kProducers * kFindingsPerProducer);
  EXPECT_NE(json.find(expected), std::string::npos);
  EXPECT_NE(json.find(std::format("\"suppressed\": {}", kProducers * kFindingsPerProducer)),
            std::string::npos);
  EXPECT_EQ(json.find("}{\n  \"version\""), std::string::npos);
}

std::string race_execution_config(std::string_view report_path = {},
                                  std::string_view checks = "all") {
  std::ifstream file(test::cdna5::kGfx1250ConfigPath);
  std::string config(std::istreambuf_iterator<char>(file), {});
  const size_t root = config.find('{');
  EXPECT_NE(root, std::string::npos);
  config.insert(root + 1,
                std::format("\"race_checking\":\"on\",\"race_checking_checks\":\"{}\",", checks));
  if (!report_path.empty())
    config.insert(root + 1, std::format("\"memory_diagnostic_report\":\"{}\",", report_path));
  return config;
}

std::vector<Finding> run_cross_wave_lds(unsigned synchronization, std::string_view report_path = {},
                                        std::string_view checks = "all") {
  std::vector<Finding> findings;
  std::mutex findings_mutex;
  Gfx1250Sim sim(race_execution_config(report_path, checks));
  const auto checker = sim.soc->memory_race_checker();
  checker->set_reporter([&](const Finding &finding) {
    std::lock_guard lock(findings_mutex);
    findings.push_back(finding);
  });

  std::vector<uint32_t> code;
  // v1 = tid * 4; v2 = (tid ^ 32) * 4. Each wave reads the slots written
  // by the other wave in the same workgroup.
  append_instruction(
      code, cdna5::build_vop2(cdna5::kVLshlrevB32Vop2, {.src0 = 132, .vsrc1 = 0, .vdst = 1}));
  append_instruction(code,
                     cdna5::build_vop2(cdna5::kVXorB32Vop2, {.src0 = 160, .vsrc1 = 0, .vdst = 2}));
  append_instruction(
      code, cdna5::build_vop2(cdna5::kVLshlrevB32Vop2, {.src0 = 132, .vsrc1 = 2, .vdst = 2}));
  append_instruction(code, cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 1, .data0 = 0}));
  if (synchronization == 2)
    append_instruction(code, cdna5::build_sopp(cdna5::kSWaitDscntSopp, {.simm16 = 0}));
  if (synchronization != 0) {
    append_instruction(code, cdna5::build_sop1(cdna5::kSBarrierSignalSop1, {.ssrc0 = 193}));
    append_instruction(code, cdna5::build_sopp(cdna5::kSBarrierWaitSopp, {.simm16 = 0xffff}));
  }
  append_instruction(code, cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 3}));
  append_instruction(code, cdna5::build_sopp(cdna5::kSWaitDscntSopp, {.simm16 = 0}));
  append_instruction(code, S_ENDPGM_GFX12);

  const uint64_t kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false,
                                           false, false, 0, 0, 0, 0, /*enable_vgpr_workitem_id=*/1);
  test::AqlQueue queue(sim.memory, sim.cp());
  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  packet.setup = 1;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 64;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.group_segment_size = 256;
  packet.kernel_object = kernel;
  queue.submit(packet);
  sim.engine->run();

  EXPECT_EQ(sim.memory->read64(test::AqlQueue::DEFAULT_READ_PTR_ADDR), 1u);
  EXPECT_EQ(checker->tracked_workgroups(), 0u);
  EXPECT_EQ(checker->global_shadow_bytes(), 0u);
  return findings;
}

TEST(CoreRaceCheckingExecutionTest, CrossWaveLdsRequiresWaitAndBarrier) {
  const auto unordered = run_cross_wave_lds(0);
  ASSERT_FALSE(unordered.empty());
  EXPECT_TRUE(std::ranges::any_of(unordered,
                                  [](const Finding &finding) { return finding.requires_barrier; }));

  const auto in_flight = run_cross_wave_lds(1);
  ASSERT_FALSE(in_flight.empty());
  EXPECT_TRUE(std::ranges::any_of(
      in_flight, [](const Finding &finding) { return !finding.requires_barrier; }));

  EXPECT_TRUE(run_cross_wave_lds(2).empty());
}

TEST(CoreRaceCheckingExecutionTest, CrossWaveLdsRequiresItsSelectedFamily) {
  EXPECT_FALSE(run_cross_wave_lds(0, {}, "lds-cross-wave").empty());
  EXPECT_TRUE(run_cross_wave_lds(0, {}, "lds-same-wave").empty());
  EXPECT_TRUE(run_cross_wave_lds(0, {}, "global").empty());
}

TEST(CoreRaceCheckingExecutionTest, LdsWorkgroupUsesBatchedCachedWaveBindings) {
  test::ScopedTempFile report("rocjitsu-core-race-lds-fast-path-");
  EXPECT_FALSE(run_cross_wave_lds(0, report.path()).empty());
  std::ifstream input(report.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"workgroup_lookups\":1"), std::string::npos);
  EXPECT_NE(json.find("\"wave_search_steps\":0"), std::string::npos);
  EXPECT_NE(json.find("\"clock_matrix_growths\":1"), std::string::npos);
}

TEST(CoreRaceCheckingExecutionTest, LdsOnlyReportFinalizesAtGridRetirement) {
  test::ScopedTempFile report("rocjitsu-core-race-lds-only-report-");
  EXPECT_FALSE(run_cross_wave_lds(0, report.path(), "lds-cross-wave").empty());
  std::ifstream input(report.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"checker\":\"memory-race\""), std::string::npos);
  EXPECT_NE(json.find("\"lds_accesses\":2"), std::string::npos);
}

std::vector<Finding> run_cross_workgroup_global(bool atomic, std::string_view report_path = {},
                                                uint32_t grid_size = 64,
                                                std::string_view checks = "all") {
  using namespace rocr::llvm::amdhsa;
  std::vector<Finding> findings;
  std::mutex findings_mutex;
  Gfx1250Sim sim(race_execution_config(report_path, checks));
  const auto checker = sim.soc->memory_race_checker();
  checker->set_reporter([&](const Finding &finding) {
    std::lock_guard lock(findings_mutex);
    findings.push_back(finding);
  });

  std::vector<uint32_t> code;
  append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 0}));
  append_instruction(code, cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 129, .vdst = 1}));
  append_instruction(code, cdna5::build_vglobal(atomic ? cdna5::kGlobalAtomicAddU32Vglobal
                                                       : cdna5::kGlobalStoreB32Vglobal,
                                                {.saddr = 0, .vdst = 2, .vsrc = 1, .vaddr = 0}));
  append_instruction(code, cdna5::build_sopp(cdna5::kSWaitLoadcntSopp, {.simm16 = 0}));
  append_instruction(code, cdna5::build_sopp(cdna5::kSWaitStorecntSopp, {.simm16 = 0}));
  append_instruction(code, S_ENDPGM_GFX12);

  uint32_t properties = 0;
  AMDHSA_BITS_SET(properties, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  constexpr uint64_t target = 0x400000;
  test::cdna5::write_global_u32(*sim.memory, target, 0);
  const uint64_t kernel = sim.write_kernel(0x10000, code.data(), code.size(), 104, 32, 2, false,
                                           false, false, properties, 8);
  test::AqlQueue queue(sim.memory, sim.cp());
  queue.dispatch(kernel, grid_size, 32, target);
  sim.engine->run();

  EXPECT_EQ(sim.memory->read64(test::AqlQueue::DEFAULT_READ_PTR_ADDR), 1u);
  EXPECT_EQ(checker->tracked_workgroups(), 0u);
  EXPECT_EQ(checker->global_shadow_bytes(), 0u);
  return findings;
}

TEST(CoreRaceCheckingExecutionTest, GlobalStoresRaceAcrossWorkgroupsButAtomicsDoNot) {
  const auto stores = run_cross_workgroup_global(false);
  ASSERT_FALSE(stores.empty());
  EXPECT_TRUE(std::ranges::all_of(stores, [](const Finding &finding) {
    return finding.space == MemoryRaceChecker::Space::Global;
  }));
  EXPECT_TRUE(run_cross_workgroup_global(true).empty());
}

TEST(CoreRaceCheckingExecutionTest, GlobalRaceRequiresItsSelectedFamily) {
  EXPECT_FALSE(run_cross_workgroup_global(false, {}, 64, "global").empty());
  EXPECT_TRUE(run_cross_workgroup_global(false, {}, 64, "lds-cross-wave").empty());
}

TEST(CoreRaceCheckingExecutionTest, ConfiguredReportIsFlushedAtGridRetirement) {
  test::ScopedTempFile report("rocjitsu-core-race-report-");
  EXPECT_FALSE(run_cross_workgroup_global(false, report.path()).empty());
  std::ifstream input(report.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"checker\":\"memory-race\""), std::string::npos);
  EXPECT_NE(json.find("\"global_accesses\":2"), std::string::npos);
  EXPECT_NE(json.find("\"affine_accesses\":2"), std::string::npos);
  EXPECT_NE(json.find("\"affine_lane_ranges\":64"), std::string::npos);
  EXPECT_NE(json.find("\"affine_dword_runs\":2"), std::string::npos);
  EXPECT_NE(json.find("\"workgroup_creations\":0"), std::string::npos);
  EXPECT_NE(json.find("\"coverage_degraded\": false"), std::string::npos);
}

TEST(CoreRaceCheckingExecutionTest, SingleWorkgroupSkipsGlobalRaceState) {
  test::ScopedTempFile report("rocjitsu-core-race-single-workgroup-");
  EXPECT_TRUE(run_cross_workgroup_global(false, report.path(), 32).empty());
  std::ifstream input(report.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"global_accesses\":0"), std::string::npos);
  EXPECT_NE(json.find("\"workgroup_creations\":0"), std::string::npos);
}

} // namespace
} // namespace rocjitsu::amdgpu
