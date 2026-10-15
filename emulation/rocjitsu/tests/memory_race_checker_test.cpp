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
  checker.lds_access(copy, dword_lanes(0x400), scoreboard);
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

std::string race_execution_config(std::string_view report_path = {}) {
  std::ifstream file(test::cdna5::kGfx1250ConfigPath);
  std::string config(std::istreambuf_iterator<char>(file), {});
  const size_t root = config.find('{');
  EXPECT_NE(root, std::string::npos);
  config.insert(root + 1, "\"race_checking\":\"on\",");
  if (!report_path.empty())
    config.insert(root + 1, std::format("\"memory_diagnostic_report\":\"{}\",", report_path));
  return config;
}

std::vector<Finding> run_cross_wave_lds(unsigned synchronization) {
  std::vector<Finding> findings;
  std::mutex findings_mutex;
  Gfx1250Sim sim(race_execution_config());
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

std::vector<Finding> run_cross_workgroup_global(bool atomic, std::string_view report_path = {}) {
  using namespace rocr::llvm::amdhsa;
  std::vector<Finding> findings;
  std::mutex findings_mutex;
  Gfx1250Sim sim(race_execution_config(report_path));
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
  queue.dispatch(kernel, 64, 32, target);
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

TEST(CoreRaceCheckingExecutionTest, ConfiguredReportIsFlushedAtGridRetirement) {
  test::ScopedTempFile report("rocjitsu-core-race-report-");
  EXPECT_FALSE(run_cross_workgroup_global(false, report.path()).empty());
  std::ifstream input(report.path());
  const std::string json(std::istreambuf_iterator<char>(input), {});
  EXPECT_NE(json.find("\"checker\":\"memory-race\""), std::string::npos);
  EXPECT_NE(json.find("\"coverage_degraded\": false"), std::string::npos);
}

} // namespace
} // namespace rocjitsu::amdgpu
