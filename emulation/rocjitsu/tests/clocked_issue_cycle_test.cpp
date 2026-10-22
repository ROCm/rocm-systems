// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file clocked_issue_cycle_test.cpp
/// @brief ComputeUnitCore::run_issue_cycle(): the per-cycle entry point for a
/// clocked driver that chooses which waves issue.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

constexpr uint64_t kProgram = 0x1000;
constexpr uint32_t kMovS5 = 0xBE850087u;  // s_mov_b32 s5, 7
constexpr uint32_t kNop = 0xBF800000u;    // s_nop 0
constexpr uint32_t kEndpgm = 0xBFB00000u; // s_endpgm

/// A two-slot RDNA4 CU over @p memory.
std::unique_ptr<ComputeUnitCore> make_cu(GpuMemory &memory, L2Cache &l2) {
  l2.set_backing_memory(&memory);
  ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA4;
  config.num_wf_slots = 2;
  config.sgprs_per_wf = 104;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  return ComputeUnitCore::create("clocked_cu", config, &memory, &l2);
}

class ClockedIssueCycleTest : public ::testing::Test {
protected:
  void SetUp() override {
    cu = make_cu(memory, l2);
    ASSERT_NE(cu, nullptr);
    const std::array words{kMovS5, kNop, kEndpgm};
    for (size_t i = 0; i < words.size(); ++i)
      memory.write32(kProgram + i * 4, words[i]);
    for (size_t i = 0; i < 4; ++i)
      memory.write32(kProgram + (words.size() + i) * 4, kEndpgm);
    for (uint32_t slot = 0; slot < 2; ++slot) {
      waves[slot] = cu->dispatch_wf(slot, kProgram, 104, 256);
      ASSERT_NE(waves[slot], nullptr);
      ASSERT_EQ(waves[slot]->wf_id(), slot);
    }
  }

  void TearDown() override {
    for (auto *wave : waves)
      if (wave && !wave->is_halted())
        wave->halt();
  }

  GpuMemory memory{"clocked_memory"};
  L2Cache l2{"clocked_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::array<Wavefront *, 2> waves{};
};

TEST_F(ClockedIssueCycleTest, IssuesOnlySelectedSlots) {
  EXPECT_EQ(cu->run_issue_cycle(0b10), 0b10u);
  EXPECT_EQ(waves[0]->pc, kProgram);
  EXPECT_EQ(waves[1]->pc, kProgram + 4);
  EXPECT_EQ(cu->read_sgpr(waves[1]->sgpr_alloc().base + 5), 7u);

  EXPECT_EQ(cu->run_issue_cycle(0), 0u);
  EXPECT_EQ(waves[0]->pc, kProgram);
  EXPECT_EQ(waves[1]->pc, kProgram + 4);

  EXPECT_EQ(cu->run_issue_cycle(0b11), 0b11u);
  EXPECT_EQ(waves[0]->pc, kProgram + 4);
  EXPECT_EQ(waves[1]->pc, kProgram + 8);
}

TEST_F(ClockedIssueCycleTest, HaltedWaveDoesNotIssueAndLeavesItsSlot) {
  for (int cycle = 0; cycle < 3; ++cycle)
    (void)cu->run_issue_cycle(0b01);
  EXPECT_TRUE(waves[0]->is_halted());
  EXPECT_EQ(cu->run_issue_cycle(0b01), 0u);
  EXPECT_FALSE(waves[1]->is_halted());
  EXPECT_EQ(waves[1]->pc, kProgram);
}

TEST_F(ClockedIssueCycleTest, SleepCountsDownWhetherOrNotSelected) {
  waves[0]->set_sleep_cycles(2);
  // Unselected: the sleep still burns one count per cycle.
  EXPECT_EQ(cu->run_issue_cycle(0), 0u);
  EXPECT_EQ(waves[0]->sleep_cycles(), 1u);
  // Selected while sleeping: the cycle is spent sleeping, not issuing.
  EXPECT_EQ(cu->run_issue_cycle(0b01), 0u);
  EXPECT_EQ(waves[0]->sleep_cycles(), 0u);
  EXPECT_EQ(waves[0]->pc, kProgram);
  EXPECT_EQ(cu->run_issue_cycle(0b01), 0b01u);
  EXPECT_EQ(waves[0]->pc, kProgram + 4);
}

TEST(ClockedIssueCycleLaunchTest, LaunchHandlerSeesEachDispatchedWave) {
  GpuMemory memory{"launch_memory"};
  L2Cache l2{"launch_l2"};
  auto cu = make_cu(memory, l2);
  std::vector<std::pair<uint32_t, uint64_t>> launched;
  cu->set_wave_launch_handler([&](Wavefront &wf) {
    launched.emplace_back(wf.wf_id(), wf.dispatch_generation());
    EXPECT_EQ(wf.state(), WfState::RUNNING);
    EXPECT_EQ(wf.pc, kProgram);
  });
  Wavefront *first = cu->dispatch_wf(0, kProgram, 104, 256);
  Wavefront *second = cu->dispatch_wf(1, kProgram, 104, 256);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(launched.size(), 2u);
  EXPECT_EQ(launched[0].first, 0u);
  EXPECT_EQ(launched[1].first, 1u);
  // A slot reused after its wave halts launches again, with a new generation.
  const uint64_t generation = first->dispatch_generation();
  first->halt();
  ASSERT_EQ(cu->dispatch_wf(2, kProgram, 104, 256), first);
  ASSERT_EQ(launched.size(), 3u);
  EXPECT_EQ(launched[2].first, 0u);
  EXPECT_NE(launched[2].second, generation);
  first->halt();
  second->halt();
}

} // namespace
