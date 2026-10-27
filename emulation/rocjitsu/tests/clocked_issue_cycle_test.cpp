// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file clocked_issue_cycle_test.cpp
/// @brief ComputeUnitCore::run_issue_cycle(): the per-cycle entry point for a
/// clocked driver that chooses which waves issue.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

constexpr uint64_t kProgram = 0x1000;
constexpr uint64_t kVectorProgram = 0x2000;
constexpr uint32_t kMovS5 = 0xBE850087u;      // s_mov_b32 s5, 7
constexpr uint32_t kNop = 0xBF800000u;        // s_nop 0
constexpr uint32_t kEndpgm = 0xBFB00000u;     // s_endpgm
constexpr uint32_t kMoveVector = 0x7E000281u; // v_mov_b32 v0, 1

constexpr uint32_t kSlots = 2;
constexpr uint32_t kSgprs = 104;
constexpr uint32_t kVgprs = 256;

/// A two-slot CU over @p memory.
std::unique_ptr<ComputeUnitCore> make_cu(GpuMemory &memory, L2Cache &l2,
                                         rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  l2.set_backing_memory(&memory);
  ComputeUnitCore::Config config{};
  config.arch = arch;
  config.num_wf_slots = kSlots;
  config.sgprs_per_wf = kSgprs;
  config.vgprs_per_wf = kVgprs;
  config.lds_size_kb = 64;
  return ComputeUnitCore::create("clocked_cu", config, &memory, &l2);
}

/// Identity translation whose instruction fetches can be made to retry.
class RetryingFetchTranslator final : public AddressSpaceTranslator {
public:
  VmTranslationResult translate(uint64_t address, std::size_t size,
                                VmAccessKind access) const override {
    if (access == VmAccessKind::Execute && fetch_unavailable)
      return {.outcome = VmAccessOutcome::Unavailable, .translation = {}};
    return identity_.translate(address, size, access);
  }

  bool fetch_unavailable = true;

private:
  IdentityAddressSpaceTranslator identity_;
};

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
    // Each wave is its own workgroup; slots fill in order.
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
      waves[slot] = cu->dispatch_wf(/*wg_id=*/slot, kProgram, kSgprs, kVgprs);
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
  std::array<Wavefront *, kSlots> waves{};
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

TEST_F(ClockedIssueCycleTest, WaitingWavesProgressWithNothingSelected) {
  // No counters are outstanding, so the wait is satisfied and the drain is done.
  waves[0]->set_state(WfState::WAITCNT);
  waves[1]->set_state(WfState::ENDING);
  EXPECT_EQ(cu->run_issue_cycle(0), 0u);
  EXPECT_EQ(waves[0]->state(), WfState::RUNNING);
  EXPECT_EQ(waves[0]->pc, kProgram);
  EXPECT_TRUE(waves[1]->is_halted());
}

TEST_F(ClockedIssueCycleTest, RetriedFetchDoesNotIssue) {
  GpuVm vm;
  auto translator = std::make_shared<RetryingFetchTranslator>();
  const AddressSpaceHandle address_space =
      vm.register_address_space(0, translator, std::make_shared<GpuMemoryPhysicalAccess>(memory));
  ASSERT_TRUE(address_space);
  cu->set_gpu_vm(&vm);
  waves[0]->set_address_space(address_space);

  EXPECT_EQ(cu->run_issue_cycle(0b01), 0u);
  EXPECT_EQ(waves[0]->state(), WfState::RUNNING);
  EXPECT_EQ(waves[0]->pc, kProgram);

  translator->fetch_unavailable = false;
  EXPECT_EQ(cu->run_issue_cycle(0b01), 0b01u);
  EXPECT_EQ(waves[0]->pc, kProgram + 4);
  EXPECT_EQ(cu->read_sgpr(waves[0]->sgpr_alloc().base + 5), 7u);
}

TEST(ClockedIssueCycleVskipTest, SkippedInstructionDoesNotIssue) {
  GpuMemory memory{"vskip_memory"};
  L2Cache l2{"vskip_l2"};
  // VSKIP is a CDNA mode bit.
  std::unique_ptr<ComputeUnitCore> cu = make_cu(memory, l2, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(cu, nullptr);
  for (size_t i = 0; i < 4; ++i)
    memory.write32(kVectorProgram + i * 4, i == 0 ? kMoveVector : kNop);
  Wavefront *wave = cu->dispatch_wf(/*wg_id=*/0, kVectorProgram, kSgprs, kVgprs);
  ASSERT_NE(wave, nullptr);
  wave->set_mode_raw(wave->mode_raw() | Wavefront::VSKIP_BIT);
  EXPECT_EQ(cu->run_issue_cycle(0b01), 0u);
  EXPECT_EQ(wave->pc, kVectorProgram + 4);
  wave->set_mode_raw(wave->mode_raw() & ~Wavefront::VSKIP_BIT);
  EXPECT_EQ(cu->run_issue_cycle(0b01), 0b01u);
  EXPECT_EQ(wave->pc, kVectorProgram + 8);
  wave->halt();
}

TEST(ClockedIssueCycleLaunchTest, LaunchHandlerSeesEachDispatchedWave) {
  GpuMemory memory{"launch_memory"};
  L2Cache l2{"launch_l2"};
  std::unique_ptr<ComputeUnitCore> cu = make_cu(memory, l2);
  std::vector<std::pair<uint32_t, uint64_t>> launched;
  cu->set_wave_launch_handler([&](Wavefront &wf) {
    launched.emplace_back(wf.wf_id(), wf.dispatch_generation());
    EXPECT_EQ(wf.state(), WfState::RUNNING);
    EXPECT_EQ(wf.pc, kProgram);
  });
  Wavefront *first = cu->dispatch_wf(/*wg_id=*/0, kProgram, kSgprs, kVgprs);
  Wavefront *second = cu->dispatch_wf(/*wg_id=*/1, kProgram, kSgprs, kVgprs);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(launched.size(), 2u);
  EXPECT_EQ(launched[0].first, 0u);
  EXPECT_EQ(launched[1].first, 1u);
  // A slot reused after its wave halts launches again, with a new generation.
  const uint64_t generation = first->dispatch_generation();
  first->halt();
  // A third workgroup takes the slot the first wave freed.
  ASSERT_EQ(cu->dispatch_wf(/*wg_id=*/2, kProgram, kSgprs, kVgprs), first);
  ASSERT_EQ(launched.size(), 3u);
  EXPECT_EQ(launched[2].first, 0u);
  EXPECT_NE(launched[2].second, generation);
  first->halt();
  second->halt();
}

} // namespace
