// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vgpr_redispatch_test.cpp
/// @brief Integration coverage for VGPR storage recycling through compute units.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

struct RedispatchCase {
  rj_code_arch_t arch;
  uint32_t wave_size;
  uint32_t vgprs_per_wf;
  const char *name;
};

class VgprRedispatchTest : public ::testing::TestWithParam<RedispatchCase> {};

TEST_P(VgprRedispatchTest, RecycledAllocationStartsZero) {
  const RedispatchCase test_case = GetParam();
  GpuMemory gpu_memory{"vgpr_redispatch_memory"};
  L2Cache l2{"vgpr_redispatch_l2"};

  ComputeUnitCore::Config config{};
  config.arch = test_case.arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = test_case.vgprs_per_wf;
  config.lds_size_kb = 64;

  auto compute_unit = ComputeUnitCore::create("vgpr_redispatch_cu", config, &gpu_memory, &l2);
  compute_unit->set_plugin_group(std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{}));
  ASSERT_EQ(compute_unit->wf_size(), test_case.wave_size);

  Wavefront *first = compute_unit->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*num_sgprs=*/32,
                                               /*num_vgprs=*/test_case.vgprs_per_wf);
  ASSERT_NE(first, nullptr);
  const uint32_t first_sgpr_base = first->sgpr_alloc().base;
  const uint32_t first_base = first->vgpr_alloc().base;
  const uint32_t block_size = compute_unit->vgpr_allocation_block_size();
  compute_unit->write_sgpr(first_sgpr_base, 0x12345678u);
  compute_unit->write_sgpr(first_sgpr_base + 31, 0x9abcdef0u);
  for (uint32_t reg = 0; reg < block_size; ++reg) {
    for (uint32_t lane = 0; lane < test_case.wave_size; ++lane)
      compute_unit->write_vgpr(first_base + reg, lane, 1u + reg * test_case.wave_size + lane);
  }

  first->halt();
  ASSERT_EQ(compute_unit->num_wfs(), 0u);

  Wavefront *second = compute_unit->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*num_sgprs=*/32,
                                                /*num_vgprs=*/test_case.vgprs_per_wf);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(second->sgpr_alloc().base, first_sgpr_base);
  ASSERT_EQ(second->vgpr_alloc().base, first_base);
  EXPECT_EQ(compute_unit->read_sgpr(first_sgpr_base), 0u);
  EXPECT_EQ(compute_unit->read_sgpr(first_sgpr_base + 31), 0u);
  for (uint32_t reg = 0; reg < block_size; ++reg) {
    for (uint32_t lane = 0; lane < test_case.wave_size; ++lane) {
      EXPECT_EQ(compute_unit->read_vgpr(first_base + reg, lane), 0u)
          << "register " << reg << ", lane " << lane;
    }
  }
  second->halt();
}

TEST_P(VgprRedispatchTest, BlockOwnershipTracksDispatchAndRetirement) {
  const RedispatchCase test_case = GetParam();
  GpuMemory gpu_memory{"vgpr_owner_memory"};
  L2Cache l2{"vgpr_owner_l2"};

  ComputeUnitCore::Config config{};
  config.arch = test_case.arch;
  config.num_wf_slots = 2;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = test_case.vgprs_per_wf;
  config.lds_size_kb = 64;

  auto compute_unit = ComputeUnitCore::create("vgpr_owner_cu", config, &gpu_memory, &l2);
  const uint32_t block_size = compute_unit->vgpr_allocation_block_size();
  ASSERT_GT(block_size, 0u);

  Wavefront *first = compute_unit->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*num_sgprs=*/32,
                                               /*num_vgprs=*/test_case.vgprs_per_wf);
  Wavefront *second = compute_unit->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*num_sgprs=*/32,
                                                /*num_vgprs=*/test_case.vgprs_per_wf);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(first->vgpr_alloc().base, second->vgpr_alloc().base);

  EXPECT_EQ(compute_unit->vgpr_owner(first->vgpr_alloc().base), first);
  EXPECT_EQ(compute_unit->vgpr_owner(first->vgpr_alloc().base + block_size - 1), first);
  EXPECT_EQ(compute_unit->vgpr_owner(second->vgpr_alloc().base), second);
  EXPECT_EQ(compute_unit->vgpr_owner(second->vgpr_alloc().base + block_size - 1), second);

  const uint32_t first_base = first->vgpr_alloc().base;
  first->halt();
  EXPECT_EQ(compute_unit->vgpr_owner(first_base), nullptr);
  EXPECT_EQ(compute_unit->vgpr_owner(first_base + block_size - 1), nullptr);
  EXPECT_EQ(compute_unit->vgpr_owner(second->vgpr_alloc().base), second);

  Wavefront *third = compute_unit->dispatch_wf(/*wg_id=*/2, /*pc=*/0, /*num_sgprs=*/32,
                                               /*num_vgprs=*/test_case.vgprs_per_wf);
  ASSERT_NE(third, nullptr);
  ASSERT_EQ(third->vgpr_alloc().base, first_base);
  EXPECT_EQ(compute_unit->vgpr_owner(first_base), third);
  EXPECT_EQ(compute_unit->vgpr_owner(first_base + block_size - 1), third);
  EXPECT_EQ(compute_unit->vgpr_owner(second->vgpr_alloc().base), second);
  third->halt();
  second->halt();
}

INSTANTIATE_TEST_SUITE_P(
    Configurations, VgprRedispatchTest,
    ::testing::Values(RedispatchCase{ROCJITSU_CODE_ARCH_RDNA4, 32, 256, "Wave32"},
                      RedispatchCase{ROCJITSU_CODE_ARCH_RDNA4, 32, 17, "OddBlock"},
                      RedispatchCase{ROCJITSU_CODE_ARCH_CDNA4, 64, 256, "Wave64"},
                      RedispatchCase{ROCJITSU_CODE_ARCH_CDNA5, 32, 1024, "Gfx1250"}),
    [](const ::testing::TestParamInfo<RedispatchCase> &info) { return info.param.name; });

} // namespace
