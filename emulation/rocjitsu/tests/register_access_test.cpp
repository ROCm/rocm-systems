// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_access_test.cpp
/// @brief Tests for the AMDGPU instruction-facing register access facade.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

constexpr uint32_t kSgprsPerWave = 104;
constexpr uint32_t kVgprsPerWave = 256;

struct ReadEvent {
  uint32_t physical_reg = 0;
  uint64_t lane_mask = 0;
  uint8_t byte_mask = 0;
};

class RecordingPlugin : public ExecutionPlugin {
public:
  RecordingPlugin() : ExecutionPlugin("register_access_recorder") {}

  void onAmdgpuReadVgprLanes(const Wavefront *, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    reads.push_back({physical_reg, lane_mask, byte_mask});
  }

  std::vector<ReadEvent> reads;
};

struct Fixture {
  GpuMemory gpu_mem{"register_access_mem"};
  L2Cache l2{"register_access_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;
  RecordingPlugin *plugin = nullptr;
  Wavefront *wf = nullptr;

  Fixture() {
    ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = kSgprsPerWave;
    cfg.vgprs_per_wf = kVgprsPerWave;
    cfg.lds_size_kb = 64;
    cu = ComputeUnitCore::create("register_access_cu", cfg, &gpu_mem, &l2);

    plugin_group = std::make_shared<ExecutionPluginGroup>();
    auto recorder = std::make_unique<RecordingPlugin>();
    plugin = recorder.get();
    plugin_group->add(std::move(recorder));
    cu->set_plugin_group(plugin_group);

    wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, kSgprsPerWave, kVgprsPerWave);
  }

  uint32_t vgpr_base() const { return wf->vgpr_alloc().base; }
};

TEST(RegisterAccessTest, ReadRegionObservesAllRegistersAndReturnsLaneSpans) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  fx.cu->write_vgpr(base + 3, 0, 0x1111u);
  fx.cu->write_vgpr(base + 3, 5, 0x3333u);
  fx.cu->write_vgpr(base + 4, 0, 0x2222u);
  fx.cu->write_vgpr(base + 4, 5, 0x4444u);

  RegisterAccess regs(*fx.cu);
  auto region = regs.read_vgpr_region(base + 3, /*reg_count=*/2, /*lane_mask=*/0x21,
                                      /*byte_mask=*/0xF);

  ASSERT_EQ(fx.plugin->reads.size(), 2u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + 3);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 0x21u);
  EXPECT_EQ(fx.plugin->reads[0].byte_mask, 0xFu);
  EXPECT_EQ(fx.plugin->reads[1].physical_reg, base + 4);
  EXPECT_EQ(fx.plugin->reads[1].lane_mask, 0x21u);
  EXPECT_EQ(fx.plugin->reads[1].byte_mask, 0xFu);

  EXPECT_EQ(region.lanes(0)[0], 0x1111u);
  EXPECT_EQ(region.lanes(0)[5], 0x3333u);
  EXPECT_EQ(region.lanes(1)[0], 0x2222u);
  EXPECT_EQ(region.lanes(1)[5], 0x4444u);
}

TEST(RegisterAccessTest, WriteRegionDoesNotObserveReadsAndHonorsLaneMask) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  for (uint32_t lane = 0; lane < fx.wf->wf_size(); ++lane)
    fx.cu->write_vgpr(base + 7, lane, 0xAAAA0000u | lane);

  RegisterAccess regs(*fx.cu);
  auto region = regs.write_vgpr_region(base + 7, /*reg_count=*/1, /*lane_mask=*/0x5);
  region.set_lane(/*relative_reg=*/0, /*lane=*/0, 0x100u);
  region.set_lane(/*relative_reg=*/0, /*lane=*/1, 0x200u);
  region.set_lane(/*relative_reg=*/0, /*lane=*/2, 0x300u);

  EXPECT_TRUE(fx.plugin->reads.empty());
  EXPECT_EQ(region.lanes(0)[0], 0x100u);
  EXPECT_EQ(region.lanes(0)[1], 0xAAAA0001u);
  EXPECT_EQ(region.lanes(0)[2], 0x300u);
}

TEST(RegisterAccessTest, ReadWriteRegionObservesThenAllowsWrites) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  fx.cu->write_vgpr(base + 9, 3, 0x1234u);

  RegisterAccess regs(*fx.cu);
  auto region = regs.readwrite_vgpr_region(base + 9, /*reg_count=*/1, /*lane_mask=*/0x8);

  ASSERT_EQ(fx.plugin->reads.size(), 1u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + 9);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 0x8u);
  EXPECT_EQ(region.read_lanes(0)[3], 0x1234u);

  region.write().set_lane(/*relative_reg=*/0, /*lane=*/3, 0x5678u);
  EXPECT_EQ(region.read_lanes(0)[3], 0x5678u);
}

TEST(RegisterAccessTest, Scalar64ReadObservesBothRegisters) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  uint32_t base = fx.vgpr_base();
  fx.cu->write_vgpr(base + 11, 6, 0x89ABCDEFu);
  fx.cu->write_vgpr(base + 12, 6, 0x01234567u);

  RegisterAccess regs(*fx.cu);
  EXPECT_EQ(regs.read_vgpr64(base + 11, 6), 0x0123456789ABCDEFull);

  ASSERT_EQ(fx.plugin->reads.size(), 2u);
  EXPECT_EQ(fx.plugin->reads[0].physical_reg, base + 11);
  EXPECT_EQ(fx.plugin->reads[0].lane_mask, 1ULL << 6);
  EXPECT_EQ(fx.plugin->reads[1].physical_reg, base + 12);
  EXPECT_EQ(fx.plugin->reads[1].lane_mask, 1ULL << 6);
}

} // namespace
