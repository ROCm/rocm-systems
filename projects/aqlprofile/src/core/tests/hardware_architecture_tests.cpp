//Copyright © Advanced Micro Devices, Inc., or its affiliates.
//SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "core/hardware_architecture.hpp"
#include "pm4/cmd_builder.h"
#include "def/gpu_block_info.h"

namespace aql_profile {
namespace {

// Mock block info for testing
static CounterRegInfo test_counter_reg = {
    .select_addr = Register(0),
    .control_addr = Register(0),
    .register_addr_lo = Register(0),
    .register_addr_hi = Register(0),
    .select1_addr = Register(0),
};
static const GpuBlockInfo test_block = {
    .name = "TestBlock",
    .id = 0,
    .instance_count = 4,
    .event_id_max = 100,
    .counter_count = 2,
    .counter_reg_info = &test_counter_reg,
    .select_value = nullptr,
    .attr = CounterBlockSeAttr,  // Per-SE block
    .delay_info = {Register(), nullptr},
    .spm_block_id = 0,
};

static const GpuBlockInfo test_block_wgp = {
    .name = "TestBlockWGP",
    .id = 1,
    .instance_count = 1,
    .event_id_max = 50,
    .counter_count = 1,
    .counter_reg_info = &test_counter_reg,
    .select_value = nullptr,
    .attr = CounterBlockWgpAttr,  // Per-WGP block
    .delay_info = {Register(), nullptr},
    .spm_block_id = 0,
};

// Concrete implementation for testing
class TestArchitecture : public HardwareArchitecture {
 public:
  TestArchitecture() {
    config_.gfxip = "test100";
    config_.name = "Test GPU";
    config_.se_count = 4;
    config_.sa_per_se_count = 2;
    config_.cu_count = 64;
    config_.wgp_count = 32;
    config_.xcc_count = 1;
  }

  const HardwareConfig& GetConfig() const override { return config_; }
  const RegisterSchema& GetRegisterSchema() const override { return schema_; }

  const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const override {
    if (block_id == 0) return &test_block;
    if (block_id == 1) return &test_block_wgp;
    return nullptr;
  }

  uint32_t FindBlockByName(const char* name) const override {
    if (strcmp(name, "TestBlock") == 0) return 0;
    if (strcmp(name, "TestBlockWGP") == 0) return 1;
    return UINT32_MAX;
  }

  uint32_t GetBlockCount() const override { return 2; }

  pm4_builder::CmdBuilder* CreateCmdBuilder() const override { return nullptr; }

 private:
  HardwareConfig config_;
  RegisterSchema schema_;
};

TEST(HardwareArchitectureTest, GetConfigReturnsCorrectValues) {
  TestArchitecture arch;
  const auto& config = arch.GetConfig();

  EXPECT_EQ(config.gfxip, "test100");
  EXPECT_EQ(config.name, "Test GPU");
  EXPECT_EQ(config.se_count, 4u);
  EXPECT_EQ(config.sa_per_se_count, 2u);
  EXPECT_EQ(config.cu_count, 64u);
  EXPECT_EQ(config.wgp_count, 32u);
}

TEST(HardwareArchitectureTest, GetBlockInfoById) {
  TestArchitecture arch;

  const auto* block = arch.GetBlockInfo(0);
  ASSERT_NE(block, nullptr);
  EXPECT_STREQ(block->name, "TestBlock");
  EXPECT_EQ(block->instance_count, 4u);
}

TEST(HardwareArchitectureTest, GetBlockInfoInvalidId) {
  TestArchitecture arch;

  const auto* block = arch.GetBlockInfo(999);
  EXPECT_EQ(block, nullptr);
}

TEST(HardwareArchitectureTest, FindBlockByName) {
  TestArchitecture arch;

  uint32_t block_id = arch.FindBlockByName("TestBlock");
  EXPECT_EQ(block_id, 0u);
}

TEST(HardwareArchitectureTest, FindBlockByNameNotFound) {
  TestArchitecture arch;

  uint32_t block_id = arch.FindBlockByName("NonExistent");
  EXPECT_EQ(block_id, UINT32_MAX);
}

TEST(HardwareArchitectureTest, GetBlockCount) {
  TestArchitecture arch;

  EXPECT_EQ(arch.GetBlockCount(), 2u);
}

TEST(HardwareArchitectureTest, GetNumWGPs) {
  TestArchitecture arch;

  EXPECT_EQ(arch.GetNumWGPs(), 32u);
}

TEST(HardwareArchitectureTest, GetDescriptionFormat) {
  TestArchitecture arch;

  std::string desc = arch.GetDescription();
  EXPECT_EQ(desc, "Test GPU (test100)");
}

TEST(HardwareArchitectureTest, DefaultArchitectureVersions) {
  TestArchitecture arch;

  // Default implementations should all return false
  EXPECT_FALSE(arch.IsGFX9());
  EXPECT_FALSE(arch.IsGFX10());
  EXPECT_FALSE(arch.IsGFX11());
  EXPECT_FALSE(arch.IsGFX12());
  EXPECT_FALSE(arch.IsMI100());
  EXPECT_FALSE(arch.IsMI200());
  EXPECT_FALSE(arch.IsMI300());
  EXPECT_FALSE(arch.IsMI350());
}

TEST(HardwareArchitectureTest, GetNumEventsForBlock_PerSE) {
  TestArchitecture arch;

  // test_block has CounterBlockSeAttr, so should multiply by SE count
  size_t num_events = arch.GetNumEventsForBlock(0);
  EXPECT_EQ(num_events, 4u);  // 4 SEs
}

TEST(HardwareArchitectureTest, GetNumEventsForBlock_PerWGP) {
  TestArchitecture arch;

  // test_block_wgp has CounterBlockWgpAttr, so should multiply by WGP count
  size_t num_events = arch.GetNumEventsForBlock(1);
  EXPECT_EQ(num_events, 32u);  // 32 WGPs
}

TEST(HardwareArchitectureTest, GetNumEventsForBlock_InvalidBlock) {
  TestArchitecture arch;

  size_t num_events = arch.GetNumEventsForBlock(999);
  EXPECT_EQ(num_events, 0u);
}

TEST(HardwareArchitectureTest, GetBytesNeededForBlock) {
  TestArchitecture arch;

  // test_block: 4 SEs * 1 XCC * 8 bytes = 32 bytes
  size_t bytes = arch.GetBytesNeededForBlock(0);
  EXPECT_EQ(bytes, 32u);
}

TEST(HardwareArchitectureTest, GetBytesNeededForBlock_MultiXCC) {
  // Create architecture with multiple XCCs
  class MultiXCCArch : public TestArchitecture {
   public:
    MultiXCCArch() {
      auto& cfg = const_cast<HardwareConfig&>(GetConfig());
      cfg.xcc_count = 4;
    }
  };

  MultiXCCArch arch;

  // test_block: 4 SEs * 4 XCC * 8 bytes = 128 bytes
  size_t bytes = arch.GetBytesNeededForBlock(0);
  EXPECT_EQ(bytes, 128u);
}

TEST(HardwareArchitectureTest, AccumulatorIDsThrowByDefault) {
  TestArchitecture arch;

  EXPECT_THROW(arch.GetAccumLowID(), std::runtime_error);
  EXPECT_THROW(arch.GetAccumHiID(), std::runtime_error);
}

TEST(HardwareArchitectureTest, GetSpmSampleDelayMax) {
  TestArchitecture arch;
  auto& cfg = const_cast<HardwareConfig&>(arch.GetConfig());
  cfg.spm_sample_delay_max = 0x34;

  EXPECT_EQ(arch.GetSpmSampleDelayMax(), 0x34u);
}

}  // namespace
}  // namespace aql_profile
