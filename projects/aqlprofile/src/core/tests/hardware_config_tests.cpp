//Copyright © Advanced Micro Devices, Inc., or its affiliates.
//SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "core/hardware_config.hpp"

namespace aql_profile {
namespace {

TEST(HardwareConfigTest, DefaultConstructor) {
  HardwareConfig config;

  EXPECT_EQ(config.gfxip, "unknown");
  EXPECT_EQ(config.name, "Unknown");
  EXPECT_EQ(config.se_count, 0u);
  EXPECT_EQ(config.sa_per_se_count, 0u);
  EXPECT_EQ(config.cu_count, 0u);
  EXPECT_EQ(config.xcc_count, 1u);
  EXPECT_EQ(config.aid_count, 1u);
  EXPECT_TRUE(config.supports_pmc);
  EXPECT_FALSE(config.supports_spm);
  EXPECT_FALSE(config.supports_sqtt);
}

TEST(HardwareConfigTest, SingleXCCDetection) {
  HardwareConfig config;
  config.xcc_count = 1;

  EXPECT_FALSE(config.IsMultiXCC());
}

TEST(HardwareConfigTest, MultiXCCDetection) {
  HardwareConfig config;
  config.xcc_count = 4;

  EXPECT_TRUE(config.IsMultiXCC());
}

TEST(HardwareConfigTest, MI300Detection) {
  HardwareConfig config;
  config.aid_count = 4;
  config.xcc_count = 8;

  EXPECT_TRUE(config.IsMI300Series());
}

TEST(HardwareConfigTest, NonMI300Detection) {
  HardwareConfig config;
  config.aid_count = 1;
  config.xcc_count = 1;

  EXPECT_FALSE(config.IsMI300Series());
}

TEST(HardwareConfigTest, GetTotalWGPsExplicit) {
  HardwareConfig config;
  config.wgp_count = 32;

  EXPECT_EQ(config.GetTotalWGPs(), 32u);
}

TEST(HardwareConfigTest, GetTotalWGPsFallback) {
  HardwareConfig config;
  config.wgp_count = 0;
  config.cu_count = 64;

  // Fallback: approximate from CU count
  EXPECT_EQ(config.GetTotalWGPs(), 32u);
}

TEST(HardwareConfigTest, GetSEPerXCC) {
  HardwareConfig config;
  config.se_count = 8;
  config.xcc_count = 2;

  EXPECT_EQ(config.GetSEPerXCC(), 4u);
}

TEST(HardwareConfigTest, MI200Config) {
  HardwareConfig config;
  config.gfxip = "gfx90a";
  config.name = "MI200";
  config.se_count = 8;
  config.sa_per_se_count = 2;
  config.cu_count = 104;
  config.xcc_count = 1;
  config.aid_count = 1;
  config.supports_pmc = true;
  config.supports_spm = true;
  config.supports_sqtt = true;
  config.has_spm_core1 = true;

  EXPECT_EQ(config.gfxip, "gfx90a");
  EXPECT_FALSE(config.IsMultiXCC());
  EXPECT_FALSE(config.IsMI300Series());
  EXPECT_TRUE(config.supports_spm);
  EXPECT_TRUE(config.has_spm_core1);
}

TEST(HardwareConfigTest, MI300Config) {
  HardwareConfig config;
  config.gfxip = "gfx940";
  config.name = "MI300";
  config.se_count = 8;
  config.sa_per_se_count = 2;
  config.cu_count = 304;
  config.xcc_count = 8;
  config.aid_count = 4;
  config.supports_pmc = true;
  config.supports_spm = true;
  config.supports_sqtt = true;
  config.has_aid_aware_counters = true;

  EXPECT_EQ(config.gfxip, "gfx940");
  EXPECT_TRUE(config.IsMultiXCC());
  EXPECT_TRUE(config.IsMI300Series());
  EXPECT_TRUE(config.has_aid_aware_counters);
  EXPECT_EQ(config.GetSEPerXCC(), 1u);
}

TEST(HardwareConfigTest, GFX11Config) {
  HardwareConfig config;
  config.gfxip = "gfx1100";
  config.name = "Navi31";
  config.se_count = 6;
  config.sa_per_se_count = 2;
  config.cu_count = 96;
  config.wgp_count = 48;
  config.xcc_count = 1;
  config.supports_pmc = true;
  config.supports_spm = false;
  config.supports_sqtt = true;

  EXPECT_EQ(config.gfxip, "gfx1100");
  EXPECT_FALSE(config.IsMultiXCC());
  EXPECT_EQ(config.GetTotalWGPs(), 48u);
}

}  // namespace
}  // namespace aql_profile
