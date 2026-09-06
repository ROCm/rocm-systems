// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for smi_amdgpu_parse_od_clk_range(), the pp_od_clk_voltage parser
// behind amd-smi's per-domain min/max clock. Driven over in-memory streams --
// no GPU required. Guards the MI45x case where pp_od_clk_voltage carries no
// OD_FCLK section: the parser must report the domain absent so the caller falls
// back to pp_dpm_fclk instead of reporting a max of 0 MHz.

#include <gtest/gtest.h>

#include <climits>
#include <sstream>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_clk_testing.h"
#include "unit_fixtures.h"

namespace {

// pp_od_clk_voltage as seen on MI45x: SCLK and MCLK sections plus an OD_RANGE
// footer, but no OD_FCLK section.
constexpr char kOdNoFclk[] =
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "OD_MCLK:\n"
    "0: 900Mhz\n"
    "1: 1200Mhz\n"
    "OD_RANGE:\n"
    "SCLK:     500Mhz        2100Mhz\n"
    "MCLK:     900Mhz        1200Mhz\n";

// Same layout with an explicit OD_FCLK section present.
constexpr char kOdWithFclk[] =
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n";

// Bare GFXCLK/MCLK/FCLK aliases in place of the OD_* headers, as some GPUs emit.
constexpr char kOdAliasHeaders[] =
    "GFXCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "MCLK:\n"
    "0: 900Mhz\n"
    "1: 1200Mhz\n"
    "FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n";

// OD_FCLK present but every level reads 0 -- no usable range.
constexpr char kOdFclkAllZero[] =
    "OD_FCLK:\n"
    "0: 0Mhz\n"
    "1: 0Mhz\n";

// OD_FCLK with a non-conforming line between two valid levels.
constexpr char kOdFclkGarbageLine[] =
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "not-a-level\n"
    "1: 1100Mhz\n";

// "Old Format" layout: OD_FCLK is immediately followed by OD_VDDC_CURVE, whose
// "idx: freq volt" lines resemble level lines and must not fold into the range.
constexpr char kOdFclkThenCurve[] =
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "1: 2100Mhz\n"
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n"
    "OD_VDDC_CURVE:\n"
    "0: 500Mhz 700mV\n"
    "1: 1354Mhz 860mV\n"
    "2: 2000Mhz 1150mV\n";

// Two OD_FCLK sections: levels from every occurrence merge into one range.
constexpr char kOdFclkDuplicate[] =
    "OD_FCLK:\n"
    "0: 1000Mhz\n"
    "1: 1100Mhz\n"
    "OD_SCLK:\n"
    "0: 500Mhz\n"
    "OD_FCLK:\n"
    "0: 900Mhz\n"
    "1: 2000Mhz\n";

// A single-level (locked-clock) section: min == max.
constexpr char kOdFclkSingleLevel[] =
    "OD_FCLK:\n"
    "0: 1500Mhz\n";

TEST_F(GpuUnit, OdClkRangeReadsSclkSection) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_GFX, &max, &min));
  EXPECT_EQ(max, 2100u);
  EXPECT_EQ(min, 500u);
}

TEST_F(GpuUnit, OdClkRangeReadsMclkSection) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_MEM, &max, &min));
  EXPECT_EQ(max, 1200u);
  EXPECT_EQ(min, 900u);
}

// The regression: with no OD_FCLK section the parser reports "absent" (false)
// and leaves the out-params untouched, so the caller derives FCLK min/max from
// pp_dpm_fclk rather than reporting 0.
TEST_F(GpuUnit, OdClkRangeFallsBackWhenFclkSectionMissing) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 4242;
  unsigned int min = 4242;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 4242u);
  EXPECT_EQ(min, 4242u);
}

// With OD_FCLK present the range is read directly, no fallback.
TEST_F(GpuUnit, OdClkRangeReadsFclkSectionWhenPresent) {
  std::istringstream od(kOdWithFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// A missing/empty pp_od_clk_voltage yields no section -> fall back.
TEST_F(GpuUnit, OdClkRangeFallsBackOnEmptyStream) {
  std::istringstream od("");
  unsigned int max = 7;
  unsigned int min = 7;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_GFX, &max, &min));
}

// Domains that have no overdrive section (e.g. SOC) are never OD-backed.
TEST_F(GpuUnit, OdClkRangeRejectsNonOdDomain) {
  std::istringstream od(kOdWithFclk);
  unsigned int max = 7;
  unsigned int min = 7;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_SOC, &max, &min));
}

// The bare GFXCLK/MCLK/FCLK aliases are accepted just like the OD_* headers.
TEST_F(GpuUnit, OdClkRangeReadsAliasHeaders) {
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  std::istringstream gfx(kOdAliasHeaders);
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(gfx, AMDSMI_CLK_TYPE_GFX, &max, &min));
  EXPECT_EQ(max, 2100u);
  EXPECT_EQ(min, 500u);

  max = 0;
  min = UINT_MAX;
  std::istringstream mem(kOdAliasHeaders);
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(mem, AMDSMI_CLK_TYPE_MEM, &max, &min));
  EXPECT_EQ(max, 1200u);
  EXPECT_EQ(min, 900u);

  max = 0;
  min = UINT_MAX;
  std::istringstream df(kOdAliasHeaders);
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(df, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// Section present but every level parses to 0 -> reported absent so the caller
// still falls back to pp_dpm_*.
TEST_F(GpuUnit, OdClkRangeFallsBackWhenAllLevelsZero) {
  std::istringstream od(kOdFclkAllZero);
  unsigned int max = 555;
  unsigned int min = 555;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 555u);
  EXPECT_EQ(min, 555u);
}

// A non-conforming line inside the section is skipped, not treated as its end,
// so levels on both sides still contribute to the range.
TEST_F(GpuUnit, OdClkRangeSkipsMalformedLineWithinSection) {
  std::istringstream od(kOdFclkGarbageLine);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// An unrecognized section header (OD_VDDC_CURVE) ends FCLK parsing, so its curve
// lines are not mistaken for FCLK levels and do not inflate the max.
TEST_F(GpuUnit, OdClkRangeStopsAtUnrecognizedSectionHeader) {
  std::istringstream od(kOdFclkThenCurve);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// Repeated section headers merge: levels from every OD_FCLK occurrence feed one
// range. No live sysfs file repeats a header; this pins the contract.
TEST_F(GpuUnit, OdClkRangeMergesRepeatedSectionHeaders) {
  std::istringstream od(kOdFclkDuplicate);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 2000u);
  EXPECT_EQ(min, 900u);
}

// A single-level (locked-clock) section reports min == max.
TEST_F(GpuUnit, OdClkRangeReadsSingleLevelSection) {
  std::istringstream od(kOdFclkSingleLevel);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1500u);
  EXPECT_EQ(min, 1500u);
}

}  // namespace
