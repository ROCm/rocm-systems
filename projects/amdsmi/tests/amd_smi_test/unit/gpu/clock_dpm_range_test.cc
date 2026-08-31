// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Parsing tests for smi_amdgpu_parse_clk_ranges() over synthetic pp_dpm_*
// tables; no GPU required. Covers the deep-sleep floor folding into min_clk
// (the case that made VCLK/DCLK MIN_CLK report the nominal clock).

#include <gtest/gtest.h>

#include <climits>
#include <sstream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_clk_testing.h"

namespace {

struct ClkRanges {
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  unsigned int dpm = 0;
  unsigned int sleep = UINT_MAX;
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
};

ClkRanges ParseTable(const std::string& table, bool derive_minmax = true) {
  ClkRanges r;
  std::istringstream stream(table);
  r.status = smi_amdgpu_parse_clk_ranges(stream, derive_minmax, &r.max, &r.min, &r.dpm, &r.sleep);
  return r;
}

// The deep-sleep "S:" line is the power-gated idle floor; it must define
// min_clk instead of the lowest numbered DPM level. Without the fold, min_clk
// collapsed onto the 2100 MHz nominal VCLK.
TEST(GpuUnit, DpmClkRangeDeepSleepFloorSetsMin) {
  const ClkRanges r = ParseTable("S: 66Mhz *\n1: 2100Mhz\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.min, 66u);
  EXPECT_EQ(r.max, 2100u);
  EXPECT_EQ(r.sleep, 66u);
}

// DCLK exposes the same shape with a lower nominal clock (1700 MHz).
TEST(GpuUnit, DpmClkRangeDeepSleepFloorSetsMinDclk) {
  const ClkRanges r = ParseTable("S: 53Mhz *\n1: 1700Mhz\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.min, 53u);
  EXPECT_EQ(r.max, 1700u);
}

// Without a deep-sleep line, min/max come straight from the DPM levels.
TEST(GpuUnit, DpmClkRangeNoDeepSleepUsesDpmLevels) {
  const ClkRanges r = ParseTable("0: 500Mhz\n1: 1100Mhz *\n2: 2100Mhz\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.min, 500u);
  EXPECT_EQ(r.max, 2100u);
  EXPECT_EQ(r.dpm, 2u);
  EXPECT_EQ(r.sleep, UINT_MAX);
}

// A single current-marked level (dpm == 0) pins both min and max to it.
TEST(GpuUnit, DpmClkRangeSingleLevelPinsMinMax) {
  const ClkRanges r = ParseTable("0: 2100Mhz *\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.min, 2100u);
  EXPECT_EQ(r.max, 2100u);
}

// When min/max are supplied by the caller (sclk/mclk/fclk read
// pp_od_clk_voltage), neither the DPM levels nor the deep-sleep floor overwrite
// them; the floor is still captured in sleep for the deep-sleep flag.
TEST(GpuUnit, DpmClkRangeDerivedMinMaxDisabledKeepsCallerRange) {
  ClkRanges r;
  r.max = 3000;
  r.min = 400;
  std::istringstream stream("S: 52Mhz *\n1: 2100Mhz\n");
  r.status = smi_amdgpu_parse_clk_ranges(stream, /*derive_minmax=*/false, &r.max, &r.min, &r.dpm,
                                         &r.sleep);
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.max, 3000u);
  EXPECT_EQ(r.min, 400u);
  EXPECT_EQ(r.sleep, 52u);
}

// A numbered DPM line missing its frequency cannot be parsed and surfaces as an
// I/O error rather than a silently skipped level.
TEST(GpuUnit, DpmClkRangeMalformedDpmLineReturnsIO) {
  const ClkRanges r = ParseTable("1: bad\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_IO);
}

// A deep-sleep "S:" line missing its frequency is reported as missing data.
TEST(GpuUnit, DpmClkRangeMalformedDeepSleepLineReturnsNoData) {
  const ClkRanges r = ParseTable("S: Mhz\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_NO_DATA);
}

// A deep-sleep floor above the lowest DPM level must not raise min_clk; the fold
// only lowers min toward the idle floor, never above the DPM range.
TEST(GpuUnit, DpmClkRangeDeepSleepFloorAboveMinKeepsDpmMin) {
  const ClkRanges r = ParseTable("S: 3000Mhz *\n1: 500Mhz\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.min, 500u);
  EXPECT_EQ(r.max, 500u);
  EXPECT_EQ(r.sleep, 3000u);
}

// A fully clock-gated deep-sleep floor (S: 0Mhz) is a real state, not the
// UINT_MAX "no deep-sleep" sentinel: sleep is 0 and the derived min folds to 0.
TEST(GpuUnit, DpmClkRangeZeroDeepSleepFloorIsValid) {
  const ClkRanges r = ParseTable("S: 0Mhz *\n0: 500Mhz\n1: 1000Mhz\n");
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.sleep, 0u);
  EXPECT_EQ(r.min, 0u);
  EXPECT_EQ(r.max, 1000u);
}

// A lone dpm level (dpm == 0) with caller-supplied min/max (derive_minmax=false)
// must not overwrite that range through the num_dpm==0 fallback; the caller's
// pp_od_clk_voltage range has to stand.
TEST(GpuUnit, DpmClkRangeSingleLevelKeepsCallerRangeWhenDeriveDisabled) {
  ClkRanges r;
  r.max = 2000;
  r.min = 200;
  std::istringstream stream("0: 500Mhz *\n");
  r.status = smi_amdgpu_parse_clk_ranges(stream, /*derive_minmax=*/false, &r.max, &r.min, &r.dpm,
                                         &r.sleep);
  EXPECT_EQ(r.status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(r.max, 2000u);
  EXPECT_EQ(r.min, 200u);
}

}  // namespace
