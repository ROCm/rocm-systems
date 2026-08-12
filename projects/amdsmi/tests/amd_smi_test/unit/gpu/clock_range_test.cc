/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Unit tests for smi_amdgpu_parse_od_clk_range(), the pp_od_clk_voltage parser
// behind amd-smi's per-domain min/max clock. Driven over in-memory streams --
// no GPU required. Guards the MI45x case where pp_od_clk_voltage carries no
// OD_FCLK section: the parser must report the domain absent so the caller falls
// back to pp_dpm_fclk instead of reporting a max of 0 MHz.

#include <gtest/gtest.h>

#include <climits>
#include <sstream>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"

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

TEST(GpuUnit, OdClkRangeReadsSclkSection) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_GFX, &max, &min));
  EXPECT_EQ(max, 2100u);
  EXPECT_EQ(min, 500u);
}

TEST(GpuUnit, OdClkRangeReadsMclkSection) {
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
TEST(GpuUnit, OdClkRangeFallsBackWhenFclkSectionMissing) {
  std::istringstream od(kOdNoFclk);
  unsigned int max = 4242;
  unsigned int min = 4242;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 4242u);
  EXPECT_EQ(min, 4242u);
}

// With OD_FCLK present the range is read directly, no fallback.
TEST(GpuUnit, OdClkRangeReadsFclkSectionWhenPresent) {
  std::istringstream od(kOdWithFclk);
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
  EXPECT_TRUE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_DF, &max, &min));
  EXPECT_EQ(max, 1100u);
  EXPECT_EQ(min, 1000u);
}

// A missing/empty pp_od_clk_voltage yields no section -> fall back.
TEST(GpuUnit, OdClkRangeFallsBackOnEmptyStream) {
  std::istringstream od("");
  unsigned int max = 7;
  unsigned int min = 7;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_GFX, &max, &min));
}

// Domains that have no overdrive section (e.g. SOC) are never OD-backed.
TEST(GpuUnit, OdClkRangeRejectsNonOdDomain) {
  std::istringstream od(kOdWithFclk);
  unsigned int max = 7;
  unsigned int min = 7;
  EXPECT_FALSE(smi_amdgpu_parse_od_clk_range(od, AMDSMI_CLK_TYPE_SOC, &max, &min));
}

}  // namespace
