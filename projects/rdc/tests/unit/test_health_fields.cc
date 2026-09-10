/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Pure-function tests for the health-field classification helpers in SmiUtils
// that the probe in rdc_health_set() relies on. No hardware or amdsmi init.

#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/impl/SmiUtils.h"

using amd::rdc::health_field_fallbacks;
using amd::rdc::is_capability_miss;
using amd::rdc::is_health_field;

TEST(HealthFieldsTest, CapabilityMissStatuses) {
  EXPECT_TRUE(is_capability_miss(RDC_ST_NOT_SUPPORTED));
  // AMDSMI_STATUS_INVAL, e.g. the xgmi_error sysfs read on MI300-series and later.
  EXPECT_TRUE(is_capability_miss(RDC_ST_BAD_PARAMETER));
  EXPECT_TRUE(is_capability_miss(RDC_ST_NOT_FOUND));
  EXPECT_TRUE(is_capability_miss(RDC_ST_PERM_ERROR));
}

TEST(HealthFieldsTest, TransientStatusesAreNotCapabilityMisses) {
  EXPECT_FALSE(is_capability_miss(RDC_ST_OK));
  EXPECT_FALSE(is_capability_miss(RDC_ST_SMI_ERROR));
  EXPECT_FALSE(is_capability_miss(RDC_ST_UNKNOWN_ERROR));
  EXPECT_FALSE(is_capability_miss(RDC_ST_NO_DATA));
  EXPECT_FALSE(is_capability_miss(RDC_ST_FILE_ERROR));
  EXPECT_FALSE(is_capability_miss(RDC_ST_CORRUPTED_EEPROM));
}

TEST(HealthFieldsTest, HealthRangeFields) {
  EXPECT_TRUE(is_health_field(RDC_HEALTH_XGMI_ERROR));
  EXPECT_TRUE(is_health_field(RDC_HEALTH_PENDING_PAGE_NUM));
  EXPECT_TRUE(is_health_field(RDC_HEALTH_POWER_THROTTLE_TIME));
  EXPECT_TRUE(is_health_field(RDC_HEALTH_GFX_CLK_LMT_TOTAL_PCT));
}

TEST(HealthFieldsTest, NonHealthFields) {
  EXPECT_FALSE(is_health_field(RDC_FI_GPU_TEMP));
  // Watched by the memory component, but a general telemetry field.
  EXPECT_FALSE(is_health_field(RDC_FI_ECC_UNCORRECT_TOTAL));
  EXPECT_FALSE(is_health_field(RDC_FI_CPU_FIRST));
}

TEST(HealthFieldsTest, FallbackTable) {
  const auto& fallbacks = health_field_fallbacks();
  ASSERT_EQ(1u, fallbacks.count(RDC_HEALTH_XGMI_ERROR));
  EXPECT_EQ(RDC_FI_ECC_XGMI_WAFL_UE, fallbacks.at(RDC_HEALTH_XGMI_ERROR));

  for (const auto& fb : fallbacks) {
    // Both ends of a fallback pair are health fields for log-level purposes.
    EXPECT_TRUE(is_health_field(fb.first));
    EXPECT_TRUE(is_health_field(fb.second));
    // A fallback is never itself a primary with its own fallback.
    EXPECT_EQ(0u, fallbacks.count(fb.second));
  }
}
