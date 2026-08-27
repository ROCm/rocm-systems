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

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// NIC exposes only getters. These tests verify each getter returns an
// acceptable status and is stable (same status) across repeated reads.
TEST_F(NicFunctionalReadOnly, DeviceBdf_InvalidHandle) {
  RequireInit();
  amdsmi_bdf_t bdf;
  memset(&bdf, 0, sizeof(bdf));
  DISPLAY_AMDSMI_API("amdsmi_get_nic_device_bdf", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_device_bdf(kInvalidHandle, &bdf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicFunctionalReadOnly, AsicInfo_NullOutput) {
  if (nics().empty()) GTEST_SKIP() << "No NIC devices";
  DISPLAY_AMDSMI_API("amdsmi_get_nic_asic_info", "nic=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_asic_info(nics()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

// Getter stability: BDF must be identical across two reads.
TEST_F(NicFunctionalReadOnly, DeviceBdf_Stable) {
  if (nics().empty()) GTEST_SKIP() << "No NIC devices";
  amdsmi::test::StatusCollector col("amdsmi_get_nic_device_bdf");
  for (size_t i = 0; i < nics().size(); ++i) {
    amdsmi_bdf_t a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    DISPLAY_AMDSMI_API("amdsmi_get_nic_device_bdf", "nic=" + std::to_string(i), kVerbose);
    amdsmi_status_t e1 = amdsmi_get_nic_device_bdf(nics()[i], &a);
    amdsmi_status_t e2 = amdsmi_get_nic_device_bdf(nics()[i], &b);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record("nic=" + std::to_string(i), e1,
               ::amdsmi::test::AmdsmiStatusIsExpected(e1, AMDSMI_STATUS_SUCCESS,
                                                      AMDSMI_STATUS_NOT_SUPPORTED,
                                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    EXPECT_EQ(e1, e2) << "nic=" << i << " getter status not stable";
    if (e1 == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(a.as_uint, b.as_uint) << "nic=" << i << " BDF not stable across reads";
    }
  }
  col.ExpectNoFailures();
}
