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

#include <cstdint>
#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// CPU core/socket boost-limit APIs. The core boost-limit calls derive a core
// index from the handle (cpu_cores()); the socket boost-limit is socket scoped
// (cpus()). All three guard only the handle. The write valid paths reuse an
// in-range boost value read back from the core getter so they do not force an
// arbitrary limit.

// ---- amdsmi_get_cpu_core_boostlimit (handle guarded only, core handle) ----
TEST_F(CpuIntegration, GetCoreBoostlimit_InvalidHandle) {
  uint32_t boost = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_boostlimit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_core_boostlimit(kInvalidHandle, &boost);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(CpuIntegration, GetCoreBoostlimit_AllCores) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_cpu_core_boostlimit");
  if (cpu_cores().empty()) GTEST_SKIP() << "No CPU cores";
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint32_t boost = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_cpu_core_boostlimit", "core=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_cpu_core_boostlimit(cpu_cores()[i], &boost);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("core=" + std::to_string(i), err,
                      ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                             AMDSMI_STATUS_NOT_SUPPORTED,
                                                             AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_set_cpu_core_boostlimit (invalid input only; valid-input cases are in functional/)
// ----
TEST_F(CpuIntegration, SetCoreBoostlimit_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_boostlimit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_core_boostlimit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
// ---- amdsmi_set_cpu_socket_boostlimit (invalid input only; valid-input cases are in functional/)
// ----
TEST_F(CpuIntegration, SetSocketBoostlimit_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_socket_boostlimit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_socket_boostlimit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
