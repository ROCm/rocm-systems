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

#include "unit/unit_test_framework.h"

using amdsmi::unittest::IsUntestableHere;
using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

// amdsmi_get_gpu_perf_level / amdsmi_set_gpu_perf_level.
// ---- invalid parameters first ----
TEST_F(GpuFunctionalReadWrite, SetPerfLevel_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_perf_level(kInvalidHandle, AMDSMI_DEV_PERF_LEVEL_AUTO);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---- store -> change -> verify changed -> restore -> verify restored ----
TEST_F(GpuFunctionalReadWrite, PerfLevel_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_gpu_perf_level");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dev_perf_level_t initial;
    if (amdsmi_get_gpu_perf_level(gpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    amdsmi_dev_perf_level_t target = (initial == AMDSMI_DEV_PERF_LEVEL_AUTO)
                                         ? AMDSMI_DEV_PERF_LEVEL_HIGH
                                         : AMDSMI_DEV_PERF_LEVEL_AUTO;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level",
                       "gpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_perf_level(gpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      // verify the value actually changed
      amdsmi_dev_perf_level_t readback;
      if (amdsmi_get_gpu_perf_level(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "gpu=" << i << " set did not take effect";
      }
      // restore initial and verify
      amdsmi_status_t rerr = amdsmi_set_gpu_perf_level(gpus()[i], initial);
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level",
                         "gpu=" + std::to_string(i) + " restore=" + std::to_string(initial),
                         kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "gpu=" << i << " failed to restore perf level";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_perf_level(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "gpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}
