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

using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

// amdsmi_set_gpu_uma_carveout: no getter exists; test verifies only the invalid-handle
// and feature-absent rejection paths, not the success path.
TEST_F(GpuFunctionalReadWrite, SetUmaCarveout_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_uma_carveout(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, UmaCarveout_Set) {
  // No getter exists to verify or restore state; SUCCESS is rejected so
  // the first successful run forces adding proper verify/restore logic.
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_gpu_uma_carveout");
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout", "gpu=" + std::to_string(i) + " set=0",
                       kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_uma_carveout(gpus()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_INVAL);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                   AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL));
  }
  col.ExpectNoFailures();
}
