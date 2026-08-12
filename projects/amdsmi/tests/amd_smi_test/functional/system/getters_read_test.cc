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

// System/topology exposes only getters. These tests verify getters return an
// acceptable status and are stable (deterministic) across repeated reads.
TEST_F(SystemFunctionalReadOnly, LibVersion_NullOutput) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_get_lib_version", "version=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_lib_version(nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

// Library version must be identical across repeated reads.
TEST_F(SystemFunctionalReadOnly, LibVersion_Stable) {
  RequireInit();
  amdsmi_version_t a, b;
  memset(&a, 0, sizeof(a));
  memset(&b, 0, sizeof(b));
  DISPLAY_AMDSMI_API("amdsmi_get_lib_version", "read x2", kVerbose);
  amdsmi_status_t e1 = amdsmi_get_lib_version(&a);
  amdsmi_status_t e2 = amdsmi_get_lib_version(&b);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(e1, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (e1 == AMDSMI_STATUS_SUCCESS && e2 == AMDSMI_STATUS_SUCCESS) {
    EXPECT_EQ(a.major, b.major);
    EXPECT_EQ(a.minor, b.minor);
    EXPECT_EQ(a.release, b.release);
  }
}

// Socket count must be identical across repeated reads.
TEST_F(SystemFunctionalReadOnly, SocketHandles_Stable) {
  RequireInit();
  uint32_t c1 = 0, c2 = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_socket_handles", "count x2", kVerbose);
  amdsmi_status_t e1 = amdsmi_get_socket_handles(&c1, nullptr);
  amdsmi_status_t e2 = amdsmi_get_socket_handles(&c2, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(e1, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (e1 == AMDSMI_STATUS_SUCCESS && e2 == AMDSMI_STATUS_SUCCESS) {
    EXPECT_EQ(c1, c2) << "socket count not stable";
  }
}

// Processor type must be identical across repeated reads for each GPU.
TEST_F(SystemFunctionalReadOnly, ProcessorType_Stable) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_get_processor_type");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_processor_type_t t1 = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
    amdsmi_processor_type_t t2 = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
    DISPLAY_AMDSMI_API("amdsmi_get_processor_type", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t e1 = amdsmi_get_processor_type(gpus()[i], &t1);
    amdsmi_status_t e2 = amdsmi_get_processor_type(gpus()[i], &t2);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record("gpu=" + std::to_string(i), e1,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(e1, AMDSMI_STATUS_SUCCESS,
                                                          AMDSMI_STATUS_NOT_SUPPORTED,
                                                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    if (e1 == AMDSMI_STATUS_SUCCESS && e2 == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(t1, t2) << "gpu=" << i << " processor type not stable";
    }
  }
  col.ExpectNoFailures();
}
