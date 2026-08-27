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

// HSMP metrics-table APIs (socket scoped). Both only guard the handle and
// dereference the output on success, so their negative coverage is the
// invalid-handle test.

// ---- amdsmi_get_hsmp_metrics_table_version (handle guarded only) ----
TEST_F(CpuIntegration, GetHsmpMetricsTableVersion_InvalidHandle) {
  uint32_t version = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_hsmp_metrics_table_version", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_hsmp_metrics_table_version(kInvalidHandle, &version);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetHsmpMetricsTableVersion_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_hsmp_metrics_table_version");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t version = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_hsmp_metrics_table_version", "cpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_hsmp_metrics_table_version(cpus()[i], &version);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_hsmp_metrics_table (handle guarded only) ----
TEST_F(CpuIntegration, GetHsmpMetricsTable_InvalidHandle) {
  amdsmi_hsmp_metrics_table_t table;
  memset(&table, 0, sizeof(table));
  DISPLAY_AMDSMI_API("amdsmi_get_hsmp_metrics_table", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_hsmp_metrics_table(kInvalidHandle, &table);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(CpuIntegration, GetHsmpMetricsTable_AllCpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_hsmp_metrics_table");
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  for (size_t i = 0; i < cpus().size(); ++i) {
    amdsmi_hsmp_metrics_table_t table;
    memset(&table, 0, sizeof(table));
    DISPLAY_AMDSMI_API("amdsmi_get_hsmp_metrics_table", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_hsmp_metrics_table(cpus()[i], &table);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("cpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}
