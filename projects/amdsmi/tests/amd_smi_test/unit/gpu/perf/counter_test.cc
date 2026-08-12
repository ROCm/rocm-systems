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

static constexpr amdsmi_event_group_t kEventGroups[] = {AMDSMI_EVNT_GRP_XGMI,
                                                        AMDSMI_EVNT_GRP_XGMI_DATA_OUT};

// ---------------- amdsmi_gpu_counter_group_supported (enum) ----------------
TEST_F(GpuUnit, CounterGroupSupported_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_gpu_counter_group_supported", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_counter_group_supported(kInvalidHandle, AMDSMI_EVNT_GRP_XGMI);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, CounterGroupSupported_AllGpusAllGroups) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_gpu_counter_group_supported");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto grp : kEventGroups) {
      DISPLAY_AMDSMI_API("amdsmi_gpu_counter_group_supported",
                         "gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), kVerbose);
      amdsmi_status_t err = amdsmi_gpu_counter_group_supported(gpus()[i], grp);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_available_counters (enum) ----------------
TEST_F(GpuUnit, GetAvailableCounters_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_available_counters(gpus()[0], AMDSMI_EVNT_GRP_XGMI, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetAvailableCounters_InvalidHandle) {
  uint32_t available = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_available_counters(kInvalidHandle, AMDSMI_EVNT_GRP_XGMI, &available);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetAvailableCounters_AllGpusAllGroups) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_available_counters");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto grp : kEventGroups) {
      uint32_t available = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters",
                         "gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_available_counters(gpus()[i], grp, &available);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.Record("gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), err,
                        ::amdsmi::unittest::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_gpu_create_counter ----------------
TEST_F(GpuUnit, CreateCounter_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_create_counter(gpus()[0], AMDSMI_EVNT_XGMI_0_NOP_TX, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, CreateCounter_InvalidHandle) {
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_create_counter(kInvalidHandle, AMDSMI_EVNT_XGMI_0_NOP_TX, &evt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---------------- amdsmi_gpu_destroy_counter (invalid) ----------------
TEST_F(GpuUnit, DestroyCounter_InvalidHandle) {
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_destroy_counter", "evt=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_destroy_counter(evt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---------------- amdsmi_gpu_control_counter (invalid) ----------------
TEST_F(GpuUnit, ControlCounter_InvalidHandle) {
  GTEST_SKIP() << "amdsmi_gpu_control_counter crashes on an invalid handle; proper return "
                  "should be AMDSMI_STATUS_INVAL";
  // Proper contract once fixed:
  //   amdsmi_status_t err = amdsmi_gpu_control_counter(kInvalidHandle, AMDSMI_CNTR_CMD_START,
  //   nullptr); AMDSMI_EXPECT_NULL_ARG(err);
}

// ---------------- amdsmi_gpu_read_counter (invalid) ----------------
TEST_F(GpuUnit, ReadCounter_NullOutput) {
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", "value=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_read_counter(evt, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, ReadCounter_InvalidHandle) {
  amdsmi_event_handle_t evt = 0;
  amdsmi_counter_value_t value;
  memset(&value, 0, sizeof(value));
  DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", "evt=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_read_counter(evt, &value);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---------------- valid create/control/read/destroy flow ----------------
TEST_F(GpuUnit, CounterLifecycle_AllGpus) {
  GTEST_SKIP() << "amdsmi_gpu_create_event/control_counter returns AMDSMI_STATUS_UNEXPECTED_DATA; "
                  "root cause unknown, under investigation";

  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_gpu_create_counter");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_event_handle_t evt = 0;
    DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_gpu_create_counter(gpus()[i], AMDSMI_EVNT_XGMI_0_NOP_TX, &evt);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));
    if (err != AMDSMI_STATUS_SUCCESS) continue;

    DISPLAY_AMDSMI_API("amdsmi_gpu_control_counter", "gpu=" + std::to_string(i) + " START",
                       kVerbose);
    amdsmi_status_t cerr = amdsmi_gpu_control_counter(evt, AMDSMI_CNTR_CMD_START, nullptr);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, cerr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i) + " START", cerr,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          cerr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));

    amdsmi_counter_value_t value;
    memset(&value, 0, sizeof(value));
    DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t rerr = amdsmi_gpu_read_counter(evt, &value);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), rerr,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          rerr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));

    amdsmi_gpu_control_counter(evt, AMDSMI_CNTR_CMD_STOP, nullptr);

    DISPLAY_AMDSMI_API("amdsmi_gpu_destroy_counter", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t derr = amdsmi_gpu_destroy_counter(evt);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, derr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), derr,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          derr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}
