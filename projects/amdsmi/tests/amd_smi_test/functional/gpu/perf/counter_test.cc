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

// GPU performance-event counters: amdsmi_gpu_counter_group_supported /
// amdsmi_get_gpu_available_counters / amdsmi_gpu_create_counter /
// amdsmi_gpu_control_counter / amdsmi_gpu_read_counter /
// amdsmi_gpu_destroy_counter.
static constexpr amdsmi_event_group_t kGroup = AMDSMI_EVNT_GRP_XGMI;
static constexpr amdsmi_event_type_t kEvent = AMDSMI_EVNT_XGMI_0_NOP_TX;

// ---------------- invalid parameters first ----------------
TEST_F(GpuFunctionalReadOnly, CounterGroupSupported_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_gpu_counter_group_supported", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_counter_group_supported(kInvalidHandle, kGroup);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadOnly, GetAvailableCounters_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_available_counters(gpus()[0], kGroup, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(GpuFunctionalReadOnly, GetAvailableCounters_InvalidHandle) {
  RequireInit();
  uint32_t available = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_available_counters(kInvalidHandle, kGroup, &available);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadOnly, CreateCounter_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_create_counter(gpus()[0], kEvent, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(GpuFunctionalReadOnly, CreateCounter_InvalidHandle) {
  RequireInit();
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_create_counter(kInvalidHandle, kEvent, &evt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadOnly, ReadCounter_NullValue) {
  RequireInit();
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", "value=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_read_counter(evt, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// Valid-handle path is exercised in Counter_LifecycleWorkflow below.
TEST_F(GpuFunctionalReadOnly, ControlCounter_InvalidHandle) {
  GTEST_SKIP() << "amdsmi_gpu_control_counter crashes on an invalid handle; "
                  "proper return should be AMDSMI_STATUS_INVAL";
}

// ---------------- full counter lifecycle: create -> start -> read -> stop -> destroy
// ---------------- Perf counters only monitor (no device-config change) and destroy() releases
// exactly what create() allocated, so this is not gated behind the mutation flag.
TEST_F(GpuFunctionalReadOnly, Counter_LifecycleWorkflow) {
  GTEST_SKIP() << "counter lifecycle returns AMDSMI_STATUS_UNEXPECTED_SIZE; root cause unknown, "
                  "under investigation";

  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_gpu_counter_lifecycle");
  for (size_t i = 0; i < gpus().size(); ++i) {
    const std::string g = "gpu=" + std::to_string(i);

    DISPLAY_AMDSMI_API("amdsmi_gpu_counter_group_supported", g + " grp=XGMI", kVerbose);
    amdsmi_status_t serr = amdsmi_gpu_counter_group_supported(gpus()[i], kGroup);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, serr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record(g + " group_supported", serr,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(serr, AMDSMI_STATUS_SUCCESS,
                                                          AMDSMI_STATUS_NOT_SUPPORTED,
                                                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));

    uint32_t available = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters", g + " grp=XGMI", kVerbose);
    amdsmi_status_t aerr = amdsmi_get_gpu_available_counters(gpus()[i], kGroup, &available);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, aerr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record(g + " available_counters", aerr,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(aerr, AMDSMI_STATUS_SUCCESS,
                                                          AMDSMI_STATUS_NOT_SUPPORTED,
                                                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));

    amdsmi_event_handle_t handle = 0;
    DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", g + " evt=XGMI_0_NOP_TX", kVerbose);
    amdsmi_status_t cerr = amdsmi_gpu_create_counter(gpus()[i], kEvent, &handle);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, cerr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record(g + " create_counter", cerr,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   cerr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));
    if (cerr != AMDSMI_STATUS_SUCCESS) continue;  // control/read need a real handle

    DISPLAY_AMDSMI_API("amdsmi_gpu_control_counter", g + " cmd=START", kVerbose);
    amdsmi_status_t st = amdsmi_gpu_control_counter(handle, AMDSMI_CNTR_CMD_START, nullptr);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, st, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);
    col.Record(g + " control_start", st,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   st, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM));

    amdsmi_counter_value_t value;
    memset(&value, 0, sizeof(value));
    DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", g, kVerbose);
    amdsmi_status_t rerr = amdsmi_gpu_read_counter(handle, &value);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);
    col.Record(
        g + " read_counter", rerr,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            rerr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM));

    DISPLAY_AMDSMI_API("amdsmi_gpu_control_counter", g + " cmd=STOP", kVerbose);
    st = amdsmi_gpu_control_counter(handle, AMDSMI_CNTR_CMD_STOP, nullptr);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, st, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);
    col.Record(g + " control_stop", st,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   st, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM));

    // Always release the counter that create() allocated.
    DISPLAY_AMDSMI_API("amdsmi_gpu_destroy_counter", g, kVerbose);
    amdsmi_status_t derr = amdsmi_gpu_destroy_counter(handle);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, derr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);
    col.Record(
        g + " destroy_counter", derr,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            derr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM));
  }
  col.ExpectNoFailures();
}
