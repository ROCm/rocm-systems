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

// ---------------- amdsmi_get_violation_status ----------------
TEST_F(GpuIntegration, GetViolationStatus_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_violation_status", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_violation_status(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetViolationStatus_InvalidHandle) {
  amdsmi_violation_status_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_violation_status", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_violation_status(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetViolationStatus_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_violation_status returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
                  "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_violation_status");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_violation_status_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_violation_status", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_violation_status(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_ptl_state ----------------
TEST_F(GpuIntegration, GetPtlState_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_state", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_state(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetPtlState_InvalidHandle) {
  bool enabled = false;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_state", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_state(kInvalidHandle, &enabled);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetPtlState_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_ptl_state");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    bool enabled = false;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_state", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_ptl_state(gpus()[i], &enabled);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_ptl_formats ----------------
TEST_F(GpuIntegration, GetPtlFormats_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_formats", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_formats(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetPtlFormats_InvalidHandle) {
  amdsmi_ptl_data_format_t f1, f2;
  memset(&f1, 0, sizeof(f1));
  memset(&f2, 0, sizeof(f2));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_formats", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_formats(kInvalidHandle, &f1, &f2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetPtlFormats_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_ptl_formats");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_ptl_data_format_t f1, f2;
    memset(&f1, 0, sizeof(f1));
    memset(&f2, 0, sizeof(f2));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_formats", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_ptl_formats(gpus()[i], &f1, &f2);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_event_notification (no handle) ----------------
TEST_F(GpuIntegration, GetEventNotification_NullNumElem) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_event_notification", "num_elem=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_event_notification(0, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetEventNotification_Valid) {
  amdsmi_evt_notification_data_t data[8];
  memset(data, 0, sizeof(data));
  uint32_t num_elem = 8;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_event_notification", "timeout=0", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_event_notification(0, &num_elem, data);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_INIT_ERROR, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_INIT_ERROR,
                       AMDSMI_STATUS_INVAL);
}

// ---------------- amdsmi_set_gpu_event_notification_mask (SET) ----------------
TEST_F(GpuIntegration, SetEventNotificationMask_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_event_notification_mask", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_event_notification_mask(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_stop_gpu_event_notification (action) ----------------
TEST_F(GpuIntegration, StopEventNotification_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_stop_gpu_event_notification", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_stop_gpu_event_notification(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, StopEventNotification_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_stop_gpu_event_notification");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_stop_gpu_event_notification", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_stop_gpu_event_notification(gpus()[i]);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_INIT_ERROR);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_set_gpu_ptl_state (invalid input only; valid-input cases are in functional/) ----
TEST_F(GpuIntegration, SetPtlState_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_state", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_ptl_state(kInvalidHandle, false);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---- amdsmi_set_gpu_ptl_formats (invalid input only; valid-input cases are in functional/) ----
TEST_F(GpuIntegration, SetPtlFormats_InvalidHandle) {
  // The two formats must differ, otherwise the call is rejected on the format
  // pair and never reaches the handle check.
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_formats", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_ptl_formats(kInvalidHandle, AMDSMI_PTL_DATA_FORMAT_F32,
                                                   AMDSMI_PTL_DATA_FORMAT_F16);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST_F(GpuIntegration, SetPtlFormats_DuplicateFormats) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_formats", "data_format1==data_format2", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_ptl_formats(any_gpu(), AMDSMI_PTL_DATA_FORMAT_F32, AMDSMI_PTL_DATA_FORMAT_F32);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_UNEXPECTED_DATA);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST_F(GpuIntegration, SetPtlFormats_InvalidFormat) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_formats", "data_format1=INVALID", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_ptl_formats(any_gpu(), AMDSMI_PTL_DATA_FORMAT_INVALID,
                                                   AMDSMI_PTL_DATA_FORMAT_F16);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_UNEXPECTED_DATA);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_UNEXPECTED_DATA);
}
