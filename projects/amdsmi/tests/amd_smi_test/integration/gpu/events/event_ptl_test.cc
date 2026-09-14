// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_violation_status ----------------
TEST(GpuIntegration, GetViolationStatus_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_violation_status", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_violation_status(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetViolationStatus_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_violation_status_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_violation_status", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_violation_status(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetViolationStatus_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_get_violation_status returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
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
TEST(GpuIntegration, GetPtlState_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_state", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_state(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetPtlState_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  bool enabled = false;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_state", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_state(kInvalidHandle, &enabled);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetPtlState_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetPtlFormats_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_formats", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_formats(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetPtlFormats_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_ptl_data_format_t f1, f2;
  memset(&f1, 0, sizeof(f1));
  memset(&f2, 0, sizeof(f2));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_ptl_formats", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_ptl_formats(kInvalidHandle, &f1, &f2);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetPtlFormats_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetEventNotification_NullNumElem) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_event_notification", "num_elem=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_event_notification(0, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST(GpuIntegration, GetEventNotification_Valid) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, SetEventNotificationMask_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_event_notification_mask", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_event_notification_mask(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---------------- amdsmi_stop_gpu_event_notification (action) ----------------
TEST(GpuIntegration, StopEventNotification_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_stop_gpu_event_notification", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_stop_gpu_event_notification(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, StopEventNotification_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, SetPtlState_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_state", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_ptl_state(kInvalidHandle, false);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---- amdsmi_set_gpu_ptl_formats (invalid input only; valid-input cases are in functional/) ----
TEST(GpuIntegration, SetPtlFormats_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  // The two formats must differ, otherwise the call is rejected on the format
  // pair and never reaches the handle check.
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_formats", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_ptl_formats(kInvalidHandle, AMDSMI_PTL_DATA_FORMAT_F32,
                                                   AMDSMI_PTL_DATA_FORMAT_F16);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST(GpuIntegration, SetPtlFormats_DuplicateFormats) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_formats", "data_format1==data_format2", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_ptl_formats(any_gpu(), AMDSMI_PTL_DATA_FORMAT_F32, AMDSMI_PTL_DATA_FORMAT_F32);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_UNEXPECTED_DATA);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_UNEXPECTED_DATA);
}

TEST(GpuIntegration, SetPtlFormats_InvalidFormat) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_ptl_formats", "data_format1=INVALID", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_ptl_formats(any_gpu(), AMDSMI_PTL_DATA_FORMAT_INVALID,
                                                   AMDSMI_PTL_DATA_FORMAT_F16);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_UNEXPECTED_DATA);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_UNEXPECTED_DATA);
}
