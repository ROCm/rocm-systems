// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

static constexpr amdsmi_event_group_t kEventGroups[] = {AMDSMI_EVNT_GRP_XGMI,
                                                        AMDSMI_EVNT_GRP_XGMI_DATA_OUT};

// ---------------- amdsmi_gpu_counter_group_supported (enum) ----------------
TEST_F(GpuIntegration, CounterGroupSupported_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_gpu_counter_group_supported", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_counter_group_supported(kInvalidHandle, AMDSMI_EVNT_GRP_XGMI);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, CounterGroupSupported_AllGpusAllGroups) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_gpu_counter_group_supported");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto grp : kEventGroups) {
      DISPLAY_AMDSMI_API("amdsmi_gpu_counter_group_supported",
                         "gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), kVerbose);
      amdsmi_status_t err = amdsmi_gpu_counter_group_supported(gpus()[i], grp);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_available_counters (enum) ----------------
TEST_F(GpuIntegration, GetAvailableCounters_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_available_counters(any_gpu(), AMDSMI_EVNT_GRP_XGMI, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetAvailableCounters_InvalidHandle) {
  uint32_t available = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_available_counters(kInvalidHandle, AMDSMI_EVNT_GRP_XGMI, &available);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetAvailableCounters_AllGpusAllGroups) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_available_counters");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (auto grp : kEventGroups) {
      uint32_t available = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_available_counters",
                         "gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), kVerbose);
      amdsmi_status_t err = amdsmi_get_gpu_available_counters(gpus()[i], grp, &available);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " grp=" + std::to_string(grp), err);
    }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_gpu_create_counter ----------------
TEST_F(GpuIntegration, CreateCounter_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_create_counter(any_gpu(), AMDSMI_EVNT_XGMI_0_NOP_TX, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, CreateCounter_InvalidHandle) {
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_create_counter", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_create_counter(kInvalidHandle, AMDSMI_EVNT_XGMI_0_NOP_TX, &evt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

// ---------------- amdsmi_gpu_destroy_counter (invalid) ----------------
TEST_F(GpuIntegration, DestroyCounter_InvalidHandle) {
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_destroy_counter", "evt=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_destroy_counter(evt);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

// ---------------- amdsmi_gpu_control_counter (invalid) ----------------
TEST_F(GpuIntegration, ControlCounter_InvalidHandle) {
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_control_counter", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_control_counter(evt, AMDSMI_CNTR_CMD_START, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

// ---------------- amdsmi_gpu_read_counter (invalid) ----------------
TEST_F(GpuIntegration, ReadCounter_NullOutput) {
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", "value=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_read_counter(evt, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, ReadCounter_InvalidHandle) {
  amdsmi_event_handle_t evt = 0;
  amdsmi_counter_value_t value;
  memset(&value, 0, sizeof(value));
  DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", "evt=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_read_counter(evt, &value);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST_F(GpuIntegration, ReadCounter_NullValue) {
  RequireInit();
  amdsmi_event_handle_t evt = 0;
  DISPLAY_AMDSMI_API("amdsmi_gpu_read_counter", "value=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_read_counter(evt, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
