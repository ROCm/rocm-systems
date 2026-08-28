// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_gpu_process_isolation ----------------
TEST_F(GpuIntegration, GetProcessIsolation_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_isolation", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_isolation(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetProcessIsolation_InvalidHandle) {
  uint32_t pisolate = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_isolation", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_isolation(kInvalidHandle, &pisolate);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetProcessIsolation_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_process_isolation");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t pisolate = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_isolation", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_process_isolation(gpus()[i], &pisolate);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_process_list ----------------
TEST_F(GpuIntegration, GetProcessList_NullMaxProcesses) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=0 max=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetProcessList_InvalidHandle) {
  uint32_t max_processes = 16;
  amdsmi_proc_info_t list[16];
  memset(list, 0, sizeof(list));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list(kInvalidHandle, &max_processes, list);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetProcessList_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_process_list");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t max_processes = 16;
    amdsmi_proc_info_t list[16];
    memset(list, 0, sizeof(list));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_process_list(gpus()[i], &max_processes, list);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_process_list_by_pid ----------------
TEST_F(GpuIntegration, GetProcessListByPid_NullMaxProcesses) {
  std::vector<amdsmi_processor_handle> handles = gpus();
  amdsmi_proc_info_by_pid_t procs[16];
  memset(procs, 0, sizeof(procs));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list_by_pid", "max=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list_by_pid(
      handles.data(), static_cast<uint32_t>(handles.size()), procs, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetProcessListByPid_NullHandles) {
  amdsmi_proc_info_by_pid_t procs[16];
  memset(procs, 0, sizeof(procs));
  uint32_t max_processes = 16;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list_by_pid", "handles=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list_by_pid(nullptr, 1, procs, &max_processes);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetProcessListByPid_Valid) {
  std::vector<amdsmi_processor_handle> handles = gpus();
  amdsmi_proc_info_by_pid_t procs[16];
  memset(procs, 0, sizeof(procs));
  uint32_t max_processes = 16;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list_by_pid", "valid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list_by_pid(
      handles.data(), static_cast<uint32_t>(handles.size()), procs, &max_processes);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---------------- amdsmi_get_gpu_compute_process_info (no handle) ----------------
TEST_F(GpuIntegration, GetComputeProcessInfo_NullNumItems) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_info", "num_items=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_process_info(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetComputeProcessInfo_Valid) {
  uint32_t num_items = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_info", "count query", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_process_info(nullptr, &num_items);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_INVAL);
}

// ---------------- amdsmi_get_gpu_compute_process_info_by_pid (no handle) ----------------
TEST_F(GpuIntegration, GetComputeProcessInfoByPid_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_info_by_pid", "out=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_compute_process_info_by_pid(static_cast<uint32_t>(getpid()), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetComputeProcessInfoByPid_Valid) {
  amdsmi_process_info_t proc;
  memset(&proc, 0, sizeof(proc));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_info_by_pid", "self pid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_compute_process_info_by_pid(static_cast<uint32_t>(getpid()), &proc);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_NOT_FOUND);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NOT_FOUND);
}

// ---------------- amdsmi_get_gpu_compute_process_gpus (no handle) ----------------
TEST_F(GpuIntegration, GetComputeProcessGpus_NullNumDevices) {
  uint32_t dv_indices[16];
  memset(dv_indices, 0, sizeof(dv_indices));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_gpus", "num_devices=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_compute_process_gpus(static_cast<uint32_t>(getpid()), dv_indices, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetComputeProcessGpus_Valid) {
  uint32_t dv_indices[16];
  memset(dv_indices, 0, sizeof(dv_indices));
  uint32_t num_devices = 16;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_gpus", "self pid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_process_gpus(static_cast<uint32_t>(getpid()),
                                                            dv_indices, &num_devices);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                        AMDSMI_STATUS_NOT_FOUND, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NOT_FOUND,
                       AMDSMI_STATUS_INVAL);
}

// ---- amdsmi_set_gpu_process_isolation (invalid input only; valid-input cases are in functional/)
// ----
TEST_F(GpuIntegration, SetProcessIsolation_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_process_isolation", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_process_isolation(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
