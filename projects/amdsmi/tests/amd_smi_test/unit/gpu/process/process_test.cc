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

#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>

#include "unit/unit_test_framework.h"

using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

// ---------------- amdsmi_get_gpu_process_isolation ----------------
TEST_F(GpuUnit, GetProcessIsolation_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_isolation", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_isolation(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetProcessIsolation_InvalidHandle) {
  uint32_t pisolate = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_isolation", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_isolation(kInvalidHandle, &pisolate);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetProcessIsolation_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_process_isolation");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t pisolate = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_isolation", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_process_isolation(gpus()[i], &pisolate);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_process_list ----------------
TEST_F(GpuUnit, GetProcessList_NullMaxProcesses) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=0 max=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list(gpus()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetProcessList_InvalidHandle) {
  uint32_t max_processes = 16;
  amdsmi_proc_info_t list[16];
  memset(list, 0, sizeof(list));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list(kInvalidHandle, &max_processes, list);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetProcessList_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_process_list");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t max_processes = 16;
    amdsmi_proc_info_t list[16];
    memset(list, 0, sizeof(list));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_process_list(gpus()[i], &max_processes, list);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_process_list_by_pid ----------------
TEST_F(GpuUnit, GetProcessListByPid_NullMaxProcesses) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  std::vector<amdsmi_processor_handle> handles = gpus();
  amdsmi_proc_info_by_pid_t procs[16];
  memset(procs, 0, sizeof(procs));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list_by_pid", "max=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list_by_pid(
      handles.data(), static_cast<uint32_t>(handles.size()), procs, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetProcessListByPid_NullHandles) {
  amdsmi_proc_info_by_pid_t procs[16];
  memset(procs, 0, sizeof(procs));
  uint32_t max_processes = 16;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list_by_pid", "handles=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_process_list_by_pid(nullptr, 1, procs, &max_processes);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetProcessListByPid_Valid) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
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
TEST_F(GpuUnit, GetComputeProcessInfo_NullNumItems) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_info", "num_items=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_process_info(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetComputeProcessInfo_Valid) {
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
TEST_F(GpuUnit, GetComputeProcessInfoByPid_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_info_by_pid", "out=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_compute_process_info_by_pid(static_cast<uint32_t>(getpid()), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetComputeProcessInfoByPid_Valid) {
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
TEST_F(GpuUnit, GetComputeProcessGpus_NullNumDevices) {
  uint32_t dv_indices[16];
  memset(dv_indices, 0, sizeof(dv_indices));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_process_gpus", "num_devices=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_compute_process_gpus(static_cast<uint32_t>(getpid()), dv_indices, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetComputeProcessGpus_Valid) {
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

// ---------------- amdsmi_set_gpu_process_isolation (SET, read/restore) ----------------
TEST_F(GpuUnit, SetProcessIsolation_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_process_isolation", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_process_isolation(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetProcessIsolation_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_process_isolation");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t pisolate = 0;
    if (amdsmi_get_gpu_process_isolation(gpus()[i], &pisolate) != AMDSMI_STATUS_SUCCESS) continue;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_process_isolation", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_process_isolation(gpus()[i], pisolate);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));
  }
  amdsmi_col.ExpectNoFailures();
}
