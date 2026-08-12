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

namespace {

amdsmi_memory_partition_type_t MemStrToEnum(const std::string& s) {
  if (s == "NPS1") return AMDSMI_MEMORY_PARTITION_NPS1;
  if (s == "NPS2") return AMDSMI_MEMORY_PARTITION_NPS2;
  if (s == "NPS4") return AMDSMI_MEMORY_PARTITION_NPS4;
  if (s == "NPS8") return AMDSMI_MEMORY_PARTITION_NPS8;
  return AMDSMI_MEMORY_PARTITION_UNKNOWN;
}

amdsmi_compute_partition_type_t CompStrToEnum(const std::string& s) {
  if (s == "SPX") return AMDSMI_COMPUTE_PARTITION_SPX;
  if (s == "DPX") return AMDSMI_COMPUTE_PARTITION_DPX;
  if (s == "TPX") return AMDSMI_COMPUTE_PARTITION_TPX;
  if (s == "QPX") return AMDSMI_COMPUTE_PARTITION_QPX;
  if (s == "CPX") return AMDSMI_COMPUTE_PARTITION_CPX;
  return AMDSMI_COMPUTE_PARTITION_INVALID;
}

}  // namespace

// amdsmi_get_gpu_memory_partition / amdsmi_set_gpu_memory_partition.
TEST_F(GpuFunctionalReadWrite, SetMemoryPartition_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_memory_partition(kInvalidHandle, AMDSMI_MEMORY_PARTITION_NPS1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, MemoryPartition_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_gpu_memory_partition");
  for (size_t i = 0; i < gpus().size(); ++i) {
    char buf[64];
    memset(buf, 0, sizeof(buf));
    if (amdsmi_get_gpu_memory_partition(gpus()[i], buf, sizeof(buf)) != AMDSMI_STATUS_SUCCESS)
      continue;
    std::string initial(buf);
    amdsmi_memory_partition_type_t initial_enum = MemStrToEnum(initial);
    if (initial_enum == AMDSMI_MEMORY_PARTITION_UNKNOWN) continue;

    amdsmi_memory_partition_type_t target_enum = (initial_enum == AMDSMI_MEMORY_PARTITION_NPS1)
                                                     ? AMDSMI_MEMORY_PARTITION_NPS2
                                                     : AMDSMI_MEMORY_PARTITION_NPS1;
    std::string target = (target_enum == AMDSMI_MEMORY_PARTITION_NPS1) ? "NPS1" : "NPS2";

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition",
                       "gpu=" + std::to_string(i) + " set=" + target, kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_memory_partition(gpus()[i], target_enum);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      char readback[64];
      memset(readback, 0, sizeof(readback));
      if (amdsmi_get_gpu_memory_partition(gpus()[i], readback, sizeof(readback)) ==
          AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(std::string(readback), target) << "gpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_gpu_memory_partition(gpus()[i], initial_enum);
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition",
                         "gpu=" + std::to_string(i) + " restore=" + initial, kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS)
          << "gpu=" << i << " failed to restore memory partition";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_memory_partition(gpus()[i], readback, sizeof(readback)) ==
              AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(std::string(readback), initial) << "gpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// amdsmi_get_gpu_compute_partition / amdsmi_set_gpu_compute_partition.
TEST_F(GpuFunctionalReadWrite, SetComputePartition_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_compute_partition(kInvalidHandle, AMDSMI_COMPUTE_PARTITION_SPX);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, ComputePartition_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_gpu_compute_partition");
  for (size_t i = 0; i < gpus().size(); ++i) {
    char buf[64];
    memset(buf, 0, sizeof(buf));
    if (amdsmi_get_gpu_compute_partition(gpus()[i], buf, sizeof(buf)) != AMDSMI_STATUS_SUCCESS)
      continue;
    std::string initial(buf);
    amdsmi_compute_partition_type_t initial_enum = CompStrToEnum(initial);
    if (initial_enum == AMDSMI_COMPUTE_PARTITION_INVALID) continue;

    amdsmi_compute_partition_type_t target_enum = (initial_enum == AMDSMI_COMPUTE_PARTITION_SPX)
                                                      ? AMDSMI_COMPUTE_PARTITION_CPX
                                                      : AMDSMI_COMPUTE_PARTITION_SPX;
    std::string target = (target_enum == AMDSMI_COMPUTE_PARTITION_SPX) ? "SPX" : "CPX";

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition",
                       "gpu=" + std::to_string(i) + " set=" + target, kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_compute_partition(gpus()[i], target_enum);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      char readback[64];
      memset(readback, 0, sizeof(readback));
      if (amdsmi_get_gpu_compute_partition(gpus()[i], readback, sizeof(readback)) ==
          AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(std::string(readback), target) << "gpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_gpu_compute_partition(gpus()[i], initial_enum);
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition",
                         "gpu=" + std::to_string(i) + " restore=" + initial, kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS)
          << "gpu=" << i << " failed to restore compute partition";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_compute_partition(gpus()[i], readback, sizeof(readback)) ==
              AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(std::string(readback), initial) << "gpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}
