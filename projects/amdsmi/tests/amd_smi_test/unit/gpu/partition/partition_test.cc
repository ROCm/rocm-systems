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

// ---------------- amdsmi_get_gpu_compute_partition (char) ----------------
TEST_F(GpuUnit, GetComputePartition_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition(gpus()[0], nullptr, 64);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetComputePartition_InvalidHandle) {
  char buf[64];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition(kInvalidHandle, buf, sizeof(buf));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetComputePartition_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_compute_partition");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char buf[64];
    memset(buf, 0, sizeof(buf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_compute_partition(gpus()[i], buf, sizeof(buf));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_memory_partition (char) ----------------
TEST_F(GpuUnit, GetMemoryPartition_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition(gpus()[0], nullptr, 64);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetMemoryPartition_InvalidHandle) {
  char buf[64];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition(kInvalidHandle, buf, sizeof(buf));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMemoryPartition_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_partition");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char buf[64];
    memset(buf, 0, sizeof(buf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_memory_partition(gpus()[i], buf, sizeof(buf));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_compute_partition_mem_alloc_mode ----------------
TEST_F(GpuUnit, GetComputePartitionMemAllocMode_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetComputePartitionMemAllocMode_InvalidHandle) {
  amdsmi_compute_partition_mem_alloc_mode_t mode;
  memset(&mode, 0, sizeof(mode));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(kInvalidHandle, &mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetComputePartitionMemAllocMode_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_compute_partition_mem_alloc_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_compute_partition_mem_alloc_mode_t mode;
    memset(&mode, 0, sizeof(mode));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(gpus()[i], &mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_accelerator_partition_mem_alloc_mode ----------------
TEST_F(GpuUnit, GetAcceleratorPartitionMemAllocMode_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetAcceleratorPartitionMemAllocMode_InvalidHandle) {
  amdsmi_accelerator_partition_mem_alloc_mode_t mode;
  memset(&mode, 0, sizeof(mode));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(kInvalidHandle, &mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetAcceleratorPartitionMemAllocMode_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col(
      "amdsmi_get_gpu_accelerator_partition_mem_alloc_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_accelerator_partition_mem_alloc_mode_t mode;
    memset(&mode, 0, sizeof(mode));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(gpus()[i], &mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_memory_partition_config ----------------
TEST_F(GpuUnit, GetMemoryPartitionConfig_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetMemoryPartitionConfig_InvalidHandle) {
  amdsmi_memory_partition_config_t config;
  memset(&config, 0, sizeof(config));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(kInvalidHandle, &config);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetMemoryPartitionConfig_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_partition_config");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_memory_partition_config_t config;
    memset(&config, 0, sizeof(config));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(gpus()[i], &config);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_accelerator_partition_profile_config ----------------
TEST_F(GpuUnit, GetAcceleratorPartitionProfileConfig_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuUnit, GetAcceleratorPartitionProfileConfig_InvalidHandle) {
  amdsmi_accelerator_partition_profile_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(kInvalidHandle, &cfg);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetAcceleratorPartitionProfileConfig_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col(
      "amdsmi_get_gpu_accelerator_partition_profile_config");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_accelerator_partition_profile_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(gpus()[i], &cfg);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_gpu_accelerator_partition_profile ----------------
TEST_F(GpuUnit, GetAcceleratorPartitionProfile_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile(gpus()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
TEST_F(GpuUnit, GetAcceleratorPartitionProfile_InvalidHandle) {
  amdsmi_accelerator_partition_profile_t profile;
  memset(&profile, 0, sizeof(profile));
  uint32_t partition_id = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_accelerator_partition_profile(kInvalidHandle, &profile, &partition_id);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, GetAcceleratorPartitionProfile_AllGpus) {
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_get_gpu_accelerator_partition_profile");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_accelerator_partition_profile_t profile;
    memset(&profile, 0, sizeof(profile));
    uint32_t partition_id = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err =
        amdsmi_get_gpu_accelerator_partition_profile(gpus()[i], &profile, &partition_id);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::unittest::AmdsmiStatusIsExpected(
                          err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- SET: mem_alloc_mode setters (read current, restore) ----------------
TEST_F(GpuUnit, SetComputePartitionMemAllocMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition_mem_alloc_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_compute_partition_mem_alloc_mode(
      kInvalidHandle, AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetComputePartitionMemAllocMode_AllGpus) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  amdsmi::unittest::StatusCollector amdsmi_col("amdsmi_set_gpu_compute_partition_mem_alloc_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_compute_partition_mem_alloc_mode_t mode = AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_INVALID;
    if (amdsmi_get_gpu_compute_partition_mem_alloc_mode(gpus()[i], &mode) != AMDSMI_STATUS_SUCCESS)
      continue;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_compute_partition_mem_alloc_mode(gpus()[i], mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL);
    amdsmi_col.Record(
        "gpu=" + std::to_string(i), err,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL));
  }
  amdsmi_col.ExpectNoFailures();
}
TEST_F(GpuUnit, SetAcceleratorPartitionMemAllocMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_mem_alloc_mode", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(
      kInvalidHandle, AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetAcceleratorPartitionMemAllocMode_AllGpus) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  amdsmi::unittest::StatusCollector amdsmi_col(
      "amdsmi_set_gpu_accelerator_partition_mem_alloc_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_accelerator_partition_mem_alloc_mode_t mode =
        AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_INVALID;
    if (amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(gpus()[i], &mode) !=
        AMDSMI_STATUS_SUCCESS)
      continue;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(gpus()[i], mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL);
    amdsmi_col.Record(
        "gpu=" + std::to_string(i), err,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_INVAL));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- SET: repartition ops exercised with non-destructive sentinel inputs
// ---------------- A successful repartition would reconfigure a live device shared with other
// processes, so the valid-handle path is intentionally driven with a sentinel/
// invalid selector that the driver rejects; only the call contract is validated.
TEST_F(GpuUnit, SetComputePartition_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_compute_partition(kInvalidHandle, AMDSMI_COMPUTE_PARTITION_SPX);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetComputePartition_AllGpusSentinel) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition", "gpu=" + std::to_string(i) + " sentinel",
                       kVerbose);
    amdsmi_status_t err =
        amdsmi_set_gpu_compute_partition(gpus()[i], AMDSMI_COMPUTE_PARTITION_INVALID);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                         AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM);
  }
}
TEST_F(GpuUnit, SetMemoryPartition_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_memory_partition(kInvalidHandle, AMDSMI_MEMORY_PARTITION_NPS1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetMemoryPartition_AllGpusSentinel) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition", "gpu=" + std::to_string(i) + " sentinel",
                       kVerbose);
    amdsmi_status_t err =
        amdsmi_set_gpu_memory_partition(gpus()[i], AMDSMI_MEMORY_PARTITION_UNKNOWN);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                         AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM);
  }
}
TEST_F(GpuUnit, SetMemoryPartitionMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_memory_partition_mode(kInvalidHandle, AMDSMI_MEMORY_PARTITION_NPS1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetMemoryPartitionMode_AllGpusSentinel) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode",
                       "gpu=" + std::to_string(i) + " sentinel", kVerbose);
    amdsmi_status_t err =
        amdsmi_set_gpu_memory_partition_mode(gpus()[i], AMDSMI_MEMORY_PARTITION_UNKNOWN);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                         AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM);
  }
}
TEST_F(GpuUnit, SetAcceleratorPartitionProfile_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_profile", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_accelerator_partition_profile(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuUnit, SetAcceleratorPartitionProfile_AllGpusSentinel) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_profile",
                       "gpu=" + std::to_string(i) + " sentinel", kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_accelerator_partition_profile(
        gpus()[i], 0xFFFFFFFF);  // out-of-range profile ID — no named invalid value
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                         AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM);
  }
}
