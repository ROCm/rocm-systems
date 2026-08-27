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

// ---------------- amdsmi_get_gpu_compute_partition (char) ----------------
TEST_F(GpuIntegration, GetComputePartition_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition(any_gpu(), nullptr, 64);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetComputePartition_InvalidHandle) {
  char buf[64];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition(kInvalidHandle, buf, sizeof(buf));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetComputePartition_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_compute_partition");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char buf[64];
    memset(buf, 0, sizeof(buf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_compute_partition(gpus()[i], buf, sizeof(buf));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_memory_partition (char) ----------------
TEST_F(GpuIntegration, GetMemoryPartition_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition(any_gpu(), nullptr, 64);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMemoryPartition_InvalidHandle) {
  char buf[64];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition(kInvalidHandle, buf, sizeof(buf));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMemoryPartition_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_partition");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char buf[64];
    memset(buf, 0, sizeof(buf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_memory_partition(gpus()[i], buf, sizeof(buf));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_compute_partition_mem_alloc_mode ----------------
TEST_F(GpuIntegration, GetComputePartitionMemAllocMode_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetComputePartitionMemAllocMode_InvalidHandle) {
  amdsmi_compute_partition_mem_alloc_mode_t mode;
  memset(&mode, 0, sizeof(mode));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(kInvalidHandle, &mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetComputePartitionMemAllocMode_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_compute_partition_mem_alloc_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_compute_partition_mem_alloc_mode_t mode;
    memset(&mode, 0, sizeof(mode));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(gpus()[i], &mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_accelerator_partition_mem_alloc_mode ----------------
TEST_F(GpuIntegration, GetAcceleratorPartitionMemAllocMode_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetAcceleratorPartitionMemAllocMode_InvalidHandle) {
  amdsmi_accelerator_partition_mem_alloc_mode_t mode;
  memset(&mode, 0, sizeof(mode));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(kInvalidHandle, &mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetAcceleratorPartitionMemAllocMode_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_accelerator_partition_mem_alloc_mode_t mode;
    memset(&mode, 0, sizeof(mode));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(gpus()[i], &mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_memory_partition_config ----------------
TEST_F(GpuIntegration, GetMemoryPartitionConfig_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetMemoryPartitionConfig_InvalidHandle) {
  amdsmi_memory_partition_config_t config;
  memset(&config, 0, sizeof(config));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(kInvalidHandle, &config);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetMemoryPartitionConfig_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_memory_partition_config");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_memory_partition_config_t config;
    memset(&config, 0, sizeof(config));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(gpus()[i], &config);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_accelerator_partition_profile_config ----------------
TEST_F(GpuIntegration, GetAcceleratorPartitionProfileConfig_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetAcceleratorPartitionProfileConfig_InvalidHandle) {
  amdsmi_accelerator_partition_profile_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(kInvalidHandle, &cfg);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetAcceleratorPartitionProfileConfig_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_accelerator_partition_profile_config");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_accelerator_partition_profile_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config",
                       "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(gpus()[i], &cfg);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_accelerator_partition_profile ----------------
TEST_F(GpuIntegration, GetAcceleratorPartitionProfile_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetAcceleratorPartitionProfile_InvalidHandle) {
  amdsmi_accelerator_partition_profile_t profile;
  memset(&profile, 0, sizeof(profile));
  uint32_t partition_id = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_gpu_accelerator_partition_profile(kInvalidHandle, &profile, &partition_id);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetAcceleratorPartitionProfile_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_accelerator_partition_profile");
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
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- mem_alloc_mode setters (invalid input only; valid-input cases are in functional/) ----
TEST_F(GpuIntegration, SetComputePartitionMemAllocMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition_mem_alloc_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_compute_partition_mem_alloc_mode(
      kInvalidHandle, AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, SetAcceleratorPartitionMemAllocMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_mem_alloc_mode", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(
      kInvalidHandle, AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
// ---- repartition ops (invalid input only; valid-input cases are in functional/) ----
// ---------------- A successful repartition would reconfigure a live device shared with other
// processes, so the valid-handle path is intentionally driven with a sentinel/
// invalid selector that the driver rejects; only the call contract is validated.
TEST_F(GpuIntegration, SetComputePartition_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_compute_partition(kInvalidHandle, AMDSMI_COMPUTE_PARTITION_SPX);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, SetMemoryPartition_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_memory_partition(kInvalidHandle, AMDSMI_MEMORY_PARTITION_NPS1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, SetMemoryPartitionMode_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_memory_partition_mode(kInvalidHandle, AMDSMI_MEMORY_PARTITION_NPS1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, SetAcceleratorPartitionProfile_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_profile", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_accelerator_partition_profile(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
