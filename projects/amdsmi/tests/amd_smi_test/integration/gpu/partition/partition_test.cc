// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_gpu_compute_partition (char) ----------------
TEST(GpuIntegration, GetComputePartition_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition(any_gpu(), nullptr, 64);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetComputePartition_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  char buf[64];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition(kInvalidHandle, buf, sizeof(buf));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetComputePartition_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetMemoryPartition_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition(any_gpu(), nullptr, 64);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetMemoryPartition_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  char buf[64];
  memset(buf, 0, sizeof(buf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition(kInvalidHandle, buf, sizeof(buf));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetMemoryPartition_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetComputePartitionMemAllocMode_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetComputePartitionMemAllocMode_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_compute_partition_mem_alloc_mode_t mode;
  memset(&mode, 0, sizeof(mode));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_compute_partition_mem_alloc_mode(kInvalidHandle, &mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetComputePartitionMemAllocMode_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetAcceleratorPartitionMemAllocMode_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetAcceleratorPartitionMemAllocMode_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_accelerator_partition_mem_alloc_mode_t mode;
  memset(&mode, 0, sizeof(mode));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(kInvalidHandle, &mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetAcceleratorPartitionMemAllocMode_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetMemoryPartitionConfig_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetMemoryPartitionConfig_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_memory_partition_config_t config;
  memset(&config, 0, sizeof(config));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_partition_config", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_memory_partition_config(kInvalidHandle, &config);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetMemoryPartitionConfig_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetAcceleratorPartitionProfileConfig_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config", "gpu=0 out=nullptr",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetAcceleratorPartitionProfileConfig_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  amdsmi_accelerator_partition_profile_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile_config", "handle=invalid",
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile_config(kInvalidHandle, &cfg);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, GetAcceleratorPartitionProfileConfig_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetAcceleratorPartitionProfile_NullOutput) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_profile", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_accelerator_partition_profile(any_gpu(), nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST(GpuIntegration, GetAcceleratorPartitionProfile_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, GetAcceleratorPartitionProfile_AllGpus) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, SetComputePartitionMemAllocMode_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition_mem_alloc_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_compute_partition_mem_alloc_mode(
      kInvalidHandle, AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, SetAcceleratorPartitionMemAllocMode_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
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
TEST(GpuIntegration, SetComputePartition_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_compute_partition(kInvalidHandle, AMDSMI_COMPUTE_PARTITION_SPX);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, SetMemoryPartition_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_memory_partition(kInvalidHandle, AMDSMI_MEMORY_PARTITION_NPS1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, SetMemoryPartitionMode_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_memory_partition_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_set_gpu_memory_partition_mode(kInvalidHandle, AMDSMI_MEMORY_PARTITION_NPS1);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST(GpuIntegration, SetAcceleratorPartitionProfile_InvalidHandle) {
  AMDSMI_API_TEST_SCOPE();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_profile", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_accelerator_partition_profile(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
