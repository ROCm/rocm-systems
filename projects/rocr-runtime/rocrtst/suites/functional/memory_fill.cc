/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "suites/functional/memory_fill.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const int kMemoryAllocSize = 4096;
static const uint32_t kFillValueA = 0xDEADBEEF;
static const uint32_t kFillValueB = 0xA5A5A5A5;

MemoryFill::MemoryFill(void) : TestBase() {
  set_num_iteration(10);
  set_title("RocR Memory Fill Test");
  set_description("This test will use hsa_amd_memory_fill on system and GPU memory.");
}

MemoryFill::~MemoryFill(void) {
}

void MemoryFill::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  return;
}

void MemoryFill::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryFill::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryFill::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryFill::Close(void) {
  TestBase::Close();
}

void MemoryFill::MemoryFillTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent) {
  hsa_status_t err;

  const int kNumDwords = kMemoryAllocSize / sizeof(uint32_t);
  const int kOffsetDwords = 64;
  const int kRegionDwords = 256;

  hsa_amd_memory_pool_t global_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent,
                                          rocrtst::GetGlobalMemoryPool,
                                          &global_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  hsa_amd_memory_pool_t gpu_pool;
  err = hsa_amd_agent_iterate_memory_pools(gpuAgent,
                                          rocrtst::GetGlobalMemoryPool,
                                          &gpu_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  hsa_amd_memory_pool_access_t access;
  err = hsa_amd_agent_memory_pool_get_info(cpuAgent, gpu_pool,
                                           HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                           &access);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  uint32_t* sysBuf = NULL;
  err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                     reinterpret_cast<void**>(&sysBuf));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  memset(sysBuf, 0, kMemoryAllocSize);

  err = hsa_amd_memory_fill(sysBuf, kFillValueA, kNumDwords);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (int i = 0; i < kNumDwords; ++i) {
    ASSERT_EQ(sysBuf[i], kFillValueA);
  }

  uint32_t* gpuBuf = NULL;
  uint32_t* g_gpuBuf = NULL;
  uint32_t* staging = NULL;
  hsa_signal_t copy_signal;

  if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&gpuBuf));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpuBuf);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    memset(gpuBuf, 0, kMemoryAllocSize);

    err = hsa_amd_memory_fill(gpuBuf, kFillValueA, kNumDwords);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_fill(gpuBuf + kOffsetDwords, kFillValueB, kRegionDwords);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    for (int i = 0; i < kNumDwords; ++i) {
      if (i >= kOffsetDwords && i < (kOffsetDwords + kRegionDwords)) {
        ASSERT_EQ(gpuBuf[i], kFillValueB);
      } else {
        ASSERT_EQ(gpuBuf[i], kFillValueA);
      }
    }
  } else {
    err = hsa_signal_create(1, 0, NULL, &copy_signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&g_gpuBuf));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&staging));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    memset(staging, 0, kMemoryAllocSize);

    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, staging);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_fill(g_gpuBuf, kFillValueA, kNumDwords);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_fill(g_gpuBuf + kOffsetDwords, kFillValueB, kRegionDwords);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    hsa_signal_store_relaxed(copy_signal, 1);
    err = hsa_amd_memory_async_copy(staging, gpuAgent, g_gpuBuf, gpuAgent,
                                    kMemoryAllocSize, 0, NULL, copy_signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    while (hsa_signal_wait_acquire(copy_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE)) { }

    for (int i = 0; i < kNumDwords; ++i) {
      if (i >= kOffsetDwords && i < (kOffsetDwords + kRegionDwords)) {
        ASSERT_EQ(staging[i], kFillValueB);
      } else {
        ASSERT_EQ(staging[i], kFillValueA);
      }
    }

    err = hsa_signal_destroy(copy_signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  if (sysBuf) {
    err = hsa_amd_memory_pool_free(sysBuf);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (staging) {
    err = hsa_amd_memory_pool_free(staging);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (g_gpuBuf) {
    err = hsa_amd_memory_pool_free(g_gpuBuf);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (gpuBuf) {
    err = hsa_amd_memory_pool_free(gpuBuf);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
}

void MemoryFill::MemoryFillTest(void) {
  hsa_status_t err;

  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    MemoryFillTest(cpus[0], gpus[i]);
  }
}
