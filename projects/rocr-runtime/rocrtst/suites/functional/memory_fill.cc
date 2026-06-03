/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/

/* Test Name: memory_fill
 *
 * Purpose: Verifies hsa_amd_memory_fill_byte APIs
 *
 * Test Description:
 * 1. Tests hsa_amd_memory_fill_byte with byte-level fills
 * 2. Tests byte fills with unaligned addresses and odd sizes
 * 3. Tests error handling (null pointers, zero count)
 *
 * Expected Results: Memory should be filled correctly with specified values.
 * Byte-level API should handle any alignment and size. Error cases should
 * return appropriate status codes.
 */

#include "suites/functional/memory_fill.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

static const size_t kMemoryAllocSize = 4096;
static const uint32_t kFillValueA = 0xDEADBEEF;
static const uint32_t kFillValueB = 0xCAFEBABE;
static const uint8_t kFillByteA = 0xAB;
static const uint8_t kFillByteB = 0xCD;

MemoryFillTest::MemoryFillTest(void) : TestBase() {
  set_num_iteration(10);
  set_title("Memory Fill Test");
  set_description("Tests hsa_amd_memory_fill_byte");
}

MemoryFillTest::MemoryFillTest(MemeoryFill type) : TestBase(), testtype_(type) {
  set_num_iteration(10);
  set_title("Memory Fill Test");
  set_description("Tests hsa_amd_memory_fill_byte");
}

MemoryFillTest::~MemoryFillTest(void) {}

void MemoryFillTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
}

void MemoryFillTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryFillTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryFillTest::DisplayResults(void) const {
  return;
}

void MemoryFillTest::Close() {
  // Cleanup
  if(resources_free){
    if(signal.handle != 0){
      hsa_signal_destroy(signal);
      signal.handle = 0;
    }
    if (sysBuf) {
      hsa_amd_memory_pool_free(sysBuf);
    }
    if (staging) {
      hsa_amd_memory_pool_free(staging);
    }
    if (g_gpuBuf) {
      hsa_amd_memory_pool_free(g_gpuBuf);
    }
    if (gpuBuf) {
      hsa_amd_memory_pool_free(gpuBuf);
    }
    resources_free = false;
  }
  TestBase::Close();
}

void MemoryFillTest::TestMemoryFillByte(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent) {
  hsa_status_t err;
  resources_free = true;
  const size_t kNumBytes = kMemoryAllocSize;
  const size_t kOffsetBytes = 256;
  const size_t kRegionBytes = 1024;

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

  err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                     reinterpret_cast<void**>(&sysBuf));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  memset(sysBuf, 0, kMemoryAllocSize);

  err = hsa_amd_memory_fill_byte(sysBuf, kFillByteA, kNumBytes);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(sysBuf[i], kFillByteA);
  }

  if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&gpuBuf));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpuBuf);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    memset(gpuBuf, 0, kMemoryAllocSize);

    // Fill entire buffer
    err = hsa_amd_memory_fill_byte(gpuBuf, kFillByteA, kNumBytes);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Fill subregion
    err = hsa_amd_memory_fill_byte(gpuBuf + kOffsetBytes, kFillByteB, kRegionBytes);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    for (size_t i = 0; i < kNumBytes; ++i) {
      if (i >= kOffsetBytes && i < (kOffsetBytes + kRegionBytes)) {
        ASSERT_EQ(gpuBuf[i], kFillByteB);
      } else {
        ASSERT_EQ(gpuBuf[i], kFillByteA);
      }
    }
  } else {
    // Test 3: Fill GPU memory (not accessible - use staging)
    err = hsa_signal_create(1, 0, NULL, &signal);
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

    // Fill GPU memory with bytes
    err = hsa_amd_memory_fill_byte(g_gpuBuf, kFillByteA, kNumBytes);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_fill_byte(g_gpuBuf + kOffsetBytes, kFillByteB, kRegionBytes);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Copy to staging to verify
    hsa_signal_store_relaxed(signal, 1);
    err = hsa_amd_memory_async_copy(staging, gpuAgent, g_gpuBuf, gpuAgent,
                                    kMemoryAllocSize, 0, NULL, signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE)) { }

    for (size_t i = 0; i < kNumBytes; ++i) {
      if (i >= kOffsetBytes && i < (kOffsetBytes + kRegionBytes)) {
        ASSERT_EQ(staging[i], kFillByteB);
      } else {
        ASSERT_EQ(staging[i], kFillByteA);
      }
    }
  }
}

void MemoryFillTest::TestMemoryFillByteUnaligned(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent) {
  hsa_status_t err;
  resources_free = true;

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

  // ============= Part 1: Test on System Memory =============
  err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                     reinterpret_cast<void**>(&sysBuf));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  memset(sysBuf, 0, kMemoryAllocSize);

  // Test 1: Unaligned address (offset by 1 byte)
  err = hsa_amd_memory_fill_byte(sysBuf + 1, kFillByteA, 100);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  ASSERT_EQ(sysBuf[0], 0);  // Before fill region
  for (int i = 1; i <= 100; ++i) {
    ASSERT_EQ(sysBuf[i], kFillByteA);
  }
  ASSERT_EQ(sysBuf[101], 0);  // After fill region

  // Test 2: Odd size
  memset(sysBuf, 0, kMemoryAllocSize);
  err = hsa_amd_memory_fill_byte(sysBuf, kFillByteB, 999);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (int i = 0; i < 999; ++i) {
    ASSERT_EQ(sysBuf[i], kFillByteB);
  }
  ASSERT_EQ(sysBuf[999], 0);  // After fill region

  // Test 3: Single byte fill
  memset(sysBuf, 0, kMemoryAllocSize);
  err = hsa_amd_memory_fill_byte(sysBuf + 500, kFillByteA, 1);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  ASSERT_EQ(sysBuf[499], 0);
  ASSERT_EQ(sysBuf[500], kFillByteA);
  ASSERT_EQ(sysBuf[501], 0);

  // ============= Part 2: Test on GPU Memory =============
  if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    // GPU memory is CPU-accessible - can directly test
    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&gpuBuf));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpuBuf);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Test 1: Unaligned address on GPU memory
    memset(gpuBuf, 0, kMemoryAllocSize);
    err = hsa_amd_memory_fill_byte(gpuBuf + 1, kFillByteA, 100);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    ASSERT_EQ(gpuBuf[0], 0);
    for (int i = 1; i <= 100; ++i) {
      ASSERT_EQ(gpuBuf[i], kFillByteA);
    }
    ASSERT_EQ(gpuBuf[101], 0);

    // Test 2: Odd size on GPU memory
    memset(gpuBuf, 0, kMemoryAllocSize);
    err = hsa_amd_memory_fill_byte(gpuBuf, kFillByteB, 999);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    for (int i = 0; i < 999; ++i) {
      ASSERT_EQ(gpuBuf[i], kFillByteB);
    }
    ASSERT_EQ(gpuBuf[999], 0);

    // Test 3: Single byte fill on GPU memory
    memset(gpuBuf, 0, kMemoryAllocSize);
    err = hsa_amd_memory_fill_byte(gpuBuf + 500, kFillByteA, 1);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    ASSERT_EQ(gpuBuf[499], 0);
    ASSERT_EQ(gpuBuf[500], kFillByteA);
    ASSERT_EQ(gpuBuf[501], 0);

  } else {
    // GPU memory is not CPU-accessible - use staging buffer + async copy
    err = hsa_signal_create(1, 0, NULL, &signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&g_gpuBuf));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&staging));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, staging);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Test 1: Unaligned address on GPU memory
    memset(staging, 0, kMemoryAllocSize);
    err = hsa_amd_memory_fill_byte(g_gpuBuf + 1, kFillByteA, 100);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    hsa_signal_store_relaxed(signal, 1);
    err = hsa_amd_memory_async_copy(staging, gpuAgent, g_gpuBuf, gpuAgent,
                                    kMemoryAllocSize, 0, NULL, signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE)) { }

    ASSERT_EQ(staging[0], 0);
    for (int i = 1; i <= 100; ++i) {
      ASSERT_EQ(staging[i], kFillByteA);
    }
    ASSERT_EQ(staging[101], 0);

    // Test 2: Odd size on GPU memory
    memset(staging, 0, kMemoryAllocSize);
    err = hsa_amd_memory_fill_byte(g_gpuBuf, kFillByteB, 999);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    hsa_signal_store_relaxed(signal, 1);
    err = hsa_amd_memory_async_copy(staging, gpuAgent, g_gpuBuf, gpuAgent,
                                    kMemoryAllocSize, 0, NULL, signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE)) { }

    for (int i = 0; i < 999; ++i) {
      ASSERT_EQ(staging[i], kFillByteB);
    }
    ASSERT_EQ(staging[999], 0);

    // Test 3: Single byte fill on GPU memory
    memset(staging, 0, kMemoryAllocSize);
    err = hsa_amd_memory_fill_byte(g_gpuBuf + 500, kFillByteA, 1);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    hsa_signal_store_relaxed(signal, 1);
    err = hsa_amd_memory_async_copy(staging, gpuAgent, g_gpuBuf, gpuAgent,
                                    kMemoryAllocSize, 0, NULL, signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE)) { }

    ASSERT_EQ(staging[499], 0);
    ASSERT_EQ(staging[500], kFillByteA);
    ASSERT_EQ(staging[501], 0);
  }
}

void MemoryFillTest::TestMemoryFillErrors() {
  hsa_status_t err;
   if (verbosity() >= VERBOSE_STANDARD) {
    std::cout << "Testing memory fill error conditions..." << std::endl;
  }
  // Test 1: NULL pointer should return error
  err = hsa_amd_memory_fill(NULL, kFillValueA, 100);
  ASSERT_EQ(err, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  err = hsa_amd_memory_fill_byte(NULL, kFillByteA, 100);
  ASSERT_EQ(err, HSA_STATUS_ERROR_INVALID_ARGUMENT);

  // Test 2: Zero count should succeed (no-op)
  uint32_t dummy;
  err = hsa_amd_memory_fill(&dummy, kFillValueA, 0);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_fill_byte(&dummy, kFillByteA, 0);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
}

void MemoryFillTest::MemoryFillByteTest(void){
  hsa_status_t err;
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  switch(testtype_){
    case ByteBasic:{
                    if (verbosity() >= VERBOSE_STANDARD) {
                      std::cout << "Testing hsa_amd_memory_fill_byte..." << std::endl;
                    }
                    for (unsigned int i = 0; i < gpus.size(); ++i) 
                      TestMemoryFillByte(cpus[0], gpus[i]);
                    break;
    }     
    case ByteUnAligned:{
                      if (verbosity() >= VERBOSE_STANDARD) {
                        std::cout << "Testing hsa_amd_memory_fill_byte with unaligned addresses..." << std::endl;
                      }
                      for (unsigned int i = 0; i < gpus.size(); ++i) 
                        TestMemoryFillByteUnaligned(cpus[0], gpus[i]);
                      break;
    }
    default:          std::cout<<"Invalid Type";
  }
}
