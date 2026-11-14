/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
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

// Long-dependency copy (ROCM-27963). SDMA arms its command watchdog from the
// KGD timeout, 100ms per count. A POLL_REGMEM that is still outstanding when
// that window expires can drop the copy's completion write, leaving the host
// waiting forever. The dependency here is held past that window, then
// released, and the copy must still retire.
static const std::chrono::milliseconds kLongDependencyHold(200);
static const uint64_t kLongDependencyCompletionTimeoutNs = 2000000000ull;
static const int kLongDependencyRepeats = 4;
static const size_t kLongLinearBytes = 1024;
static const size_t kLongRectWidth = 128;
static const size_t kLongRectHeight = 32;
static const size_t kLongRectSrcPitch = 128;
static const size_t kLongRectDstPitch = 256;

static bool LongDependencySdmaDisabled(void) {
  const char* env = getenv("HSA_ENABLE_SDMA");
  return env != nullptr && env[0] == '0';
}

// Blocking copy used to stage data over the ordinary copy path. The signal is
// owned by the caller and reused across calls.
static hsa_status_t LongDependencyBlockingCopy(hsa_signal_t signal, void* dst,
                                               hsa_agent_t dst_agent,
                                               const void* src,
                                               hsa_agent_t src_agent,
                                               size_t size) {
  hsa_signal_store_relaxed(signal, 1);
  hsa_status_t err = hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent,
                                               size, 0, nullptr, signal);
  if (err != HSA_STATUS_SUCCESS) return err;

  if (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED) != 0) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

// Holds the gate for kLongDependencyHold, then releases it and waits a bounded
// time for the copy to retire. Returns false once a failure is recorded.
static bool LongDependencyRelease(hsa_signal_t gate, hsa_signal_t done,
                                  const char* what) {
  std::this_thread::sleep_for(kLongDependencyHold);

  if (hsa_signal_load_scacquire(done) != 1) {
    ADD_FAILURE() << what << " retired while its dependency was still set";
    hsa_signal_store_screlease(gate, 0);
    return false;
  }

  hsa_signal_store_screlease(gate, 0);

  const hsa_signal_value_t left = hsa_signal_wait_scacquire(
      done, HSA_SIGNAL_CONDITION_LT, 1, kLongDependencyCompletionTimeoutNs,
      HSA_WAIT_STATE_BLOCKED);
  if (left != 0) {
    ADD_FAILURE() << what << " did not complete after a dependency held for "
                  << kLongDependencyHold.count()
                  << " ms was released (completion signal is "
                  << hsa_signal_load_scacquire(done)
                  << "). An SDMA poll that outlasts the command timeout "
                     "drops the copy's completion write.";
    return false;
  }
  return true;
}

void MemoryFill::LongDependencyCopyTest(void) {
  std::cout << "  *** Long dependency copy (ROCM-27963) ***" << std::endl;

  if (LongDependencySdmaDisabled()) {
    std::cout << "  Skipping: the command timeout is on the SDMA poll path "
                 "and HSA_ENABLE_SDMA=0" << std::endl;
    return;
  }

  hsa_signal_t xfer;
  ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &xfer));
  MAKE_SCOPE_GUARD([&]() { hsa_signal_destroy(xfer); });

  uint32_t* host_buf = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(
      cpu_pool(), kLongLinearBytes, 0, reinterpret_cast<void**>(&host_buf)));
  MAKE_SCOPE_GUARD([&]() { hsa_amd_memory_pool_free(host_buf); });
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, gpu_device1(), nullptr,
                                             host_buf));

  void* dev_buf = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(device_pool(), kLongLinearBytes,
                                              0, &dev_buf));
  MAKE_SCOPE_GUARD([&]() { hsa_amd_memory_pool_free(dev_buf); });

  const size_t kLinearDwords = kLongLinearBytes / sizeof(uint32_t);
  std::vector<uint32_t> pattern(kLinearDwords);
  for (size_t i = 0; i < kLinearDwords; ++i) {
    pattern[i] = 0x5EED0000u + static_cast<uint32_t>(i);
  }

  hsa_signal_t gate;
  ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &gate));
  MAKE_SCOPE_GUARD([&]() { hsa_signal_destroy(gate); });

  hsa_signal_t done;
  ASSERT_SUCCESS(hsa_signal_create(1, 0, nullptr, &done));
  MAKE_SCOPE_GUARD([&]() { hsa_signal_destroy(done); });

  for (int iter = 0; iter < kLongDependencyRepeats; ++iter) {
    for (size_t i = 0; i < kLinearDwords; ++i) {
      host_buf[i] = 0xDEADBEEFu;
    }
    ASSERT_SUCCESS(LongDependencyBlockingCopy(xfer, dev_buf, *gpu_device1(),
                                              host_buf, *cpu_device(),
                                              kLongLinearBytes));
    memcpy(host_buf, pattern.data(), kLongLinearBytes);

    hsa_signal_store_screlease(gate, 1);
    hsa_signal_store_screlease(done, 1);
    ASSERT_SUCCESS(hsa_amd_memory_async_copy(dev_buf, *gpu_device1(), host_buf,
                                             *cpu_device(), kLongLinearBytes, 1,
                                             &gate, done));
    if (!LongDependencyRelease(gate, done, "linear copy")) {
      return;
    }

    memset(host_buf, 0, kLongLinearBytes);
    ASSERT_SUCCESS(LongDependencyBlockingCopy(xfer, host_buf, *cpu_device(),
                                              dev_buf, *gpu_device1(),
                                              kLongLinearBytes));
    EXPECT_EQ(0, memcmp(host_buf, pattern.data(), kLongLinearBytes))
        << "linear copy iteration " << iter
        << " destination does not hold the source contents";
  }

  // Same dependency on the pitched host-to-device rectangle that still hung
  // after the firmware update: 128x32, source pitch 128, destination pitch 256.
  const size_t src_bytes = kLongRectSrcPitch * kLongRectHeight;
  const size_t dst_bytes = kLongRectDstPitch * kLongRectHeight;

  uint8_t* rect_src = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(
      cpu_pool(), src_bytes, 0, reinterpret_cast<void**>(&rect_src)));
  MAKE_SCOPE_GUARD([&]() { hsa_amd_memory_pool_free(rect_src); });

  uint8_t* rect_host_dst = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(
      cpu_pool(), dst_bytes, 0, reinterpret_cast<void**>(&rect_host_dst)));
  MAKE_SCOPE_GUARD([&]() { hsa_amd_memory_pool_free(rect_host_dst); });

  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, gpu_device1(), nullptr,
                                             rect_src));
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, gpu_device1(), nullptr,
                                             rect_host_dst));

  void* rect_dst = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(device_pool(), dst_bytes, 0,
                                              &rect_dst));
  MAKE_SCOPE_GUARD([&]() { hsa_amd_memory_pool_free(rect_dst); });

  for (size_t y = 0; y < kLongRectHeight; ++y) {
    for (size_t x = 0; x < kLongRectWidth; ++x) {
      rect_src[y * kLongRectSrcPitch + x] = static_cast<uint8_t>(x ^ y);
    }
  }

  const hsa_pitched_ptr_t src_pitched = {rect_src, kLongRectSrcPitch, 0};
  const hsa_pitched_ptr_t dst_pitched = {rect_dst, kLongRectDstPitch, 0};
  const hsa_dim3_t offset = {0, 0, 0};
  const hsa_dim3_t range = {static_cast<uint32_t>(kLongRectWidth),
                            static_cast<uint32_t>(kLongRectHeight), 1};
  const std::vector<uint8_t> poison_tail(kLongRectDstPitch - kLongRectWidth,
                                         0xA5);

  for (int iter = 0; iter < kLongDependencyRepeats; ++iter) {
    memset(rect_host_dst, 0xA5, dst_bytes);
    ASSERT_SUCCESS(LongDependencyBlockingCopy(xfer, rect_dst, *gpu_device1(),
                                              rect_host_dst, *cpu_device(),
                                              dst_bytes));

    hsa_signal_store_screlease(gate, 1);
    hsa_signal_store_screlease(done, 1);
    ASSERT_SUCCESS(hsa_amd_memory_async_copy_rect(
        &dst_pitched, &offset, &src_pitched, &offset, &range, *gpu_device1(),
        hsaHostToDevice, 1, &gate, done));
    if (!LongDependencyRelease(gate, done, "pitched 2D copy")) {
      return;
    }

    memset(rect_host_dst, 0, dst_bytes);
    ASSERT_SUCCESS(LongDependencyBlockingCopy(xfer, rect_host_dst, *cpu_device(),
                                              rect_dst, *gpu_device1(),
                                              dst_bytes));

    for (size_t y = 0; y < kLongRectHeight; ++y) {
      EXPECT_EQ(0, memcmp(rect_host_dst + y * kLongRectDstPitch,
                          rect_src + y * kLongRectSrcPitch, kLongRectWidth))
          << "pitched 2D copy iteration " << iter << " row " << y
          << " does not hold the source contents";
      EXPECT_EQ(0, memcmp(rect_host_dst + y * kLongRectDstPitch + kLongRectWidth,
                          poison_tail.data(), poison_tail.size()))
          << "pitched 2D copy iteration " << iter << " row " << y
          << " wrote past the row width";
    }
  }

  if (verbosity() > 0) {
    std::cout << "  Repeated a linear copy and a " << kLongRectWidth << "x"
              << kLongRectHeight << " pitched copy " << kLongDependencyRepeats
              << " times behind a " << kLongDependencyHold.count()
              << " ms dependency" << std::endl;
  }
}
