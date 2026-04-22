/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "suites/performance/agent_preload.h"
#include "common/base_rocr_utils.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"

// Find a global memory pool
static hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);

  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  bool alloc_allowed = false;
  hsa_amd_memory_pool_get_info(pool,
                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);

  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_t* out_pool = static_cast<hsa_amd_memory_pool_t*>(data);
  *out_pool = pool;
  return HSA_STATUS_INFO_BREAK;
}

AgentPreloadTest::AgentPreloadTest() : TestBase() {
  set_num_iteration(10);
  set_title("Agent Preload Latency Test");
  set_description("This test measures the latency reduction achieved by using "
                  "hsa_amd_agent_preload() before calling "
                  "hsa_amd_profiling_async_copy_enable(). The preload API warms"
                  "up the clock counter cache to avoid first-call latency.");

  latency_without_preload_us_ = 0.0;
  latency_with_preload_us_ = 0.0;
  latency_improvement_us_ = 0.0;
  blit_latency_without_preload_us_ = 0.0;
  blit_latency_with_preload_us_ = 0.0;
  blit_latency_improvement_us_ = 0.0;
}

AgentPreloadTest::~AgentPreloadTest() {
}

void AgentPreloadTest::SetUp() {
  hsa_status_t err;
  TestBase::SetUp();

  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpu_agents_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpu_agents_);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  if (gpu_agents_.empty()) {
    std::cout << "No GPU agents found. Test will be skipped." << std::endl;
  }
}

void AgentPreloadTest::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();

  if (gpu_agents_.empty()) {
    std::cout << "No GPU agents available. Skipping test." << std::endl;
    return;
  }

  latency_without_preload_us_ = MeasureProfilingEnableLatencyWithoutPreload();
  latency_with_preload_us_ = MeasureProfilingEnableLatencyWithPreload();
  latency_improvement_us_ = latency_without_preload_us_ - latency_with_preload_us_;
  EXPECT_GT(latency_improvement_us_, 0.0);

  blit_latency_without_preload_us_ = MeasureFirstAsyncCopyLatencyWithoutPreload();
  blit_latency_with_preload_us_ = MeasureFirstAsyncCopyLatencyWithPreload();
  blit_latency_improvement_us_ = blit_latency_without_preload_us_ - blit_latency_with_preload_us_;
  EXPECT_GT(blit_latency_improvement_us_, 0.0);
}

double AgentPreloadTest::MeasureProfilingEnableLatencyWithoutPreload() {
  hsa_status_t err;

  err = hsa_shut_down();
  EXPECT_TRUE(err == HSA_STATUS_SUCCESS ||
                                      err == HSA_STATUS_ERROR_NOT_INITIALIZED);

  err = hsa_init();
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpus;
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  if (gpus.empty()) {
    return 0.0;
  }

  // Measure time for hsa_amd_profiling_async_copy_enable without preload
  auto start = std::chrono::high_resolution_clock::now();
  err = hsa_amd_profiling_async_copy_enable(true);
  auto end = std::chrono::high_resolution_clock::now();

  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_profiling_async_copy_enable(false);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  double latency_us = duration.count() / 1000.0;

  if (verbosity() > 0) {
    std::cout << "  Profiling enable latency WITHOUT preload: "
              << std::fixed << std::setprecision(2) << latency_us << " us" << std::endl;
  }

  return latency_us;
}

double AgentPreloadTest::MeasureProfilingEnableLatencyWithPreload() {
  hsa_status_t err;

  err = hsa_shut_down();
  EXPECT_TRUE(err == HSA_STATUS_SUCCESS ||
                                      err == HSA_STATUS_ERROR_NOT_INITIALIZED);

  err = hsa_init();
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpus;
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  if (gpus.empty()) {
    return 0.0;
  }

  // Preload clock sync for all GPU agents
  for (const auto& gpu : gpus) {
    err = hsa_amd_agent_preload(gpu, HSA_AMD_AGENT_PRELOAD_SKIP_BLITS);
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
  }

  // Measure time for hsa_amd_profiling_async_copy_enable with preload
  auto start = std::chrono::high_resolution_clock::now();
  err = hsa_amd_profiling_async_copy_enable(true);
  auto end = std::chrono::high_resolution_clock::now();

  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_profiling_async_copy_enable(false);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  double latency_us = duration.count() / 1000.0;

  if (verbosity() > 0) {
    std::cout << "  Profiling enable latency WITH preload: "
              << std::fixed << std::setprecision(2) << latency_us << " us" << std::endl;
  }

  return latency_us;
}

double AgentPreloadTest::MeasureFirstAsyncCopyLatencyWithoutPreload() {
  hsa_status_t err;
  err = hsa_shut_down();
  EXPECT_TRUE(err == HSA_STATUS_SUCCESS ||
                                      err == HSA_STATUS_ERROR_NOT_INITIALIZED);

  err = hsa_init();
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpus;
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  if (gpus.empty() || cpus.empty()) {
    return 0.0;
  }

  hsa_amd_memory_pool_t gpu_pool;
  err = hsa_amd_agent_iterate_memory_pools(gpus[0], FindGlobalPool, &gpu_pool);
  if (err != HSA_STATUS_INFO_BREAK) {
    return 0.0;
  }

  hsa_amd_memory_pool_t cpu_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpus[0], FindGlobalPool, &cpu_pool);
  if (err != HSA_STATUS_INFO_BREAK) {
    return 0.0;
  }

  const size_t size = 4096;
  void* src_ptr = nullptr;
  void* dst_ptr = nullptr;

  err = hsa_amd_memory_pool_allocate(cpu_pool, size, 0, &src_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    return 0.0;
  }

  err = hsa_amd_memory_pool_allocate(gpu_pool, size, 0, &dst_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_amd_memory_pool_free(src_ptr);
    return 0.0;
  }

  err = hsa_amd_agents_allow_access(1, &gpus[0], nullptr, src_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }

  // Measure first async copy without preloading blits
  auto start = std::chrono::high_resolution_clock::now();
  err = hsa_amd_memory_async_copy(dst_ptr, gpus[0], src_ptr, cpus[0], size, 0,
                                                              nullptr, signal);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_signal_destroy(signal);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }

  hsa_signal_value_t signal_value =
                      hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT,
                                        1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  if (signal_value != 0) {
    EXPECT_EQ(signal_value, 0);
    hsa_signal_destroy(signal);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }

  auto end = std::chrono::high_resolution_clock::now();

  hsa_signal_destroy(signal);
  hsa_amd_memory_pool_free(src_ptr);
  hsa_amd_memory_pool_free(dst_ptr);

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  double latency_us = duration.count() / 1000.0;

  if (verbosity() > 0) {
    std::cout << "  First async copy latency WITHOUT blit preload: "
              << std::fixed << std::setprecision(2) << latency_us << " us" << std::endl;
  }

  return latency_us;
}

double AgentPreloadTest::MeasureFirstAsyncCopyLatencyWithPreload() {
  hsa_status_t err;
  err = hsa_shut_down();
  EXPECT_TRUE(err == HSA_STATUS_SUCCESS ||
                                      err == HSA_STATUS_ERROR_NOT_INITIALIZED);

  err = hsa_init();
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpus;
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  EXPECT_EQ(err, HSA_STATUS_SUCCESS);

  if (gpus.empty() || cpus.empty()) {
    return 0.0;
  }

  // Preload blits
  for (const auto& gpu : gpus) {
    err = hsa_amd_agent_preload(gpu, HSA_AMD_AGENT_PRELOAD_SKIP_CLOCK_SYNC);
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
  }

  hsa_amd_memory_pool_t gpu_pool;
  err = hsa_amd_agent_iterate_memory_pools(gpus[0], FindGlobalPool, &gpu_pool);
  if (err != HSA_STATUS_INFO_BREAK) {
    return 0.0;
  }

  hsa_amd_memory_pool_t cpu_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpus[0], FindGlobalPool, &cpu_pool);
  if (err != HSA_STATUS_INFO_BREAK) {
    return 0.0;
  }

  const size_t size = 4096;
  void* src_ptr = nullptr;
  void* dst_ptr = nullptr;

  err = hsa_amd_memory_pool_allocate(cpu_pool, size, 0, &src_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    return 0.0;
  }

  err = hsa_amd_memory_pool_allocate(gpu_pool, size, 0, &dst_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_amd_memory_pool_free(src_ptr);
    return 0.0;
  }

  err = hsa_amd_agents_allow_access(1, &gpus[0], nullptr, src_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }

  hsa_signal_t signal;
  err = hsa_signal_create(1, 0, nullptr, &signal);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }

  // Measure first async copy with blits preloaded
  auto start = std::chrono::high_resolution_clock::now();
  err = hsa_amd_memory_async_copy(dst_ptr, gpus[0], src_ptr, cpus[0], size, 0, nullptr, signal);
  if (err != HSA_STATUS_SUCCESS) {
    EXPECT_EQ(err, HSA_STATUS_SUCCESS);
    hsa_signal_destroy(signal);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }

  hsa_signal_value_t signal_value =
                      hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT,
                                        1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  if (signal_value != 0) {
    EXPECT_EQ(signal_value, 0);
    hsa_signal_destroy(signal);
    hsa_amd_memory_pool_free(src_ptr);
    hsa_amd_memory_pool_free(dst_ptr);
    return 0.0;
  }
  auto end = std::chrono::high_resolution_clock::now();

  hsa_signal_destroy(signal);
  hsa_amd_memory_pool_free(src_ptr);
  hsa_amd_memory_pool_free(dst_ptr);

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  double latency_us = duration.count() / 1000.0;

  if (verbosity() > 0) {
    std::cout << "  First async copy latency WITH blit preload: "
              << std::fixed << std::setprecision(2) << latency_us << " us" << std::endl;
  }

  return latency_us;
}

void AgentPreloadTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void AgentPreloadTest::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void AgentPreloadTest::Close() {
  TestBase::Close();
}
