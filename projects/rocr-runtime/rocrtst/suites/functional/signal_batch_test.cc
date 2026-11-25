/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include "suites/functional/signal_batch_test.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

static const char kSubTestSeparator[] = "  **************************";

static void PrintSubtestHeader(const char* header) {
  std::cout << "  *** Signal Batch Subtest: " << header << " ***" << std::endl;
}

namespace {

struct PoolInfo {
  hsa_amd_memory_pool_t pool;
  uint32_t global_flags;
};

hsa_status_t CollectGlobalPools(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  hsa_status_t err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (err != HSA_STATUS_SUCCESS) return err;
  if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
  PoolInfo pool_info = {pool, 0};
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                     &pool_info.global_flags);
  if (err != HSA_STATUS_SUCCESS) return err;
  static_cast<std::vector<PoolInfo>*>(data)->push_back(pool_info);
  return HSA_STATUS_SUCCESS;
}

std::vector<PoolInfo> GlobalPools(hsa_agent_t agent) {
  std::vector<PoolInfo> pools;
  EXPECT_EQ(hsa_amd_agent_iterate_memory_pools(agent, CollectGlobalPools, &pools),
            HSA_STATUS_SUCCESS);
  return pools;
}

bool SignalCapable(hsa_amd_memory_pool_t pool) {
  bool signal_capable = false;
  EXPECT_EQ(hsa_amd_memory_pool_get_info(
                pool, static_cast<hsa_amd_memory_pool_info_t>(HSA_AMD_MEMORY_POOL_INFO_SIGNAL_CAPABLE),
                &signal_capable),
            HSA_STATUS_SUCCESS);
  return signal_capable;
}

hsa_amd_memory_pool_t HostFinePool(hsa_agent_t cpu_agent) {
  for (const PoolInfo& pool_info : GlobalPools(cpu_agent)) {
    if (pool_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) return pool_info.pool;
  }
  return hsa_amd_memory_pool_t{0};
}

std::vector<hsa_agent_t> Agents(hsa_status_t (*iterate)(hsa_agent_t, void*)) {
  std::vector<hsa_agent_t> agents;
  EXPECT_EQ(hsa_iterate_agents(iterate, &agents), HSA_STATUS_SUCCESS);
  return agents;
}

}  // namespace

SignalBatchTest::SignalBatchTest(void) : TestBase() {
  set_num_iteration(1);
  set_title("  *** Signal Batch Tests ***");
  set_description(
      "  *** Tests hsa_amd_signal_batch_create/destroy, the signal pool and agent queries, and "
      "signal handle hints ***");
}

SignalBatchTest::~SignalBatchTest(void) {}

void SignalBatchTest::SetUp(void) {
  TestBase::SetUp();
}

void SignalBatchTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) return;
  TestBase::Run();
}

void SignalBatchTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void SignalBatchTest::DisplayResults(void) const {
  TestBase::DisplayResults();
  std::cout << std::endl;
  for (const auto& result : resultList_) std::cout << result << std::endl;
}

void SignalBatchTest::Close() { TestBase::Close(); }

void SignalBatchTest::HostPoolBatch() {
  PrintSubtestHeader("Batch in host fine-grained memory");
  const std::vector<hsa_agent_t> cpus = Agents(rocrtst::IterateCPUAgents);
  ASSERT_GT(cpus.size(), 0u);
  const hsa_amd_memory_pool_t host_pool = HostFinePool(cpus[0]);
  ASSERT_NE(host_pool.handle, 0u);
  ASSERT_TRUE(SignalCapable(host_pool));

  for (uint64_t attributes : {uint64_t{0}, uint64_t{HSA_AMD_SIGNAL_AMD_GPU_ONLY}}) {
    constexpr uint32_t kCount = 16;
    std::vector<hsa_signal_value_t> initial_values(kCount);
    for (uint32_t member = 0; member < kCount; ++member) initial_values[member] = 100 + member;
    std::vector<hsa_signal_t> signals(kCount);
    hsa_amd_signal_batch_t batch;
    ASSERT_EQ(hsa_amd_signal_batch_create(kCount, initial_values.data(), 0, nullptr, attributes,
                                          host_pool, &batch, signals.data()),
              HSA_STATUS_SUCCESS);

    for (uint32_t member = 0; member < kCount; ++member) {
      ASSERT_EQ(hsa_signal_load_scacquire(signals[member]), initial_values[member]);
      ASSERT_EQ(hsa_amd_signal_get_hints(signals[member]), 0u);
      for (uint32_t earlier = 0; earlier < member; ++earlier) {
        ASSERT_NE(signals[earlier].handle, signals[member].handle);
      }
    }

    // A blocked waiter is released by another thread.
    hsa_signal_store_screlease(signals[3], 1);
    std::thread releaser([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      hsa_signal_subtract_screlease(signals[3], 1);
    });
    const hsa_signal_value_t observed = hsa_signal_wait_scacquire(
        signals[3], HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    releaser.join();
    ASSERT_EQ(observed, 0);

    // Members and the batch handle may be destroyed in any order; survivors stay usable.
    ASSERT_EQ(hsa_signal_destroy(signals[0]), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_signal_batch_destroy(batch), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_signal_batch_destroy(batch), HSA_STATUS_ERROR_INVALID_ARGUMENT);
    hsa_signal_store_screlease(signals[5], 42);
    ASSERT_EQ(hsa_signal_load_scacquire(signals[5]), 42);
    for (uint32_t member = 1; member < kCount; ++member) {
      ASSERT_EQ(hsa_signal_destroy(signals[member]), HSA_STATUS_SUCCESS);
    }
  }
  resultList_.push_back("    Host fine-grained batches: values, waits and destroy order OK");
  std::cout << kSubTestSeparator << std::endl;
}

void SignalBatchTest::DevicePoolsFollowQuery() {
  PrintSubtestHeader("Device pools accepted exactly when SIGNAL_CAPABLE");
  const std::vector<hsa_agent_t> cpus = Agents(rocrtst::IterateCPUAgents);
  const std::vector<hsa_agent_t> gpus = Agents(rocrtst::IterateGPUAgents);
  ASSERT_GT(gpus.size(), 0u);

  for (const PoolInfo& pool_info : GlobalPools(cpus[0])) {
    const bool fine = pool_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
    ASSERT_EQ(SignalCapable(pool_info.pool), fine);
  }

  for (uint32_t gpu_idx = 0; gpu_idx < gpus.size(); ++gpu_idx) {
    uint32_t num_capable = 0;
    for (const PoolInfo& pool_info : GlobalPools(gpus[gpu_idx])) {
      const bool signal_capable = SignalCapable(pool_info.pool);
      constexpr uint32_t kCount = 4;
      const hsa_signal_value_t initial_values[kCount] = {1, 2, 3, 4};
      hsa_signal_t signals[kCount];
      hsa_amd_signal_batch_t batch;
      const hsa_status_t status =
          hsa_amd_signal_batch_create(kCount, initial_values, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY,
                                      pool_info.pool, &batch, signals);
      if (!signal_capable) {
        ASSERT_EQ(status, static_cast<hsa_status_t>(HSA_STATUS_ERROR_INVALID_MEMORY_POOL));
        continue;
      }
      ++num_capable;
      ASSERT_EQ(status, HSA_STATUS_SUCCESS);
      // Device-memory members take host loads and stores; RMW needs host atomics.
      for (uint32_t member = 0; member < kCount; ++member) {
        ASSERT_EQ(hsa_signal_load_scacquire(signals[member]), initial_values[member]);
        hsa_signal_store_screlease(signals[member], 0);
        ASSERT_EQ(hsa_signal_load_scacquire(signals[member]), 0);
      }
      ASSERT_EQ(hsa_amd_signal_batch_destroy(batch), HSA_STATUS_SUCCESS);
      for (uint32_t member = 0; member < kCount; ++member) {
        ASSERT_EQ(hsa_signal_destroy(signals[member]), HSA_STATUS_SUCCESS);
      }
    }
    std::stringstream summary;
    summary << "    GPU[" << gpu_idx << "] signal-capable device pools: " << num_capable;
    resultList_.push_back(summary.str());
    std::cout << summary.str() << std::endl;
  }
  std::cout << kSubTestSeparator << std::endl;
}

void SignalBatchTest::AsyncCopyCompletesMember() {
  PrintSubtestHeader("Async copy completes a batch member");
  const std::vector<hsa_agent_t> cpus = Agents(rocrtst::IterateCPUAgents);
  const std::vector<hsa_agent_t> gpus = Agents(rocrtst::IterateGPUAgents);
  ASSERT_GT(gpus.size(), 0u);
  const hsa_agent_t gpu = gpus[0];
  const hsa_amd_memory_pool_t host_pool = HostFinePool(cpus[0]);

  hsa_amd_memory_pool_t device_buffer_pool{0};
  std::vector<hsa_amd_memory_pool_t> signal_pools = {host_pool};
  for (const PoolInfo& pool_info : GlobalPools(gpu)) {
    if ((pool_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
        device_buffer_pool.handle == 0) {
      device_buffer_pool = pool_info.pool;
    }
    if (SignalCapable(pool_info.pool)) signal_pools.push_back(pool_info.pool);
  }
  ASSERT_NE(device_buffer_pool.handle, 0u);

  constexpr size_t kBytes = 1 << 20;
  void* host_buffer = nullptr;
  void* device_buffer = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(host_pool, kBytes, 0, &host_buffer), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(device_buffer_pool, kBytes, 0, &device_buffer),
            HSA_STATUS_SUCCESS);
  const hsa_agent_t access_agents[2] = {cpus[0], gpu};
  ASSERT_EQ(hsa_amd_agents_allow_access(2, access_agents, nullptr, host_buffer), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_agents_allow_access(2, access_agents, nullptr, device_buffer),
            HSA_STATUS_SUCCESS);
  memset(host_buffer, 0x5a, kBytes);

  for (const hsa_amd_memory_pool_t& signal_pool : signal_pools) {
    const hsa_signal_value_t initial_value = 1;
    hsa_signal_t completion;
    hsa_amd_signal_batch_t batch;
    ASSERT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 0, nullptr, 0, signal_pool, &batch,
                                          &completion),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_memory_async_copy(device_buffer, gpu, host_buffer, cpus[0], kBytes, 0,
                                        nullptr, completion),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_signal_wait_scacquire(completion, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                        HSA_WAIT_STATE_BLOCKED),
              0);
    ASSERT_EQ(hsa_signal_destroy(completion), HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_signal_batch_destroy(batch), HSA_STATUS_SUCCESS);
  }
  ASSERT_EQ(hsa_amd_memory_pool_free(host_buffer), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_free(device_buffer), HSA_STATUS_SUCCESS);

  std::stringstream summary;
  summary << "    Async copy completed members from " << signal_pools.size() << " pools";
  resultList_.push_back(summary.str());
  std::cout << summary.str() << std::endl << kSubTestSeparator << std::endl;
}

void SignalBatchTest::NegativeArguments() {
  PrintSubtestHeader("Invalid arguments");
  const std::vector<hsa_agent_t> cpus = Agents(rocrtst::IterateCPUAgents);
  const hsa_amd_memory_pool_t host_pool = HostFinePool(cpus[0]);
  const hsa_signal_value_t initial_value = 1;
  hsa_signal_t signal;
  hsa_amd_signal_batch_t batch;

  ASSERT_EQ(hsa_amd_signal_batch_create(0, &initial_value, 0, nullptr, 0, host_pool, &batch,
                                        &signal),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(hsa_amd_signal_batch_create(1, nullptr, 0, nullptr, 0, host_pool, &batch, &signal),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 0, nullptr, 0, host_pool, nullptr,
                                        &signal),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 0, nullptr, 0, host_pool, &batch,
                                        nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 0, nullptr, HSA_AMD_SIGNAL_IPC,
                                        host_pool, &batch, &signal),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  const hsa_agent_t duplicate_consumers[2] = {cpus[0], cpus[0]};
  ASSERT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 2, duplicate_consumers, 0, host_pool,
                                        &batch, &signal),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 0, nullptr, 0,
                                        hsa_amd_memory_pool_t{0}, &batch, &signal),
            static_cast<hsa_status_t>(HSA_STATUS_ERROR_INVALID_MEMORY_POOL));
  for (const PoolInfo& pool_info : GlobalPools(cpus[0])) {
    if (pool_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) continue;
    ASSERT_EQ(hsa_amd_signal_batch_create(1, &initial_value, 0, nullptr, 0, pool_info.pool,
                                          &batch, &signal),
              static_cast<hsa_status_t>(HSA_STATUS_ERROR_INVALID_MEMORY_POOL));
  }
  ASSERT_EQ(hsa_amd_signal_batch_destroy(hsa_amd_signal_batch_t{0}),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  resultList_.push_back("    Invalid arguments rejected");
  std::cout << kSubTestSeparator << std::endl;
}

void SignalBatchTest::AgentQueries() {
  PrintSubtestHeader("PCIE_STORES_NEED_FLUSH and HOST_ATOMICS_SUPPORTED");
  const std::vector<hsa_agent_t> cpus = Agents(rocrtst::IterateCPUAgents);
  const std::vector<hsa_agent_t> gpus = Agents(rocrtst::IterateGPUAgents);
  const auto pcie_stores_attribute =
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH);
  const auto host_atomics_attribute =
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HOST_ATOMICS_SUPPORTED);

  for (uint32_t gpu_idx = 0; gpu_idx < gpus.size(); ++gpu_idx) {
    bool pcie_stores_need_flush = false;
    bool host_atomics_supported = false;
    ASSERT_EQ(hsa_agent_get_info(gpus[gpu_idx], pcie_stores_attribute, &pcie_stores_need_flush),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_agent_get_info(gpus[gpu_idx], host_atomics_attribute, &host_atomics_supported),
              HSA_STATUS_SUCCESS);
    // Memory reached over PCIe takes no host atomics; coherent xGMI (A+A) needs no flush.
    ASSERT_NE(pcie_stores_need_flush, host_atomics_supported);
    std::stringstream summary;
    summary << "    GPU[" << gpu_idx << "] PCIe stores need flush: " << pcie_stores_need_flush
            << ", host atomics: " << host_atomics_supported;
    resultList_.push_back(summary.str());
    std::cout << summary.str() << std::endl;
  }

  bool unused = false;
  ASSERT_EQ(hsa_agent_get_info(cpus[0], pcie_stores_attribute, &unused),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  ASSERT_EQ(hsa_agent_get_info(cpus[0], host_atomics_attribute, &unused),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  std::cout << kSubTestSeparator << std::endl;
}

void SignalBatchTest::HandleHints() {
  PrintSubtestHeader("Signal handle hints");
  hsa_signal_t signal;
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  ASSERT_EQ(signal.handle & AMD_SIGNAL_HINT_MASK, 0u);
  ASSERT_EQ(hsa_amd_signal_get_hints(signal), 0u);

  const uint64_t hints =
      AMD_SIGNAL_HINT_INTR_NONE | AMD_SIGNAL_HINT_VALUE_BINARY | AMD_SIGNAL_HINT_TS_OFF;
  const hsa_signal_t hinted = hsa_amd_signal_set_hints(signal, hints);
  ASSERT_EQ(hsa_amd_signal_get_hints(hinted), hints);
  ASSERT_EQ(amd_signal_from_handle(hinted.handle),
            reinterpret_cast<amd_signal_t*>(static_cast<uintptr_t>(signal.handle)));
  // Setting replaces the old hints, and bits above the mask never reach the address.
  ASSERT_EQ(hsa_amd_signal_get_hints(hsa_amd_signal_set_hints(hinted, AMD_SIGNAL_HINT_TS_ON)),
            static_cast<uint64_t>(AMD_SIGNAL_HINT_TS_ON));
  ASSERT_EQ(hsa_amd_signal_set_hints(signal, (AMD_SIGNAL_HINT_MASK + 1) | hints).handle,
            hinted.handle);
  ASSERT_EQ(hsa_amd_signal_set_hints(hinted, 0).handle, signal.handle);

  ASSERT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  resultList_.push_back("    Handle hints set, read back and stripped");
  std::cout << kSubTestSeparator << std::endl;
}
