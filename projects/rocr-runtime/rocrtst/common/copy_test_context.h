/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_COMMON_COPY_TEST_CONTEXT_H_
#define ROCRTST_COMMON_COPY_TEST_CONTEXT_H_

// =============================================================================
// HsaTestContext: minimal HSA context manager for copy-operation tests.
//
// Stands apart from BaseRocR because these tests exercise the async batch copy
// API directly and need nothing the harness fixture sets up beyond an agent, a
// pool, and buffers reachable from both sides.
// =============================================================================

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "common/common.h"

// A completion signal that releases itself. The batch copy API validates
// op.completion_signal before it reaches any of the per-type fields, so a test
// that hands it a zero-handle signal gets INVALID_ARGUMENT from the signal
// check no matter what else the op says. A negative test built that way passes
// without ever reaching the argument it names.
class ScopedTestSignal {
 public:
  ScopedTestSignal() {
    if (hsa_signal_create(1, 0, nullptr, &signal_) != HSA_STATUS_SUCCESS) {
      signal_.handle = 0;
    }
  }

  ~ScopedTestSignal() {
    if (signal_.handle != 0) hsa_signal_destroy(signal_);
  }

  ScopedTestSignal(const ScopedTestSignal&) = delete;
  ScopedTestSignal& operator=(const ScopedTestSignal&) = delete;

  bool valid() const { return signal_.handle != 0; }
  hsa_signal_t get() const { return signal_; }

 private:
  hsa_signal_t signal_{0};
};

class HsaTestContext {
 public:
  hsa_agent_t gpu_agent;
  hsa_agent_t cpu_agent;
  hsa_amd_memory_pool_t gpu_pool;
  hsa_amd_memory_pool_t cpu_pool;
  std::vector<hsa_agent_t> all_gpu_agents;

  HsaTestContext() : gpu_agent{0}, cpu_agent{0}, gpu_pool{0}, cpu_pool{0} {
    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      std::cerr << "Warning: hsa_init failed" << std::endl;
      return;
    }
    initialized_ = true;

    // Agent discovery is the shared rocrtst callbacks, not a local copy of
    // them: a re-implementation here would drift from whatever the harness
    // considers a GPU the next time that definition changes.
    hsa_iterate_agents(rocrtst::IterateGPUAgents, &all_gpu_agents);

    if (!all_gpu_agents.empty()) gpu_agent = all_gpu_agents[0];

    hsa_iterate_agents(rocrtst::FindCPUDevice, &cpu_agent);

    if (gpu_agent.handle != 0) gpu_pool = FindCoarseGlobalPool(gpu_agent);
    if (cpu_agent.handle != 0) cpu_pool = FindCoarseGlobalPool(cpu_agent);
  }

  ~HsaTestContext() {
    // hsa_init/hsa_shut_down are reference counted, so releasing the reference
    // this context took cannot tear the runtime down under a concurrently live
    // user. Leaving it unbalanced is what breaks the harness: CommonCleanUp()
    // asserts that a second hsa_shut_down() returns NOT_INITIALIZED, which can
    // never hold while leaked references are outstanding.
    if (initialized_) hsa_shut_down();
  }

  // Each instance owns exactly one hsa_init reference; copying would release it
  // more than once.
  HsaTestContext(const HsaTestContext&) = delete;
  HsaTestContext& operator=(const HsaTestContext&) = delete;

  bool HasGPUAgent() const { return gpu_agent.handle != 0 && gpu_pool.handle != 0; }

  bool HasCPUAgent() const { return cpu_agent.handle != 0 && cpu_pool.handle != 0; }

  // Allocate a buffer the GPU can address and the CPU can still read and write,
  // so a test can fill a pattern, run the copy, and verify the result without a
  // staging step of its own. Allocated from system memory for that reason, not
  // from device-local memory. Returns nullptr on allocation failure.
  void* AllocateGPUBuffer(size_t size) {
    void* ptr = nullptr;
    if (hsa_amd_memory_pool_allocate(cpu_pool, size, 0, &ptr) != HSA_STATUS_SUCCESS) {
      return nullptr;
    }
    if (gpu_agent.handle != 0) hsa_amd_agents_allow_access(1, &gpu_agent, nullptr, ptr);
    return ptr;
  }

  // Allocate from the agent's own device-local pool. Use this where the point
  // of the test is the device-local copy path itself, since AllocateGPUBuffer
  // hands back system memory and would exercise a different route through the
  // runtime. The result is not guaranteed to be CPU-addressable, so a test that
  // verifies contents has to stage through a buffer from AllocateGPUBuffer.
  //
  // Returns nullptr on allocation failure, or if the agent exposes no
  // device-local pool.
  void* AllocateDeviceLocalBuffer(size_t size) {
    if (gpu_pool.handle == 0) return nullptr;

    void* ptr = nullptr;
    if (hsa_amd_memory_pool_allocate(gpu_pool, size, 0, &ptr) != HSA_STATUS_SUCCESS) {
      return nullptr;
    }
    return ptr;
  }

  void Free(void* ptr) {
    if (ptr != nullptr) hsa_amd_memory_pool_free(ptr);
  }

 private:
  // First coarse-grained global pool owned by agent, or a zero handle if it has
  // none.
  static hsa_amd_memory_pool_t FindCoarseGlobalPool(hsa_agent_t agent) {
    hsa_amd_memory_pool_t pool = {0};
    hsa_amd_agent_iterate_memory_pools(
        agent,
        [](hsa_amd_memory_pool_t candidate, void* data) -> hsa_status_t {
          hsa_amd_segment_t segment;
          hsa_amd_memory_pool_get_info(candidate, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
          if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

          uint32_t flags;
          hsa_amd_memory_pool_get_info(candidate, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
          if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0) {
            return HSA_STATUS_SUCCESS;
          }

          *static_cast<hsa_amd_memory_pool_t*>(data) = candidate;
          return HSA_STATUS_INFO_BREAK;
        },
        &pool);
    return pool;
  }

  // True only if this context's hsa_init() succeeded and therefore owes a
  // matching hsa_shut_down().
  bool initialized_ = false;
};

#endif  // ROCRTST_COMMON_COPY_TEST_CONTEXT_H_
