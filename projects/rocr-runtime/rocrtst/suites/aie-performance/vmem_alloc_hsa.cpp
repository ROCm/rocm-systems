// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Microbenchmark for hsa_amd_vmem_handle_create / hsa_amd_vmem_handle_release
// of a fixed 4096-byte allocation.

#include <benchmark/benchmark.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr std::size_t ALLOC_SIZE = 4096;

/// @brief `hsa_iterate_agents` callback that selects the first agent whose device
/// type matches `DeviceType`.
///
/// @tparam    DeviceType  Target device class (e.g. HSA_DEVICE_TYPE_AIE).
/// @param[in] agent       Agent supplied by the runtime on each iteration.
/// @param[out] data       Must point to an `hsa_agent_t` that receives the
///                        matching agent on success.
/// @return HSA_STATUS_INFO_BREAK once a match is stored (stops iteration),
///         HSA_STATUS_SUCCESS to continue iterating, or the underlying
///         error from `hsa_agent_get_info` on failure.
template <hsa_device_type_t DeviceType> hsa_status_t find_agent(hsa_agent_t agent, void* data) {
  hsa_device_type_t type{};
  if (auto s = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type); s != HSA_STATUS_SUCCESS) {
    return s;
  }
  if (type == DeviceType) {
    *static_cast<hsa_agent_t*>(data) = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

/// @brief `hsa_amd_agent_iterate_memory_pools` callback that selects the first
/// global-segment pool whose runtime allocation granule is non-zero and
/// no larger than `ALLOC_SIZE`. A granule larger than the request would
/// either fail or round up the allocation, skewing the benchmark.
///
/// @param[in]  pool  Memory pool supplied by the runtime on each iteration.
/// @param[out] data  Must point to an `hsa_amd_memory_pool_t` that receives
///                   the matching pool on success.
/// @return HSA_STATUS_INFO_BREAK once a match is stored (stops iteration),
///         HSA_STATUS_SUCCESS to skip and continue, or the underlying
///         error from `hsa_amd_memory_pool_get_info`.
hsa_status_t find_vmem_pool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment{};
  if (auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
      s != HSA_STATUS_SUCCESS || segment != HSA_AMD_SEGMENT_GLOBAL) {
    return s;
  }

  std::size_t granule = 0;
  if (auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                            &granule);
      s != HSA_STATUS_SUCCESS || granule == 0 || granule > ALLOC_SIZE) {
    return s;
  }

  *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
  return HSA_STATUS_INFO_BREAK;
}

}  // namespace

namespace {

/// @brief Time a batch of `hsa_amd_vmem_handle_create` calls followed by the
/// matching `hsa_amd_vmem_handle_release` calls, for `ALLOC_SIZE` virtual-memory
/// handles on the first agent of type `DeviceType`.
///
/// The batch size is `state.range(0)`. Each timing iteration allocates that
/// many handles, then releases them all, so item/byte counters scale with the
/// batch. Partial batches are released on failure to avoid leaking handles
/// across iterations.
///
/// Handles HSA lifetime (init/shut_down), agent and pool discovery. If the
/// target agent or a suitable pool is unavailable the case is skipped rather
/// than failed, so the suite can run on machines lacking the device.
///
/// @tparam        DeviceType   Device class to benchmark (AIE or GPU).
/// @param[in,out] state        Google Benchmark state driving the timing loop;
///                             `range(0)` is the per-iteration allocation count,
///                             and skip status / counters are written back.
/// @param[in]     agent_label  Human-readable name used in skip messages.
template <hsa_device_type_t DeviceType>
void RunVMemAllocRelease(benchmark::State& state, const char* agent_label) {
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    state.SkipWithError("hsa_init failed");
    return;
  }

  hsa_agent_t agent{};
  if (hsa_iterate_agents(find_agent<DeviceType>, &agent) != HSA_STATUS_INFO_BREAK) {
    state.SkipWithError((std::string("No ") + agent_label + " agent found").c_str());
    hsa_shut_down();
    return;
  }

  hsa_amd_memory_pool_t pool{};
  if (hsa_amd_agent_iterate_memory_pools(agent, find_vmem_pool, &pool) != HSA_STATUS_INFO_BREAK) {
    state.SkipWithError("No suitable memory pool for vmem allocation");
    hsa_shut_down();
    return;
  }

  const auto num_allocs = state.range(0);
  std::vector<hsa_amd_vmem_alloc_handle_t> handles;
  handles.reserve(static_cast<std::size_t>(num_allocs));

  for (auto _ : state) {
    handles.clear();
    bool failed = false;
    for (std::int64_t i = 0; i < num_allocs; ++i) {
      hsa_amd_vmem_alloc_handle_t handle{};
      if (hsa_amd_vmem_handle_create(pool, ALLOC_SIZE, MEMORY_TYPE_NONE, 0, &handle) !=
          HSA_STATUS_SUCCESS) {
        state.SkipWithError("hsa_amd_vmem_handle_create failed");
        failed = true;
        break;
      }
      handles.push_back(handle);
    }
    benchmark::DoNotOptimize(handles.data());
    for (auto h : handles) {
      if (hsa_amd_vmem_handle_release(h) != HSA_STATUS_SUCCESS) {
        state.SkipWithError("hsa_amd_vmem_handle_release failed");
        failed = true;
      }
    }
    if (failed) break;
  }

  state.SetItemsProcessed(state.iterations() * num_allocs);
  state.SetBytesProcessed(state.iterations() * num_allocs * static_cast<std::int64_t>(ALLOC_SIZE));

  hsa_shut_down();
}

}  // namespace

/// vmem create/release on the first AIE agent.
static void VMemAllocReleaseAIE(benchmark::State& state) {
  RunVMemAllocRelease<HSA_DEVICE_TYPE_AIE>(state, "AIE");
}

/// vmem create/release on the first GPU agent.
static void VMemAllocReleaseGPU(benchmark::State& state) {
  RunVMemAllocRelease<HSA_DEVICE_TYPE_GPU>(state, "GPU");
}

BENCHMARK(VMemAllocReleaseAIE)->RangeMultiplier(2)->Range(1, 1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(VMemAllocReleaseGPU)->RangeMultiplier(2)->Range(1, 1024)->Unit(benchmark::kMicrosecond);
