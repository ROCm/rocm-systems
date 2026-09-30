// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Microbenchmark comparing HSA virtual-memory allocation cost across a sweep of
// allocation sizes (8 KiB, 16 KiB, 32 KiB, 64 KiB) on the NPU/AIE agent.
//
// Two cases are provided:
//   VMemHandleAIE  - physical allocation only (handle create/release).
//   VMemMappedAIE  - full usable allocation (handle + reserve + map + access).

#include <benchmark/benchmark.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

// Smallest size in the sweep; the chosen pool granule must be no larger than
// this so every swept size is representable without rounding up.
constexpr std::size_t MIN_ALLOC_SIZE = 8192;

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
/// global-segment pool whose runtime allocation granule is non-zero and no
/// larger than `MIN_ALLOC_SIZE`. A granule larger than the smallest request
/// would round up the allocation and skew the benchmark.
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
      s != HSA_STATUS_SUCCESS || granule == 0 || granule > MIN_ALLOC_SIZE) {
    return s;
  }

  *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
  return HSA_STATUS_INFO_BREAK;
}

/// @brief Resolve the first agent of `DeviceType` and a suitable vmem pool.
///
/// Performs HSA init, agent discovery, and pool discovery. On any failure the
/// benchmark case is skipped (not failed) so the suite can run on machines
/// lacking the device, and HSA is shut down before returning.
///
/// @tparam         DeviceType   Device class to target (AIE).
/// @param[in,out]  state        Benchmark state; skip status is written here.
/// @param[in]      agent_label  Human-readable name used in skip messages.
/// @param[out]     agent        Receives the matching agent on success.
/// @param[out]     pool         Receives the matching pool on success.
/// @return true if both were found (HSA left initialized), false otherwise
///         (HSA already shut down).
template <hsa_device_type_t DeviceType>
bool setup_agent_pool(benchmark::State& state, const char* agent_label, hsa_agent_t* agent,
                      hsa_amd_memory_pool_t* pool) {
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    state.SkipWithError("hsa_init failed");
    return false;
  }

  if (hsa_iterate_agents(find_agent<DeviceType>, agent) != HSA_STATUS_INFO_BREAK) {
    state.SkipWithError((std::string("No ") + agent_label + " agent found").c_str());
    hsa_shut_down();
    return false;
  }

  if (hsa_amd_agent_iterate_memory_pools(*agent, find_vmem_pool, pool) != HSA_STATUS_INFO_BREAK) {
    state.SkipWithError("No suitable memory pool for vmem allocation");
    hsa_shut_down();
    return false;
  }

  return true;
}

/// @brief Time `hsa_amd_vmem_handle_create` + `hsa_amd_vmem_handle_release` for a
/// single allocation of `state.range(0)` bytes on the first agent of
/// `DeviceType`. This measures physical allocation only, without mapping the
/// handle into a virtual address range.
///
/// @tparam        DeviceType   Device class to benchmark (AIE).
/// @param[in,out] state        Benchmark state; `range(0)` is the byte size.
/// @param[in]     agent_label  Human-readable name used in skip messages.
template <hsa_device_type_t DeviceType>
void RunVMemHandle(benchmark::State& state, const char* agent_label) {
  hsa_agent_t agent{};
  hsa_amd_memory_pool_t pool{};
  if (!setup_agent_pool<DeviceType>(state, agent_label, &agent, &pool)) {
    return;
  }

  const auto size = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    hsa_amd_vmem_alloc_handle_t handle{};
    if (hsa_amd_vmem_handle_create(pool, size, MEMORY_TYPE_NONE, 0, &handle) !=
        HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_handle_create failed");
      break;
    }
    benchmark::DoNotOptimize(handle);
    if (hsa_amd_vmem_handle_release(handle) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_handle_release failed");
      break;
    }
  }

  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(size));

  hsa_shut_down();
}

/// @brief Time the full usable vmem workflow for a single allocation of
/// `state.range(0)` bytes on the first agent of `DeviceType`: handle create,
/// address reserve, map, and set_access, followed by the matching teardown.
///
/// On any failure inside the loop the partial state is torn down in reverse
/// order to avoid leaking handles or reservations across iterations.
///
/// @tparam        DeviceType   Device class to benchmark (AIE).
/// @param[in,out] state        Benchmark state; `range(0)` is the byte size.
/// @param[in]     agent_label  Human-readable name used in skip messages.
template <hsa_device_type_t DeviceType>
void RunVMemMapped(benchmark::State& state, const char* agent_label) {
  hsa_agent_t agent{};
  hsa_amd_memory_pool_t pool{};
  if (!setup_agent_pool<DeviceType>(state, agent_label, &agent, &pool)) {
    return;
  }

  const auto size = static_cast<std::size_t>(state.range(0));

  const hsa_amd_memory_access_desc_t access_desc{HSA_ACCESS_PERMISSION_RW, agent};

  for (auto _ : state) {
    hsa_amd_vmem_alloc_handle_t handle{};
    if (hsa_amd_vmem_handle_create(pool, size, MEMORY_TYPE_NONE, 0, &handle) !=
        HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_handle_create failed");
      break;
    }

    void* va = nullptr;
    if (hsa_amd_vmem_address_reserve(&va, size, 0, 0) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_address_reserve failed");
      hsa_amd_vmem_handle_release(handle);
      break;
    }

    if (hsa_amd_vmem_map(va, size, 0, handle, 0) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_map failed");
      hsa_amd_vmem_address_free(va, size);
      hsa_amd_vmem_handle_release(handle);
      break;
    }

    if (hsa_amd_vmem_set_access(va, size, &access_desc, 1) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_set_access failed");
      hsa_amd_vmem_unmap(va, size);
      hsa_amd_vmem_address_free(va, size);
      hsa_amd_vmem_handle_release(handle);
      break;
    }

    benchmark::DoNotOptimize(va);

    bool failed = false;
    if (hsa_amd_vmem_unmap(va, size) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_unmap failed");
      failed = true;
    }
    if (hsa_amd_vmem_address_free(va, size) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_address_free failed");
      failed = true;
    }
    if (hsa_amd_vmem_handle_release(handle) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("hsa_amd_vmem_handle_release failed");
      failed = true;
    }
    if (failed) break;
  }

  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(size));

  hsa_shut_down();
}

}  // namespace

/// vmem handle create/release on the first AIE agent.
static void VMemHandleAIE(benchmark::State& state) {
  RunVMemHandle<HSA_DEVICE_TYPE_AIE>(state, "AIE");
}

/// full vmem reserve/map/access workflow on the first AIE agent.
static void VMemMappedAIE(benchmark::State& state) {
  RunVMemMapped<HSA_DEVICE_TYPE_AIE>(state, "AIE");
}

BENCHMARK(VMemHandleAIE)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
BENCHMARK(VMemMappedAIE)->RangeMultiplier(2)->Range(8192, 65536)->Unit(benchmark::kMicrosecond);
