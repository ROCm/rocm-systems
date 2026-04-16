//===-- HSAPoolManager.cpp - HSA pool discovery implementation ------------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/HSAPoolManager.h"
#include "aegisbit/RuntimeConfig.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#include <string>
#include <vector>

namespace aegisbit {

void HSAPoolManager::ensureDiscovered() {
  std::call_once(OnceFlag, [this]() { discoverImpl(); });
}

void HSAPoolManager::discoverImpl() {
  RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  Cfg.log("Discovering HSA GPU agent and kernarg memory pool...");

#ifdef AEGISBIT_HAS_GPU
  // Enumerate ALL GPU agents (multi-GPU awareness).
  std::vector<hsa_agent_t> gpu_agents;
  hsa_status_t status = hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU) {
          static_cast<std::vector<hsa_agent_t> *>(data)->push_back(agent);
        }
        return HSA_STATUS_SUCCESS;
      },
      &gpu_agents);

  if (status != HSA_STATUS_SUCCESS) {
    Cfg.log("Failed to iterate HSA agents");
    return;
  }

  if (gpu_agents.empty()) {
    Cfg.log("No GPU agent found");
    return;
  }

  Cfg.log("Found " + std::to_string(gpu_agents.size()) + " GPU agent(s)");

  // Apply GPU filter from AEGISBIT_GPU_FILTER (parsed once in RuntimeConfig).
  // Negative = no filter (use GPU 0).
  int SelectedGPU = 0;
  if (Cfg.Debug.GPUFilter >= 0) {
    SelectedGPU = Cfg.Debug.GPUFilter;
    if (static_cast<size_t>(SelectedGPU) >= gpu_agents.size()) {
      Cfg.log("WARNING: AEGISBIT_GPU_FILTER=" +
              std::to_string(Cfg.Debug.GPUFilter) + " out of range (0-" +
              std::to_string(gpu_agents.size() - 1) + "), using GPU 0");
      SelectedGPU = 0;
    }
  }

  hsa_agent_t gpu_agent = gpu_agents[SelectedGPU];
  GPUAgentHandle = gpu_agent.handle;
  Cfg.log("Using GPU agent " + std::to_string(SelectedGPU) +
          " (handle: " + std::to_string(GPUAgentHandle) + ")");

  // Priority on GPU agent:
  //   KERNARG_INIT > FINE_GRAINED > EXTENDED_SCOPE_FINE_GRAINED
  // Fallback: CPU agent's KERNARG pool (accessible by all agents on gfx950).
  struct PoolSearchResult {
    hsa_amd_memory_pool_t kernarg_pool = {0};
    hsa_amd_memory_pool_t fine_grained_pool = {0};
    hsa_amd_memory_pool_t ext_fine_grained_pool = {0};
    hsa_amd_memory_pool_t vram_pool = {0};
  };

  auto pool_scanner = [](hsa_amd_memory_pool_t pool,
                         void *data) -> hsa_status_t {
    auto *result = static_cast<PoolSearchResult *>(data);
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                 &segment);
    if (segment != HSA_AMD_SEGMENT_GLOBAL)
      return HSA_STATUS_SUCCESS;

    uint32_t flags;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                 &flags);
    bool alloc_allowed = false;
    hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
    if (!alloc_allowed)
      return HSA_STATUS_SUCCESS;

    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
      result->kernarg_pool = pool;
    if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
        !(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) &&
        result->fine_grained_pool.handle == 0)
      result->fine_grained_pool = pool;
    if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED) &&
        result->ext_fine_grained_pool.handle == 0)
      result->ext_fine_grained_pool = pool;
    if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
        result->vram_pool.handle == 0)
      result->vram_pool = pool;
    return HSA_STATUS_SUCCESS;
  };

  PoolSearchResult gpu_pools;
  status =
      hsa_amd_agent_iterate_memory_pools(gpu_agent, pool_scanner, &gpu_pools);

  hsa_amd_memory_pool_t selected_pool = gpu_pools.kernarg_pool;
  if (selected_pool.handle == 0)
    selected_pool = gpu_pools.fine_grained_pool;
  if (selected_pool.handle == 0)
    selected_pool = gpu_pools.ext_fine_grained_pool;

  // Fallback: CPU agent KERNARG pool (accessible by all agents).
  if (selected_pool.handle == 0) {
    Cfg.log("No suitable GPU memory pool, trying CPU agent kernarg pool...");
    hsa_agent_t cpu_agent = {0};
    hsa_iterate_agents(
        [](hsa_agent_t agent, void *data) -> hsa_status_t {
          hsa_device_type_t type;
          hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
          if (type == HSA_DEVICE_TYPE_CPU) {
            *static_cast<hsa_agent_t *>(data) = agent;
            return HSA_STATUS_INFO_BREAK;
          }
          return HSA_STATUS_SUCCESS;
        },
        &cpu_agent);

    if (cpu_agent.handle != 0) {
      PoolSearchResult cpu_pools;
      hsa_amd_agent_iterate_memory_pools(cpu_agent, pool_scanner, &cpu_pools);
      if (cpu_pools.kernarg_pool.handle != 0)
        selected_pool = cpu_pools.kernarg_pool;
      else if (cpu_pools.fine_grained_pool.handle != 0)
        selected_pool = cpu_pools.fine_grained_pool;
    }
  }

  if (selected_pool.handle == 0) {
    Cfg.log("No CPU-accessible memory pool found on any agent");
    return;
  }

  hsa_amd_memory_pool_t kernarg_pool = selected_pool;
  KernargMemoryPool = kernarg_pool.handle;
  Cfg.log("Found kernarg memory pool: " + std::to_string(KernargMemoryPool));

  // For GPU-atomic-safe trace buffers: prefer VRAM, then fine-grained variants.
  if (gpu_pools.vram_pool.handle != 0) {
    FineGrainedMemoryPool = gpu_pools.vram_pool.handle;
    Cfg.log("Found VRAM pool for atomic-safe trace buffers: " +
            std::to_string(FineGrainedMemoryPool));
  } else if (gpu_pools.fine_grained_pool.handle != 0) {
    FineGrainedMemoryPool = gpu_pools.fine_grained_pool.handle;
    Cfg.log("Found fine-grained pool for atomic-safe trace buffers: " +
            std::to_string(FineGrainedMemoryPool));
  } else if (gpu_pools.ext_fine_grained_pool.handle != 0) {
    FineGrainedMemoryPool = gpu_pools.ext_fine_grained_pool.handle;
    Cfg.log("Found extended-fine-grained pool for atomic-safe trace buffers: " +
            std::to_string(FineGrainedMemoryPool));
  }

  bool cpu_accessible = false;
  hsa_amd_memory_pool_get_info(kernarg_pool,
                               HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                               &cpu_accessible);
  Cfg.log("Kernarg pool CPU-accessible: " +
          std::string(cpu_accessible ? "yes" : "no"));

  Initialized = true;
  Cfg.log("HSA pool discovery complete");
#else
  Cfg.log("GPU not available, skipping HSA pool discovery");
#endif
}

} // namespace aegisbit
