//===-- aegisbit/HSAPoolManager.h - HSA pool discovery ----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Discovers the GPU agent, kernarg memory pool, and fine-grained
/// (GPU-atomic-safe) memory pool used by AegisBit's tracing buffers. Honors
/// `AEGISBIT_GPU_FILTER` via `RuntimeConfig::Debug::GPUFilter` when multiple
/// GPUs are present.
///
/// Extracted from TracingEngine. All state here is plain integer handles
/// (`uint64_t`) and a `std::once_flag`, so the manager has no destructor-side
/// HSA calls and is safe to outlive the HSA runtime.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_HSA_POOL_MANAGER_H
#define AEGISBIT_HSA_POOL_MANAGER_H

#include <cstdint>
#include <mutex>

namespace aegisbit {

class HSAPoolManager {
public:
  /// Discover pools on first call; subsequent calls are no-ops. Thread-safe.
  void ensureDiscovered();

  /// Whether `ensureDiscovered` has completed and a kernarg pool was found.
  bool isInitialized() const { return Initialized; }

  /// Raw `hsa_agent_t::handle` for the selected GPU agent. 0 if unavailable.
  uint64_t gpuAgent() const { return GPUAgentHandle; }

  /// Raw `hsa_amd_memory_pool_t::handle` for the CPU-accessible kernarg pool.
  /// 0 if discovery failed.
  uint64_t kernargPool() const { return KernargMemoryPool; }

  /// Raw `hsa_amd_memory_pool_t::handle` for a fine-grained (GPU-atomic-safe)
  /// pool when one is available, otherwise 0.
  uint64_t fineGrainedPool() const { return FineGrainedMemoryPool; }

private:
  void discoverImpl();

  std::once_flag OnceFlag;
  bool Initialized = false;
  uint64_t GPUAgentHandle = 0;
  uint64_t KernargMemoryPool = 0;
  uint64_t FineGrainedMemoryPool = 0;
};

} // namespace aegisbit

#endif // AEGISBIT_HSA_POOL_MANAGER_H
