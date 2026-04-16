//===-- TraceBufferAllocator.cpp - Trace buffer alloc implementation ------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/TraceBufferAllocator.h"
#include "aegisbit/RuntimeConfig.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#include <cstring>
#include <string>

namespace aegisbit {

#ifdef AEGISBIT_HAS_GPU
namespace {

// Walk the HSA agent list, return the first CPU agent or the zero-handle if
// none exists. Used to grant the CPU explicit access to fine-grained
// (VRAM-backed) memory so that host memset/readback works.
hsa_agent_t findCpuAgent() {
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
  return cpu_agent;
}

} // namespace
#endif

std::tuple<void *, void *, size_t, bool>
TraceBufferAllocator::allocate(size_t Size) {
  RuntimeConfig &Cfg = RuntimeConfig::getInstance();

  if (!Pools.isInitialized() || Pools.kernargPool() == 0) {
    Cfg.log("HSA pools not initialized, cannot allocate trace buffer");
    return {nullptr, nullptr, 0, false};
  }

#ifdef AEGISBIT_HAS_GPU
  // Prefer fine-grained pool (supports GPU atomics); fall back to kernarg.
  bool UsedFineGrained = false;
  hsa_amd_memory_pool_t pool = {Pools.kernargPool()};
  if (Pools.fineGrainedPool() != 0) {
    pool = {Pools.fineGrainedPool()};
    UsedFineGrained = true;
  }

  void *trace_buf = nullptr;
  hsa_status_t status =
      hsa_amd_memory_pool_allocate(pool, Size, 0, &trace_buf);
  if (status != HSA_STATUS_SUCCESS || trace_buf == nullptr) {
    if (UsedFineGrained) {
      Cfg.log("Atomic-safe pool alloc failed, falling back to kernarg pool");
      pool = {Pools.kernargPool()};
      UsedFineGrained = false;
      status = hsa_amd_memory_pool_allocate(pool, Size, 0, &trace_buf);
    }
    if (status != HSA_STATUS_SUCCESS || trace_buf == nullptr) {
      Cfg.log("Failed to allocate trace buffer from HSA pool");
      return {nullptr, nullptr, 0, false};
    }
  }

  // VRAM/device memory needs explicit CPU access grant for memset/readback.
  if (UsedFineGrained) {
    hsa_agent_t cpu_agent = findCpuAgent();
    if (cpu_agent.handle != 0)
      hsa_amd_agents_allow_access(1, &cpu_agent, nullptr, trace_buf);
  }
  std::memset(trace_buf, 0, Size);

  // Allocate write offset (8 bytes, aligned) from the same pool.
  void *write_offset = nullptr;
  status = hsa_amd_memory_pool_allocate(pool, 8, 0, &write_offset);
  if (status != HSA_STATUS_SUCCESS || write_offset == nullptr) {
    Cfg.log("Failed to allocate write offset from HSA pool");
    hsa_amd_memory_pool_free(trace_buf);
    return {nullptr, nullptr, 0, false};
  }
  if (UsedFineGrained) {
    hsa_agent_t cpu_agent = findCpuAgent();
    if (cpu_agent.handle != 0)
      hsa_amd_agents_allow_access(1, &cpu_agent, nullptr, write_offset);
  }
  std::memset(write_offset, 0, 8);

  Cfg.log("Allocated HSA trace buffer: " + std::to_string(Size) +
          " bytes at " + std::to_string(reinterpret_cast<uint64_t>(trace_buf)) +
          ", write offset at " +
          std::to_string(reinterpret_cast<uint64_t>(write_offset)) +
          (UsedFineGrained ? " (fine-grained, atomics OK)" : " (kernarg)"));

  return {trace_buf, write_offset, Size, UsedFineGrained};
#else
  (void)Size;
  return {nullptr, nullptr, 0, false};
#endif
}

void TraceBufferAllocator::freeBuffer(void *TraceBufferPtr,
                                      void *WriteOffsetPtr) {
#ifdef AEGISBIT_HAS_GPU
  if (TraceBufferPtr)
    hsa_amd_memory_pool_free(TraceBufferPtr);
  if (WriteOffsetPtr)
    hsa_amd_memory_pool_free(WriteOffsetPtr);
#else
  (void)TraceBufferPtr;
  (void)WriteOffsetPtr;
#endif
}

} // namespace aegisbit
