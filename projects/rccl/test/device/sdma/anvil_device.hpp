/* Test stub: no-op anvil device ops for template coverage tests. */
#pragma once

#include "sdma_opcodes.h"

namespace sdma_anvil {

// Declared here; the single definition lives in anvil_stub_quiet_count.cpp
// because every TU that includes gin_anvil_sdma.h would otherwise emit one.
// test/CMakeLists.txt compiles IPC, Suite H and that TU -fgpu-rdc and
// device-links rccl-UnitTestsFixtures whenever ENABLE_ROCSHMEM_GIN is on.
extern __device__ unsigned long long g_sdmaStubQuietCount;
// When non-null, quiet() ORs g_sdmaStubMarkBitOnQuiet into it, simulating a put
// that rings its doorbell and marks dirty while Flush is draining.
extern __device__ uint64_t* g_sdmaStubMarkDirtyOnQuiet;
extern __device__ uint64_t g_sdmaStubMarkBitOnQuiet;
// When non-null, quiet() accumulates the dirty word as it is on entry into
// g_sdmaStubDirtyAtQuiet. Lets a test see whether Flush cleared a bit before or
// after draining it, which is what a concurrently flushing CTA would observe.
extern __device__ uint64_t* g_sdmaStubObserveDirtyOnQuiet;
extern __device__ unsigned long long g_sdmaStubDirtyAtQuiet;

struct SdmaQueueDeviceHandle {
  int tag;
};

struct SdmaQueueSingleProducerDeviceHandle {
  int tag;
};

__device__ __forceinline__ void memcpyDevice(void* dst, const void* src, size_t size) {
  if (dst == nullptr || src == nullptr || size == 0) return;
  auto* d = static_cast<char*>(dst);
  const auto* s = static_cast<const char*>(src);
  for (size_t i = 0; i < size; ++i) d[i] = s[i];
}

__device__ __forceinline__ void put(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size) {
  (void)handle;
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void putSignal(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size,
                                          uint64_t* signal) {
  (void)signal;
  (void)handle;
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void quiet(SdmaQueueDeviceHandle& handle) {
  (void)handle;
  atomicAdd(&g_sdmaStubQuietCount, 1ULL);
  uint64_t* observed = g_sdmaStubObserveDirtyOnQuiet;
  if (observed != nullptr) {
    atomicOr(&g_sdmaStubDirtyAtQuiet,
             static_cast<unsigned long long>(__hip_atomic_load(observed, __ATOMIC_RELAXED,
                                                               __HIP_MEMORY_SCOPE_AGENT)));
  }
  uint64_t* dirty = g_sdmaStubMarkDirtyOnQuiet;
  if (dirty != nullptr) {
    atomicOr(reinterpret_cast<unsigned long long*>(dirty),
             static_cast<unsigned long long>(g_sdmaStubMarkBitOnQuiet));
  }
}

}  // namespace sdma_anvil
