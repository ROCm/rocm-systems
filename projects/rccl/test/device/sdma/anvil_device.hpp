/* Test stub: no-op anvil device ops for template coverage tests. */
#pragma once

#include "sdma_opcodes.h"

namespace sdma_anvil {

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

// Call log so tests can observe the *order and kind* of ::sdma_anvil::put()
// vs ::sdma_anvil::putSignal() calls the device backend issues per segment,
// without needing to inspect real SDMA hardware queues.
struct SdmaCallLogEntry {
  int kind;      // 0 = put (no signal), 1 = putSignal (fused signal)
  size_t bytes;  // segment size passed to the call
};
static constexpr int kSdmaCallLogCapacity = 8;
__device__ SdmaCallLogEntry g_sdmaCallLog[kSdmaCallLogCapacity];
__device__ int g_sdmaCallLogCount = 0;

__device__ __forceinline__ void put(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size) {
  (void)handle;
  int i = atomicAdd(&g_sdmaCallLogCount, 1);
  if (i >= 0 && i < kSdmaCallLogCapacity) {
    g_sdmaCallLog[i].kind = 0;
    g_sdmaCallLog[i].bytes = size;
  }
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void putSignal(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size,
                                          uint64_t* signal) {
  (void)signal;
  (void)handle;
  int i = atomicAdd(&g_sdmaCallLogCount, 1);
  if (i >= 0 && i < kSdmaCallLogCapacity) {
    g_sdmaCallLog[i].kind = 1;
    g_sdmaCallLog[i].bytes = size;
  }
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void quiet(SdmaQueueDeviceHandle& handle) { (void)handle; }

}  // namespace sdma_anvil
