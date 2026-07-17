/* Test stub: no-op anvil device ops for template coverage tests. */
#pragma once

#include "sdma_opcodes.h"

namespace gin_anvil {
namespace sdma {

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

__device__ __forceinline__ void quiet(SdmaQueueDeviceHandle& handle) { (void)handle; }

}  // namespace sdma
}  // namespace gin_anvil
