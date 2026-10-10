/* Test stub: no-op anvil device ops for template coverage tests. */
#pragma once

#include <cstddef>
#include <cstdint>

#include "sdma_opcodes.h"

namespace sdma_anvil {

// Declared here; the single definition lives in anvil_stub_quiet_count.cpp
// because every TU that includes gin_anvil_sdma.h would otherwise emit one.
// test/CMakeLists.txt compiles IPC, Suite H and that TU -fgpu-rdc and
// device-links rccl-UnitTestsFixtures whenever ENABLE_ROCSHMEM_GIN is on.
extern __device__ unsigned long long g_sdmaStubQuietCount;

// Opt-in call log: with recordOnly set, put/putSignal/quiet append here and put/putSignal skip the copy.
enum SdmaStubOp : int { kSdmaStubPut = 1, kSdmaStubPutSignal = 2, kSdmaStubQuiet = 3 };

struct SdmaStubCall {
  SdmaStubOp op;
  void* dst;
  void* src;
  size_t size;
  uint64_t* signal;
};

constexpr int kSdmaStubLogCap = 8;

struct SdmaStubLog {
  bool recordOnly;
  unsigned int count;
  SdmaStubCall calls[kSdmaStubLogCap];
};

// Defined in anvil_stub_quiet_count.cpp next to g_sdmaStubQuietCount.
extern __device__ SdmaStubLog g_sdmaStubLog;

// Returns true in record-only mode (caller skips the copy); calls past kSdmaStubLogCap only bump count.
__device__ __forceinline__ bool sdmaStubRecord(SdmaStubOp op, void* dst, void* src, size_t size, uint64_t* signal) {
  if (!g_sdmaStubLog.recordOnly) {
    return false;
  }
  const unsigned int idx = atomicAdd(&g_sdmaStubLog.count, 1U);
  if (idx < static_cast<unsigned int>(kSdmaStubLogCap)) {
    g_sdmaStubLog.calls[idx] = SdmaStubCall{op, dst, src, size, signal};
  }
  return true;
}

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
  if (sdmaStubRecord(kSdmaStubPut, dst, src, size, nullptr)) {
    return;
  }
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void putSignal(SdmaQueueDeviceHandle& handle, void* dst, void* src, size_t size,
                                          uint64_t* signal) {
  (void)handle;
  if (sdmaStubRecord(kSdmaStubPutSignal, dst, src, size, signal)) {
    return;
  }
  memcpyDevice(dst, src, size);
}

__device__ __forceinline__ void quiet(SdmaQueueDeviceHandle& handle) {
  (void)handle;
  sdmaStubRecord(kSdmaStubQuiet, nullptr, nullptr, 0, nullptr);
  atomicAdd(&g_sdmaStubQuietCount, 1ULL);
}

}  // namespace sdma_anvil
