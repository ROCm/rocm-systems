/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tracks device memory a unit allocates through the HIP VMM / malloc seams,
// on top of InstallHipVmmEmulator(). Clean() is true when everything was freed
// and no call was refused (bad free, over-release, out-of-range copy/memset).

#ifndef RCCL_TEST_HOST_HIPVMMLEDGER_H_
#define RCCL_TEST_HOST_HIPVMMLEDGER_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "fakes/hip_fakes.h"

// [dst, dst + n) lies inside one entry of a ptr -> size map.
inline bool RangeCovers(const std::map<void*, size_t>& allocs, const void* dst, size_t n) {
  auto it = allocs.upper_bound(const_cast<void*>(dst));
  if (it == allocs.begin()) return false;
  --it;
  const auto* base = static_cast<const char*>(it->first);
  const auto* p = static_cast<const char*>(dst);
  return p >= base && p + n <= base + it->second;
}

class HipVmmLedger {
 public:
  HipVmmLedger() = default;
  HipVmmLedger(const HipVmmLedger&) = delete;
  HipVmmLedger& operator=(const HipVmmLedger&) = delete;

  void Install() {
    InstallHipVmmEmulator();
    auto emulatedReserve = g_hipMemAddressReserve;
    auto emulatedFree = g_hipMemAddressFree;
    g_hipMemAddressReserve = [this, emulatedReserve](void** ptr, size_t size, size_t align, void* addr,
                                                     unsigned long long flags) {
      hipError_t err = emulatedReserve(ptr, size, align, addr, flags);
      if (err == hipSuccess) reserved[*ptr] = size;
      return err;
    };
    g_hipMemAddressFree = [this, emulatedFree](void* ptr, size_t size) {
      addressFrees.push_back(ptr);
      auto it = reserved.find(ptr);
      if (it == reserved.end() || it->second != size) return Reject("hipMemAddressFree of an unknown range");
      if (mappedHandle.count(ptr)) return Reject("hipMemAddressFree of a range still mapped");
      reserved.erase(it);
      return emulatedFree(ptr, size);
    };
    g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      auto it = reserved.find(ptr);
      if (it == reserved.end()) return Reject("hipMemGetAddressRange of an unknown range");
      *base = ptr;
      *size = it->second;
      return hipSuccess;
    };
    g_hipMemMap = [this](void* ptr, size_t, size_t, hipMemGenericAllocationHandle_t handle, unsigned long long) {
      mappedHandle[ptr] = handle;
      return hipSuccess;
    };
    g_hipMemUnmap = [this](void* ptr, size_t) {
      if (mappedHandle.erase(ptr) == 0) return Reject("hipMemUnmap of an unmapped range");
      return hipSuccess;
    };

    g_hipMemCreate = [this](hipMemGenericAllocationHandle_t* handle, size_t, const hipMemAllocationProp*,
                            unsigned long long) {
      *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(nextHandle_++);
      created_.insert(*handle);
      liveHandles[*handle] = 1;
      return hipSuccess;
    };
    g_hipMemRetainAllocationHandle = [this](hipMemGenericAllocationHandle_t* handle, void* ptr) {
      auto it = mappedHandle.find(ptr);
      if (it == mappedHandle.end()) return Reject("hipMemRetainAllocationHandle of an unmapped range");
      *handle = it->second;
      if (created_.count(*handle)) {
        if (!liveHandles.count(*handle)) return Reject("hipMemRetainAllocationHandle of a released handle");
        ++liveHandles[*handle];
      }
      return hipSuccess;
    };
    g_hipMemRelease = [this](hipMemGenericAllocationHandle_t handle) {
      released.push_back(handle);
      if (!created_.count(handle)) return hipSuccess;  // imported
      auto it = liveHandles.find(handle);
      if (it == liveHandles.end()) return Reject("hipMemRelease of a released handle");
      if (--it->second == 0) liveHandles.erase(it);
      return hipSuccess;
    };

    g_hipMalloc = [this](void** ptr, size_t size) { return Malloc(ptr, size); };
    g_hipExtMallocWithFlags = [this](void** ptr, size_t size, unsigned) { return Malloc(ptr, size); };
    g_hipFree = [this](void* ptr) {
      if (ptr == nullptr) return hipSuccess;
      if (liveBuffers.erase(ptr) == 0) return Reject("hipFree of an unknown buffer");
      std::free(ptr);
      return hipSuccess;
    };

    g_hipMemcpy = [this](void* dst, const void* src, size_t n, hipMemcpyKind) {
      if (!Covers(dst, n)) return Reject("hipMemcpy outside any live allocation");
      std::memcpy(dst, src, n);
      return hipSuccess;
    };
    g_hipMemset = [this](void* dst, int value, size_t n) {
      if (!Covers(dst, n)) return Reject("hipMemset outside any live allocation");
      std::memset(dst, value, n);
      return hipSuccess;
    };
  }

  bool Empty() const { return reserved.empty() && mappedHandle.empty() && liveHandles.empty() && liveBuffers.empty(); }
  bool Clean() const { return Empty() && rejected.empty(); }

  std::map<void*, size_t> reserved;  // live VA reservations -> size
  std::vector<void*> addressFrees;   // every hipMemAddressFree, in order
  std::map<void*, hipMemGenericAllocationHandle_t> mappedHandle;  // live mappings
  std::map<hipMemGenericAllocationHandle_t, int> liveHandles;     // created handle -> references
  std::vector<hipMemGenericAllocationHandle_t> released;          // every hipMemRelease, in order
  std::map<void*, size_t> liveBuffers;                            // hipMalloc / hipExtMallocWithFlags -> size
  std::vector<std::string> rejected;                              // calls refused, in order

 private:
  hipError_t Reject(const char* what) {
    rejected.emplace_back(what);
    return hipErrorInvalidValue;
  }

  hipError_t Malloc(void** ptr, size_t size) {
    *ptr = std::calloc(1, size == 0 ? 1 : size);
    if (*ptr == nullptr) return hipErrorOutOfMemory;
    liveBuffers[*ptr] = size;
    return hipSuccess;
  }

  bool Covers(const void* dst, size_t n) const {
    return RangeCovers(liveBuffers, dst, n) || RangeCovers(reserved, dst, n);
  }

  std::set<hipMemGenericAllocationHandle_t> created_;
  uintptr_t nextHandle_ = 0xC0000;
};

#endif  // RCCL_TEST_HOST_HIPVMMLEDGER_H_
