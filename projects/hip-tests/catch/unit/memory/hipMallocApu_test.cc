/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef _WIN32
#define NOMINMAX
#endif /* _WIN32 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#ifdef _WIN32
#include <windows.h>
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")

#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {
constexpr size_t kOneMB = 1024 * 1024;
// The runtime sizes the carve-out from its heaps, the OS from its segment; allow the difference
constexpr size_t kSizeTolerance = 512 * kOneMB;
// Device memory is allocated in steps of 1/kSteps of the reported total
constexpr size_t kSteps = 64;
constexpr size_t kThreads = 4;

struct OsMemory {
  size_t dedicated = 0;  // carve-out
  size_t shared = 0;     // host RAM the OS lets the GPU share
  size_t hostRam = 0;
};

bool IsApu() {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  return prop.integrated;
}

// What the OS reports for the adapter of device 0; false if it isn't found
bool QueryOsMemory(OsMemory* os) {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  LUID luid{};
  memcpy(&luid, prop.luid, sizeof(luid));

  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
    return false;
  }
  bool found = false;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT i = 0; !found && factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    if ((desc.AdapterLuid.LowPart == luid.LowPart) &&
        (desc.AdapterLuid.HighPart == luid.HighPart)) {
      os->dedicated = desc.DedicatedVideoMemory;
      os->shared = desc.SharedSystemMemory;
      found = true;
    }
    adapter->Release();
  }
  factory->Release();

  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  os->hostRam = status.ullTotalPhys;
  return found && (os->dedicated > 0);
}

size_t AvailableHostRam() {
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullAvailPhys;
}

void MemInfo(size_t* freeMem, size_t* totalMem) { HIP_CHECK(hipMemGetInfo(freeMem, totalMem)); }

// Device allocations freed when destroyed, also when a REQUIRE throws
class HeldAllocations {
 public:
  HeldAllocations() = default;
  HeldAllocations(const HeldAllocations&) = delete;
  HeldAllocations& operator=(const HeldAllocations&) = delete;
  ~HeldAllocations() {
    // No Catch macros, don't throw in the destructor
    for (void* ptr : ptrs_) {
      static_cast<void>(hipFree(ptr));
    }
  }

  // Allocates and touches size bytes; returns the hipMalloc result
  hipError_t Add(size_t size) {
    void* ptr = nullptr;
    const hipError_t result = hipMalloc(&ptr, size);
    if (result == hipSuccess) {
      ptrs_.push_back(ptr);
      if (hipMemset(ptr, 0x5a, size) != hipSuccess) {
        return hipErrorUnknown;
      }
    }
    return result;
  }

  // The device must still run work on every held allocation
  void CheckUsable(size_t size) const {
    for (void* ptr : ptrs_) {
      HIP_CHECK(hipMemset(ptr, 0xa5, size));
    }
    HIP_CHECK(hipDeviceSynchronize());
  }

  size_t Count() const { return ptrs_.size(); }

 private:
  std::vector<void*> ptrs_;
};

// Skips the test unless device 0 is a Windows APU with OS memory statistics
bool SetUpApu(OsMemory* os) {
  if (!IsApu()) {
    HIP_SKIP_TEST("dGPU --- tests APU device memory placement");
    return false;
  }
  if (!QueryOsMemory(os)) {
    HIP_SKIP_TEST("The OS doesn't report the adapter memory");
    return false;
  }
  char stack[8] = {};
  if ((GetEnvironmentVariableA("PAL_APU_STACK_APERTURE", stack, sizeof(stack)) > 0) &&
      (std::atoi(stack) != 0)) {
    HIP_SKIP_TEST("PAL_APU_STACK_APERTURE restores the stacked total");
    return false;
  }
  return true;
}
}  // namespace
#endif /* _WIN32 */

/**
 * Test Description
 * ------------------------
 * - The device memory of an APU must not exceed the share of all DRAM the OS lets a GPU use:
 *   the carve-out (dedicated), plus the part of the OS share (shared / host RAM) of all DRAM the
 *   carve-out doesn't cover.
 * - Regression test: the runtime reported the carve-out plus 75% of the aperture as device memory,
 *   though both are host DRAM. Applications filled it, the OS ran out of RAM and hung.
 * Test source
 * ------------------------
 * - unit/memory/hipMallocApu_test.cc
 * Test requirements
 * ------------------------
 * - Windows, APU
 */
HIP_TEST_CASE(Unit_hipMemGetInfo_Apu_TotalWithinOsShare) {
#ifndef _WIN32
  HIP_SKIP_TEST("Tests the Windows runtime");
#else
  OsMemory os;
  if (!SetUpApu(&os)) {
    return;
  }
  size_t freeMem = 0, totalMem = 0;
  MemInfo(&freeMem, &totalMem);
  // The OS share of all DRAM, carve-out included, less the carve-out
  const double share = static_cast<double>(os.shared) / os.hostRam;
  const double borrow = share * (os.dedicated + os.hostRam) - os.dedicated;
  const size_t bound = os.dedicated + static_cast<size_t>(borrow > 0 ? borrow : 0);
  INFO("Total : " << totalMem / kOneMB << " MB, dedicated : " << os.dedicated / kOneMB
                  << " MB, shared : " << os.shared / kOneMB << " MB, host RAM : "
                  << os.hostRam / kOneMB << " MB, bound : " << bound / kOneMB << " MB");
  REQUIRE(totalMem <= bound + kSizeTolerance);
#endif /* _WIN32 */
}

/**
 * Test Description
 * ------------------------
 * - Allocates the free device memory hipMemGetInfo reports, then more until hipMalloc fails.
 *   Everything reported free must be allocatable, the failure must be hipErrorOutOfMemory, the
 *   device must still run work on what it holds, and host RAM must not shrink by more than the
 *   device memory reported beyond the carve-out.
 * - Regression test: device memory past the carve-out went to host RAM without a limit, until
 *   host RAM ran out and the device faulted.
 * Test source
 * ------------------------
 * - unit/memory/hipMallocApu_test.cc
 * Test requirements
 * ------------------------
 * - Windows, APU
 */
HIP_TEST_CASE(Unit_hipMalloc_Apu_ReportedFreeIsAllocatable) {
#ifndef _WIN32
  HIP_SKIP_TEST("Tests the Windows runtime");
#else
  OsMemory os;
  if (!SetUpApu(&os)) {
    return;
  }
  size_t freeMem = 0, totalMem = 0;
  MemInfo(&freeMem, &totalMem);
  const size_t step = totalMem / kSteps;
  const size_t borrowed = (totalMem > os.dedicated) ? totalMem - os.dedicated : 0;
  const size_t availBefore = AvailableHostRam();
  INFO("Free : " << freeMem / kOneMB << " MB, total : " << totalMem / kOneMB
                 << " MB, step : " << step / kOneMB << " MB");

  HeldAllocations held;
  size_t allocated = 0;
  while (allocated + step + kSizeTolerance <= freeMem) {
    REQUIRE(held.Add(step) == hipSuccess);
    allocated += step;
  }
  hipError_t result = hipSuccess;
  for (size_t i = 0; (i < kSteps) && (result == hipSuccess); ++i) {
    result = held.Add(step);
  }
  INFO("Held : " << held.Count() * step / kOneMB << " MB, failure : " << hipGetErrorName(result));
  REQUIRE(result == hipErrorOutOfMemory);
  static_cast<void>(hipGetLastError());
  held.CheckUsable(step);

  const size_t availAfter = AvailableHostRam();
  const size_t shrink = (availBefore > availAfter) ? availBefore - availAfter : 0;
  INFO("Host RAM available before : " << availBefore / kOneMB
                                      << " MB, after : " << availAfter / kOneMB << " MB");
  REQUIRE(shrink <= borrowed + step + kSizeTolerance);
#endif /* _WIN32 */
}

/**
 * Test Description
 * ------------------------
 * - kThreads threads allocate device memory until hipMalloc fails. Every failure must be
 *   hipErrorOutOfMemory, together they must not hold more than the reported total plus one step per
 *   thread, and the device must still run work on what they hold.
 * Test source
 * ------------------------
 * - unit/memory/hipMallocApu_test.cc
 * Test requirements
 * ------------------------
 * - Windows, APU
 */
HIP_TEST_CASE(Unit_hipMalloc_Apu_ConcurrentAllocationsStayWithinTotal) {
#ifndef _WIN32
  HIP_SKIP_TEST("Tests the Windows runtime");
#else
  OsMemory os;
  if (!SetUpApu(&os)) {
    return;
  }
  size_t freeMem = 0, totalMem = 0;
  MemInfo(&freeMem, &totalMem);
  const size_t step = totalMem / kSteps;

  std::vector<HeldAllocations> held(kThreads);
  std::vector<hipError_t> failure(kThreads, hipSuccess);
  std::vector<std::thread> threads;
  for (size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      static_cast<void>(hipSetDevice(0));
      hipError_t result = hipSuccess;
      // Bounded, in case allocations never fail
      for (size_t i = 0; (i < 2 * kSteps) && (result == hipSuccess); ++i) {
        result = held[t].Add(step);
      }
      failure[t] = result;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  static_cast<void>(hipGetLastError());

  size_t count = 0;
  for (size_t t = 0; t < kThreads; ++t) {
    INFO("Thread " << t << " held " << held[t].Count()
                   << " steps, failure : " << hipGetErrorName(failure[t]));
    REQUIRE(failure[t] == hipErrorOutOfMemory);
    count += held[t].Count();
  }
  INFO("Held : " << count * step / kOneMB << " MB, total : " << totalMem / kOneMB << " MB");
  REQUIRE(count * step <= totalMem + kThreads * step);
  for (const auto& h : held) {
    h.CheckUsable(step);
  }
#endif /* _WIN32 */
}
