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
#include <resource_guards.hh>

#ifdef _WIN32
#include <windows.h>

#include <algorithm>

namespace {
constexpr size_t ONE_MB = 1024 * 1024;
// Noise allowed when comparing free memory, e.g. from the OS budget query
constexpr size_t FREE_MEM_TOLERANCE = 64 * ONE_MB;
// Memory held for the whole test to create memory pressure, as a share of the idle free memory
constexpr size_t HELD_DEVICE_PERCENT = 40;
constexpr size_t HELD_HOST_PERCENT = 20;
constexpr size_t MAX_HELD_HOST_SIZE = 4096 * ONE_MB;
// Requests grow in steps of 1/REQUEST_STEPS of the idle free memory until one fails
constexpr size_t REQUEST_STEPS = 20;

constexpr size_t PercentOf(size_t value, size_t percent) { return value / 100 * percent; }

size_t FreeMemory() {
  size_t freeMem = 0, totalMem = 0;
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
  return freeMem;
}

bool IsApu() {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  return prop.integrated;
}

enum class AllocKind { Device, HostMalloc, HostRegister };

// One allocation that is allowed to fail. Frees whatever it allocated when destroyed, also when a
// REQUIRE throws. LinearAllocGuard can't be used here, it fails the test if the allocation fails.
class Allocation {
 public:
  explicit Allocation(AllocKind kind) : kind_{kind} {}
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;

  ~Allocation() {
    // No Catch macros, don't throw in the destructor
    if (ptr_ == nullptr) {
      return;
    }
    switch (kind_) {
      case AllocKind::Device:
        static_cast<void>(hipFree(ptr_));
        break;
      case AllocKind::HostMalloc:
        static_cast<void>(hipHostFree(ptr_));
        break;
      case AllocKind::HostRegister:
        if (registered_) {
          static_cast<void>(hipHostUnregister(ptr_));
        }
        VirtualFree(ptr_, 0, MEM_RELEASE);
        break;
    }
  }

  // Gets the pageable memory for hipHostRegister; false if Windows refuses it
  bool Prepare(size_t size) {
    if (kind_ != AllocKind::HostRegister) {
      return true;
    }
    ptr_ = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    return ptr_ != nullptr;
  }

  hipError_t Allocate(size_t size) {
    switch (kind_) {
      case AllocKind::Device:
        return hipMalloc(&ptr_, size);
      case AllocKind::HostMalloc:
        return hipHostMalloc(&ptr_, size, hipHostMallocDefault);
      case AllocKind::HostRegister: {
        const hipError_t result = hipHostRegister(ptr_, size, hipHostRegisterDefault);
        registered_ = result == hipSuccess;
        return result;
      }
    }
    return hipErrorInvalidValue;
  }

 private:
  const AllocKind kind_;
  void* ptr_ = nullptr;
  bool registered_ = false;
};

// Holds device and host memory, then makes larger and larger allocations of the given kind until
// one fails. Free memory must be unchanged by the failure and grow again after the release.
void CheckFailedAllocKeepsFreeMemory(AllocKind kind) {
  const size_t idle = FreeMemory();
  const size_t heldDevSize = PercentOf(idle, HELD_DEVICE_PERCENT);
  const size_t heldHostSize = std::min(PercentOf(idle, HELD_HOST_PERCENT), MAX_HELD_HOST_SIZE);
  INFO("Idle free memory : " << idle / ONE_MB << " MB, held device : " << heldDevSize / ONE_MB
                             << " MB, held host : " << heldHostSize / ONE_MB << " MB");

  // Guards free the held memory even if a REQUIRE below fails
  LinearAllocGuard<char> heldDev(LinearAllocs::hipMalloc, heldDevSize);
  LinearAllocGuard<char> heldHost(LinearAllocs::hipHostMalloc, heldHostSize);
  // Touch the held device memory so it is backed by physical memory
  HIP_CHECK(hipMemset(heldDev.ptr(), 0, heldDevSize));
  HIP_CHECK(hipDeviceSynchronize());

  bool failed = false;
  size_t beforeFailure = 0;
  for (size_t step = 1; step <= REQUEST_STEPS && !failed; ++step) {
    const size_t size = idle / REQUEST_STEPS * step;
    Allocation alloc(kind);
    // Windows refusing the pageable memory is not a runtime failure
    if (!alloc.Prepare(size)) {
      break;
    }
    const size_t before = FreeMemory();
    const hipError_t result = alloc.Allocate(size);
    if (result == hipSuccess) {
      continue;
    }
    // Clear the error, as an application recovering from out of memory would
    static_cast<void>(hipGetLastError());
    const size_t after = FreeMemory();
    failed = true;
    beforeFailure = before;
    INFO("Request : " << size / ONE_MB << " MB -> " << hipGetErrorName(result) << ", free before : "
                      << before / ONE_MB << " MB, after : " << after / ONE_MB << " MB");
    REQUIRE(after + FREE_MEM_TOLERANCE >= before);
  }

  heldDev = LinearAllocGuard<char>();
  heldHost = LinearAllocGuard<char>();
  HIP_CHECK(hipDeviceSynchronize());
  if (!failed) {
    HIP_SKIP_TEST("No allocation failed, the failure path was not exercised");
  }

  // Not compared with idle: freed host allocations can stay cached in the runtime. The held device
  // memory must come back, though.
  const size_t released = FreeMemory();
  INFO("Free memory after releasing the held memory : " << released / ONE_MB << " MB");
  REQUIRE(released + FREE_MEM_TOLERANCE >= beforeFailure + heldDevSize);
}
}  // namespace
#endif /* _WIN32 */

// One test case per allocation kind, so that ctest runs each in its own process: in one process,
// the memory held by the next kind can fault on Windows APUs (a separate runtime issue).

/**
 * Test Description
 * ------------------------
 * - A failed hipMalloc must not change the free memory reported by hipMemGetInfo.
 * - Holds device and host memory, then makes larger and larger requests until one fails. Free
 *   memory must be unchanged by the failure and grow by the held device memory when that is freed.
 *   Skipped if no request fails.
 * - Regression test: when memory could not be made resident, the runtime subtracted the size of the
 *   failed allocation from its free memory counters without having added it. The counter wrapped,
 *   hipMemGetInfo reported 0 for the rest of the process and, on APUs, later device allocations
 *   went to system memory, ending in hipErrorLaunchFailure.
 * Test source
 * ------------------------
 * - unit/memory/hipMallocFailure_test.cc
 * Test requirements
 * ------------------------
 * - Windows
 */
HIP_TEST_CASE(Unit_hipMalloc_Negative_FailureKeepsFreeMemory) {
#ifndef _WIN32
  HIP_SKIP_TEST("Regression test for the Windows runtime");
#else
  CheckFailedAllocKeepsFreeMemory(AllocKind::Device);
#endif /* _WIN32 */
}

/**
 * Test Description
 * ------------------------
 * - Same as Unit_hipMalloc_Negative_FailureKeepsFreeMemory, for hipHostMalloc.
 * Test source
 * ------------------------
 * - unit/memory/hipMallocFailure_test.cc
 * Test requirements
 * ------------------------
 * - Windows, APU (host allocation sizes follow device memory)
 */
HIP_TEST_CASE(Unit_hipHostMalloc_Negative_FailureKeepsFreeMemory) {
#ifndef _WIN32
  HIP_SKIP_TEST("Regression test for the Windows runtime");
#else
  if (!IsApu()) {
    HIP_SKIP_TEST("dGPU --- host allocation sizes are tuned for APUs");
  }
  CheckFailedAllocKeepsFreeMemory(AllocKind::HostMalloc);
#endif /* _WIN32 */
}

/**
 * Test Description
 * ------------------------
 * - Same as Unit_hipMalloc_Negative_FailureKeepsFreeMemory, for hipHostRegister.
 * Test source
 * ------------------------
 * - unit/memory/hipMallocFailure_test.cc
 * Test requirements
 * ------------------------
 * - Windows, APU (host allocation sizes follow device memory)
 */
HIP_TEST_CASE(Unit_hipHostRegister_Negative_FailureKeepsFreeMemory) {
#ifndef _WIN32
  HIP_SKIP_TEST("Regression test for the Windows runtime");
#else
  if (!IsApu()) {
    HIP_SKIP_TEST("dGPU --- host allocation sizes are tuned for APUs");
  }
  CheckFailedAllocKeepsFreeMemory(AllocKind::HostRegister);
#endif /* _WIN32 */
}
