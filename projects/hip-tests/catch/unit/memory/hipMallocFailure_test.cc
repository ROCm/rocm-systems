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

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
constexpr size_t ONE_MB = 1024 * 1024;
// Noise allowed when comparing free memory, e.g. from the OS budget query
constexpr size_t FREE_MEM_TOLERANCE = 64 * ONE_MB;

size_t FreeMemory() {
  size_t freeMem = 0, totalMem = 0;
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
  return freeMem;
}

enum class AllocKind { Device, HostMalloc, HostRegister };

#ifdef _WIN32
// Holds device and host memory, makes one allocation of the given kind fail under that pressure
// and checks that free memory is unchanged by the failure and back to idle after the release.
void CheckFailedAllocKeepsFreeMemory(AllocKind kind) {
  const size_t idle = FreeMemory();
  // Ratios reproduce the failure on an APU with 8 GB dedicated memory (17 GB reported): the
  // request passes the runtime's own free memory check, but WDDM can't make it resident
  const size_t heldDevSize = idle / 100 * 41;
  const size_t heldHostSize = std::min(idle / 100 * 23, static_cast<size_t>(4096) * ONE_MB);
  const size_t size = idle / 100 * 54;
  INFO("Idle free memory : " << idle / ONE_MB << " MB, held device : " << heldDevSize / ONE_MB
                             << " MB, held host : " << heldHostSize / ONE_MB
                             << " MB, request : " << size / ONE_MB << " MB");

  void* heldDev = nullptr;
  void* heldHost = nullptr;
  HIP_CHECK(hipMalloc(&heldDev, heldDevSize));
  HIP_CHECK(hipMemset(heldDev, 0, heldDevSize));
  HIP_CHECK(hipHostMalloc(&heldHost, heldHostSize, hipHostMallocDefault));
  HIP_CHECK(hipDeviceSynchronize());
  const size_t before = FreeMemory();

  void* ptr = nullptr;
  void* sysMem = nullptr;
  hipError_t result = hipSuccess;
  switch (kind) {
    case AllocKind::Device:
      result = hipMalloc(&ptr, size);
      break;
    case AllocKind::HostMalloc:
      result = hipHostMalloc(&ptr, size, hipHostMallocDefault);
      break;
    case AllocKind::HostRegister:
      sysMem = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
      REQUIRE(sysMem != nullptr);
      memset(sysMem, 1, size);
      result = hipHostRegister(sysMem, size, hipHostRegisterDefault);
      break;
  }
  // Clear the error, as an application recovering from out of memory would
  static_cast<void>(hipGetLastError());
  const size_t after = FreeMemory();
  INFO("Allocation result : " << hipGetErrorName(result) << ", free before : " << before / ONE_MB
                              << " MB, after : " << after / ONE_MB << " MB");

  if (result == hipSuccess) {
    WARN("Allocation did not fail; the failure path was not exercised");
    if (kind == AllocKind::Device) HIP_CHECK(hipFree(ptr));
    if (kind == AllocKind::HostMalloc) HIP_CHECK(hipHostFree(ptr));
    if (kind == AllocKind::HostRegister) HIP_CHECK(hipHostUnregister(sysMem));
  } else {
    REQUIRE(after + FREE_MEM_TOLERANCE >= before);
  }

  HIP_CHECK(hipFree(heldDev));
  HIP_CHECK(hipHostFree(heldHost));
  HIP_CHECK(hipDeviceSynchronize());
  if (sysMem != nullptr) VirtualFree(sysMem, 0, MEM_RELEASE);
  const size_t released = FreeMemory();
  INFO("Free memory after releasing everything : " << released / ONE_MB << " MB");
  REQUIRE(released + FREE_MEM_TOLERANCE >= idle);
}
#endif /* _WIN32 */
}  // namespace

/**
 * Test Description
 * ------------------------
 * - A failed allocation must not change the free memory reported by hipMemGetInfo.
 * - Holds device and host memory, then makes a hipMalloc, hipHostMalloc or hipHostRegister fail
 *   because the memory can't be made resident. Free memory must be unchanged by the failure and
 *   return to its idle value after everything is released.
 * - Regression test: the runtime subtracted the size of a failed allocation from its free memory
 *   counters without having added it; the counter wrapped, hipMemGetInfo reported 0 for the rest
 *   of the process and, on APUs, later device allocations were placed in system memory.
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
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  SECTION("hipMalloc") { CheckFailedAllocKeepsFreeMemory(AllocKind::Device); }
  // The host allocations are sized from device memory; only an APU has host memory to match
  SECTION("hipHostMalloc") {
    if (!prop.integrated) HIP_SKIP_TEST("dGPU --- host allocation sizes are tuned for APUs");
    CheckFailedAllocKeepsFreeMemory(AllocKind::HostMalloc);
  }
  SECTION("hipHostRegister") {
    if (!prop.integrated) HIP_SKIP_TEST("dGPU --- host allocation sizes are tuned for APUs");
    CheckFailedAllocKeepsFreeMemory(AllocKind::HostRegister);
  }
#endif /* _WIN32 */
}
