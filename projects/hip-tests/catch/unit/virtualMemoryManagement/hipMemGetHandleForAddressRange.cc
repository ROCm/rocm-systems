/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>

#if __linx__
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "hip_vmm_common.hh"
#include "hipMemGetHandleForAddressRange_common.hh"

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following Negative scenarios,
 *  - 1) With device pointer as nullptr
 *  - 2) With size as 0
 *  - 3) With Invalid hipMemRangeHandleType
 *  - 4) With Invalid Flags
 *  - 5) With device pointer as already freed memory
 *  - 6) With Host Memory
 *  - 7) With Unmapped Virtual memory
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_Negative") {
  int handle = -1;
  int* dptr = nullptr;
  constexpr int size = 10;
  constexpr int sizeBytes = size * sizeof(int);
  HIP_CHECK(hipMalloc(&dptr, sizeBytes));

  #if HT_AMD
    hipDeviceptr_t nptr = nullptr;
  #else
    hipDeviceptr_t nptr = 0;
  #endif

  SECTION("nullptr") {
    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle, nptr, sizeBytes,
                                                   hipMemRangeHandleTypeDmaBufFd, 0),
                    hipErrorInvalidValue);
  }

  SECTION("size 0") {
    HIP_CHECK_ERROR(
        hipMemGetHandleForAddressRange(&handle, reinterpret_cast<hipDeviceptr_t>(dptr),
                                       0, hipMemRangeHandleTypeDmaBufFd, 0),
        hipErrorInvalidValue);
  }

  SECTION("Invalid Handle type") {
    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle,
                    reinterpret_cast<hipDeviceptr_t>(dptr), sizeBytes,
                    static_cast<hipMemRangeHandleType>(-1), 0),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Flags") {
    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle,
                    reinterpret_cast<hipDeviceptr_t>(dptr), sizeBytes,
                    hipMemRangeHandleTypeDmaBufFd, 0xFF),
                    hipErrorInvalidValue);
  }

  SECTION("With Freed Memory") {
    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, sizeBytes));
    HIP_CHECK(hipFree(devMem));

    HIP_CHECK_ERROR(hipMemGetHandleForAddressRange(&handle, reinterpret_cast<hipDeviceptr_t>(devMem), sizeBytes,
                                                   hipMemRangeHandleTypeDmaBufFd, 0),
                    hipErrorInvalidValue);
  }

  SECTION("With Host memory") {
    int* hptr = new int[size];
    HIP_CHECK_ERROR(
        hipMemGetHandleForAddressRange(&handle, reinterpret_cast<hipDeviceptr_t>(hptr), sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0),
        hipErrorInvalidValue);
    delete[] hptr;
  }

  SECTION("With Unmapped Virtual Memory") {
    hipDevice_t device;
    constexpr int kDeviceId = 0;
    HIP_CHECK(hipDeviceGet(&device, kDeviceId));
    checkVMMSupported(device);

    size_t granularity = GetGranularity(kDeviceId);
    assert(granularity > 0);

    size_t size_mem = ((granularity + sizeBytes - 1) / granularity) * granularity;
    hipDeviceptr_t ptrA;
    HIP_CHECK(hipMemAddressReserve(reinterpret_cast<void**>(&ptrA), size_mem, granularity, 0, 0));

    REQUIRE(reinterpret_cast<void*>(ptrA) != nullptr);

    HIP_CHECK_ERROR(
        hipMemGetHandleForAddressRange(&handle, ptrA, size_mem, hipMemRangeHandleTypeDmaBufFd, 0),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(dptr));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario,
 *  - 1) Create the device memory
 *  - 2) Get handle from hipMemGetHandleForAddressRange
 *  - 3) Validate the handle by doing Read and Write operations
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_DeviceMemory") {
  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);
  CTX_CREATE();

  void* srcDevMem = createDeviceMemoryAndFillData(size);
  REQUIRE(srcDevMem != nullptr);

  int handle = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(&handle, reinterpret_cast<hipDeviceptr_t>(srcDevMem), sizeBytes,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  hipDevice_t device;
  constexpr int kDeviceId = 0;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);
  REQUIRE(validateHandle(handle, size) == true);

  HIP_CHECK(hipFree(srcDevMem));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Create the Virtual memory
 *  - 2) Get handle from hipMemGetHandleForAddressRange
 *  - 3) Validate the handle by doing Read and Write operations
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_VM") {
  CTX_CREATE();
  hipDevice_t device;
  constexpr int kDeviceId = 0;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);

  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);

  hipDeviceptr_t ptrA;
  int reservedAddrSize;
  ptrA = createVirtualMemoryAndFillData(size, &reservedAddrSize);
  REQUIRE(reinterpret_cast<void*>(ptrA) != nullptr);

  int handle = -1;
  HIP_CHECK(
      hipMemGetHandleForAddressRange(&handle, ptrA, sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  REQUIRE(validateHandle(handle, size) == true);

  HIP_CHECK(hipMemUnmap(reinterpret_cast<void*>(ptrA), reservedAddrSize));
  HIP_CHECK(hipMemAddressFree(reinterpret_cast<void*>(ptrA), reservedAddrSize));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Create the Device memory in a device
 *  - 2) Get handle from hipMemGetHandleForAddressRange in same device
 *  - 3) Using the handle do Read and Write operations in another device
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_DeviceMemory_InAnotherDevice",
          "[multigpu]") {
  CTX_CREATE();
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  constexpr int srcDeviceId = 0;
  constexpr int dstDeviceId = 1;

  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);

  HIP_CHECK(hipSetDevice(srcDeviceId));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, srcDeviceId));
  checkVMMSupported(device);

  void* srcDevMem = nullptr;
  srcDevMem = createDeviceMemoryAndFillData(size);
  REQUIRE(srcDevMem != nullptr);

  int handle = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(&handle, reinterpret_cast<hipDeviceptr_t>(srcDevMem), sizeBytes,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  HIP_CHECK(hipSetDevice(dstDeviceId));

  HIP_CHECK(hipDeviceGet(&device, dstDeviceId));
  checkVMMSupported(device);
  REQUIRE(validateHandle(handle, size, dstDeviceId) == true);

  HIP_CHECK(hipFree(srcDevMem));
  HIP_CHECK(hipDeviceReset());
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario,
 *  - 1) Create the Virtual memory in a device
 *  - 2) Get handle from hipMemGetHandleForAddressRange in same device
 *  - 3) Using the handle do Read and Write operations in another device
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_VM_InAnotherDevice",
          "[multigpu]") {
  CTX_CREATE();
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  constexpr int srcDeviceId = 0;
  constexpr int dstDeviceId = 1;

  HIP_CHECK(hipSetDevice(srcDeviceId));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, srcDeviceId));
  checkVMMSupported(device);

  constexpr int kNumElemsSize = 1024;
  constexpr int size = 1024;
  constexpr int sizeBytes = kNumElemsSize * sizeof(int);

  hipDeviceptr_t ptrA;
  int reservedAddrSize;
  ptrA = createVirtualMemoryAndFillData(size, &reservedAddrSize);
  REQUIRE(reinterpret_cast<void*>(ptrA) != nullptr);

  int handle = 0;
  HIP_CHECK(
      hipMemGetHandleForAddressRange(&handle, ptrA, sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  HIP_CHECK(hipSetDevice(dstDeviceId));

  HIP_CHECK(hipDeviceGet(&device, dstDeviceId));
  checkVMMSupported(device);

  REQUIRE(validateHandle(handle, size, dstDeviceId) == true);

  HIP_CHECK(hipMemUnmap(reinterpret_cast<void*>(ptrA), reservedAddrSize));
  HIP_CHECK(hipMemAddressFree(reinterpret_cast<void*>(ptrA), reservedAddrSize));

  HIP_CHECK(hipDeviceReset());
  CTX_DESTROY();
}

/*
 * Helper function to create the Device Memory and get handle
 */
void launchForDevMem() {
  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);
  void* srcDevMem = createDeviceMemoryAndFillData(size);

  int handle = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(&handle, reinterpret_cast<hipDeviceptr_t>(srcDevMem), sizeBytes,
                                           hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);
  HIP_CHECK(hipFree(srcDevMem));
}

/*
 * Helper function to create the Virtual Memory and get handle
 */
void launchForVM() {
  constexpr int size = 1024;
  constexpr int sizeBytes = size * sizeof(int);

  hipDeviceptr_t ptrA;
  int reservedAddrSize;
  ptrA = createVirtualMemoryAndFillData(size, &reservedAddrSize);
  REQUIRE(reinterpret_cast<void*>(ptrA) != nullptr);

  int handle = -1;
  HIP_CHECK(
      hipMemGetHandleForAddressRange(&handle, ptrA, sizeBytes, hipMemRangeHandleTypeDmaBufFd, 0));
  REQUIRE(handle > 0);

  HIP_CHECK(hipMemUnmap(reinterpret_cast<void*>(ptrA), reservedAddrSize));
  HIP_CHECK(hipMemAddressFree(reinterpret_cast<void*>(ptrA), reservedAddrSize));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following behaviour of
 *  - hipMemGetHandleForAddressRange in the Multi threaded
 *  - scenario with both Device Memory and the Virtual Memory
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_MultipleThreads") {
  hipDevice_t device;
  constexpr int kDeviceId = 0;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);

  const unsigned int threadsSupported = std::thread::hardware_concurrency();
  const int numberOfThreads = (threadsSupported >= 10) ? 10 : threadsSupported;

  std::vector<std::thread> threads;

  SECTION("For DeviceMemory") {
    for (int t = 0; t < numberOfThreads; t++) {
      threads.push_back(std::thread(launchForDevMem));
    }
  }

  SECTION("For VM") {
    for (int t = 0; t < numberOfThreads; t++) {
      threads.push_back(std::thread(launchForVM));
    }
  }

  for (int t = 0; (t < numberOfThreads) && (t < threads.size()); t++) {
    threads[t].join();
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Create the device memory of 5 integers
 *  - 2) Get handle from hipMemGetHandleForAddressRange for all offsets (0-4)
 *  - 3) Check the handle is valid or not
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRange.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
TEST_CASE("Unit_hipMemGetHandleForAddressRange_DifferentOffsets") {
  int handle;
  int size = 5;
  int sizeBytes = size * sizeof(int);
  int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, sizeBytes));
  REQUIRE(dptr != nullptr);

  for (int i = 0; i < size; i++) {
    handle = -1;
    HIP_CHECK(hipMemGetHandleForAddressRange(&handle, reinterpret_cast<hipDeviceptr_t>(dptr + i), sizeBytes - (i * sizeof(int)),
                                             hipMemRangeHandleTypeDmaBufFd, 0));
    REQUIRE(handle > 0);
  }

  HIP_CHECK(hipFree(dptr));
}
