/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifdef _WIN32
#define NOMINMAX
#endif /* _WIN32 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#ifdef __linux__
#include <sys/sysinfo.h>
#else
#include <windows.h>
#include <sysinfoapi.h>
#endif

static constexpr size_t ONE_MB = 1024 * 1024;

TEST_CASE("Unit_hipMalloc_Positive_Basic") {
  constexpr size_t page_size = 4096;
  void* ptr = nullptr;
  const auto alloc_size =
      GENERATE_COPY(10, page_size / 2, page_size, page_size * 3 / 2, page_size * 2);
  HIP_CHECK(hipMalloc(&ptr, alloc_size));
  CHECK(ptr != nullptr);
  CHECK(reinterpret_cast<intptr_t>(ptr) % 256 == 0);
  HIP_CHECK(hipFree(ptr));
}

TEST_CASE("Unit_hipMalloc_Positive_Zero_Size") {
  void* ptr = reinterpret_cast<void*>(0x1);
  HIP_CHECK(hipMalloc(&ptr, 0));
  REQUIRE(ptr == nullptr);
}

TEST_CASE("Unit_hipMalloc_Positive_Alignment") {
  void *ptr1 = nullptr, *ptr2 = nullptr;
  HIP_CHECK(hipMalloc(&ptr1, 1));
  HIP_CHECK(hipMalloc(&ptr2, 10));
  CHECK(reinterpret_cast<intptr_t>(ptr1) % 256 == 0);
  CHECK(reinterpret_cast<intptr_t>(ptr2) % 256 == 0);
  HIP_CHECK(hipFree(ptr1));
  HIP_CHECK(hipFree(ptr2));
}

TEST_CASE("Unit_hipMalloc_Negative_Parameters") {
  SECTION("ptr == nullptr") { HIP_CHECK_ERROR(hipMalloc(nullptr, 4096), hipErrorInvalidValue); }
  SECTION("size == max size_t") {
    void* ptr;
    HIP_CHECK_ERROR(hipMalloc(&ptr, std::numeric_limits<size_t>::max()), hipErrorOutOfMemory);
  }
}

// Commenting this due to defect SWDEV-501675, used in below commented tests
#if 0
/**
/**
 * Local Function to Get total RAM size in bytes
 */
static inline size_t getTotalRAM() {
#ifdef __linux__
  struct sysinfo info {};
  sysinfo(&info);
  return info.totalram;
#else
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  GlobalMemoryStatusEx(&statex);
  return (statex.ullTotalPhys);
#endif
}

/**
 * Local Function to Get Available RAM size in bytes
 */
static inline size_t getAvailableRAM() {
#ifdef __linux__
  struct sysinfo info {};
  sysinfo(&info);
  return info.freeram;
#else
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  GlobalMemoryStatusEx(&statex);
  return (statex.ullAvailPhys);
#endif
}
#endif

/**
 * In addKernel function, all elements of the array a increased by 1
 */
static __global__ void addKernel(char* a, size_t size) {
  for (int i = 0; i < size; i++) {
    a[i] += 1;
  }
}

/**
 * Local Function to perform some operations on device memory and validate it
 */
static void performOperations(char* devMem, size_t size) {
  /**
   * Validating 1MB or less than that, since the most of the scenarios uses
   * this function allocates memory in GB's and execution time will be more
   */
  const size_t sizeToCheck = (size < ONE_MB) ? size : ONE_MB;
  constexpr char value = 'A';

  HIP_CHECK(hipMemset(devMem, value, sizeToCheck));
  addKernel<<<1, 1>>>(devMem, sizeToCheck);

  char* arrToCheck = new char[sizeToCheck];
  memset(arrToCheck, '0', sizeToCheck);

  HIP_CHECK(hipMemcpy(arrToCheck, devMem, sizeToCheck, hipMemcpyDeviceToHost));

  for (int i = 0; i < sizeToCheck; i++) {
    INFO("At index : " << i << " Got value : " << arrToCheck[i] << " Expected value : B ");
    REQUIRE(arrToCheck[i] == 'B');
  }
  delete[] arrToCheck;
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the following scenario:-
 * - 1) Get the Available VRAM size
 * - 2) Allocate 90% of that using hipMalloc
 * - 3) Perform operations on device memory
 * Test source
 * ------------------------
 * - unit/memory/hipMalloc.cc
 */
TEST_CASE("Unit_hipMalloc_Allocate90PercentOfDeviceMemory") {
  char* devMem = nullptr;
  size_t freeVRAM = 0, totalVRAM = 0;
  HIP_CHECK(hipMemGetInfo(&freeVRAM, &totalVRAM));
  INFO("Available device memory : " << freeVRAM);

  /**
   * Allocating 90% of available VRAM.
   * Avoided allocating total available VRAM just for stability
   * and to keep some buffer memory.
   */
  size_t size = freeVRAM * 0.9;
  INFO("Size going to allocate : " << size);

  HIP_CHECK(hipMalloc(&devMem, size));
  REQUIRE(devMem != nullptr);

  performOperations(devMem, size);
  HIP_CHECK(hipFree(devMem));
}

// Commenting this due to defect SWDEV-501675
#if 0
/**
 * Test Description
 * ------------------------
 * - This test case tests the following scenario:-
 * - 1) Get the Available VRAM size
 * - 2) Allocate 110% of that using hipMalloc
 * - 3) Check the behaviour in linux and windows
 * - 4) In linux it should give hipErrorOutOfMemory
 * - 5) In windows it should success by using the Shared GPU memory
 * - 6) Perform operations on device memory
 * Test source
 * ------------------------
 * - unit/memory/hipMalloc.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipMalloc_Allocate110PercentOfDeviceMemory") {
  char *devMem = nullptr;
  size_t freeVRAM = 0, totalVRAM = 0;
  HIP_CHECK(hipMemGetInfo(&freeVRAM, &totalVRAM));
  INFO("Available Free device memory : " << freeVRAM);

  // Allocating 110% of available VRAM
  size_t size = freeVRAM * 1.1;
  INFO("Size Planning to allocate : " << size);

  // Determine amount of memory which is needed from RAM
  size_t requiredMemFromRAM = size - freeVRAM;
  INFO("Required Memory From RAM : " << requiredMemFromRAM);

#ifdef __linux__
  INFO("LINUX");
  HIP_CHECK_ERROR(hipMalloc(&devMem, size), hipErrorOutOfMemory);
#else
  INFO("WINDOWS");
  size_t totalRAM = getTotalRAM();
  size_t availableRAM = getAvailableRAM();
  INFO("Total RAM : " << totalRAM);
  INFO("Available RAM : " << availableRAM);

  /**
   * Validating size of Required Memory From RAM,
   *  1) Since we can use half of RAM as shared GPU memory,
   *     Required Memory From RAM should be less than that.
   *  2) And Required Memory From RAM should be less than Available RAM.
   * If Required Memory From RAM is crossing the threshold,
   * allocating 90% available VRAM
   */
  if ((requiredMemFromRAM < (totalRAM / 2)) &&
      (requiredMemFromRAM < availableRAM)) {
    INFO("Required Memory From RAM is Available");
  } else {
    INFO("Required Memory From RAM is Not available, "
         "allocating 90% of available VRAM");
    // Avoided using total available VRAM for stability, to keep buffer memory
    size = freeVRAM * 0.9;
  }
  INFO("Size going to allocate : " << size);

  HIP_CHECK(hipMalloc(&devMem, size));
  REQUIRE(devMem != nullptr);
  performOperations(devMem, size);
  HIP_CHECK(hipFree(devMem));
#endif
}
#endif

// Commenting this due to defect SWDEV-501675
#if 0
/**
 * Test Description
 * ------------------------
 * - This test case tests the following scenario:-
 * - 1) Get the Available VRAM size, Total RAM size and Available RAM size
 * - 2) In case if Available RAM size is greater than half of Total RAM,
 * -    allocate 90% of half of Total RAM
 * - 3) In case if Available RAM size is less than half of Total RAM,
 * -    allocate 50% of half of Available RAM
 * - 4) Check the behaviour in linux and windows,
 * -    In linux it should give hipErrorOutOfMemory and
 * -    in windows it should success by using the Shared GPU memory
 * - 5) Perform operations on device memory
 * Test source
 * ------------------------
 * - unit/memory/hipMalloc.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipMalloc_AllocateAvailableVRAMAndPossibleRAM") {
  char *devMem = nullptr;
  size_t freeVRAM = 0, totalVRAM = 0;
  HIP_CHECK(hipMemGetInfo(&freeVRAM, &totalVRAM));
  INFO("Available Free device memory : " << freeVRAM);

  size_t totalRAM = getTotalRAM();
  size_t availableRAM = getAvailableRAM();
  INFO("Total RAM : " << totalRAM);
  INFO("Available RAM : " << availableRAM);

  // Determine usable Shared GPU Memory by GPU From RAM which is half of RAM
  size_t usableSharedGPUMemFromRam = totalRAM / 2;

  if (availableRAM > usableSharedGPUMemFromRam) {
    /*
     * If Available RAM is greater than the usable Shared GPU Memory
     * Allocating 90% of Usable Shared GPU Memory
     * Not allocating 100% of Usable Shared GPU Memory just to keep
     * buffer memory and for stability
     */
    usableSharedGPUMemFromRam = usableSharedGPUMemFromRam * 0.9;
  } else {
    /*
     * If Available RAM is less than the usable Shared GPU Memory
     * Allocating 50% of RAM only just to keep buffer memory and for stability
     */
    usableSharedGPUMemFromRam = availableRAM * 0.5;
  }
  INFO("Usable Shared GPU Memory from RAM : " << usableSharedGPUMemFromRam);

  size_t size = freeVRAM + usableSharedGPUMemFromRam;
  INFO("Size going to allocate : " << size);

#ifdef __linux__
  INFO("LINUX");
  HIP_CHECK_ERROR(hipMalloc(&devMem, size), hipErrorOutOfMemory);
#else
  INFO("WINDOWS");
  HIP_CHECK(hipMalloc(&devMem, size));
  REQUIRE(devMem != nullptr);
  performOperations(devMem, size);
  HIP_CHECK(hipFree(devMem));
#endif
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the following scenario:-
 * - 1) Get the Total RAM size
 * - 2) Allocate 1MB more than that using hipMalloc
 * - 3) Check the behaviour in linux and windows.
 * -    In linux, windows it should give hipErrorOutOfMemory
 * Test source
 * ------------------------
 * - unit/memory/hipMalloc.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipMalloc_AllocateMoreThanTotalRAM") {
  char *devMem = nullptr;

  size_t totalRAM = getTotalRAM();
  INFO("Total RAM : " << totalRAM);

  size_t size = totalRAM + ONE_MB;
  INFO("Size going to allocate : " << size);

  HIP_CHECK_ERROR(hipMalloc(&devMem, size), hipErrorOutOfMemory);
}

/**
 * Test Description
 * ------------------------
 * - This test case tests the following scenario:-
 * - 1) Get the Total VRAM size
 * - 2) Allocate 1MB more than total VRAM size using hipMalloc
 * - 3) Check the behaviour in linux and windows
 * - 4) In linux it should give hipErrorOutOfMemory
 * - 5) In windows it should success using shared GPU memory
 * - 6) Perform operations on device memory
 * Test source
 * ------------------------
 * - unit/memory/hipMalloc.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipMalloc_AllocateMoreThanTotalVRAM") {
  char *devMem = nullptr;

  size_t freeVRAM = 0, totalVRAM = 0;
  HIP_CHECK(hipMemGetInfo(&freeVRAM, &totalVRAM));
  INFO("Available total device memory : " << totalVRAM);

  size_t size = totalVRAM + ONE_MB;
  INFO("Size Planning to allocate : " << size);

#ifdef __linux__
  INFO("LINUX");
  HIP_CHECK_ERROR(hipMalloc(&devMem, size), hipErrorOutOfMemory);
#else
  INFO("WINDOWS");

  size_t requiredMemFromRAM = size - freeVRAM;
  INFO("Required Memory From RAM : " << requiredMemFromRAM);

  size_t totalRAM = getTotalRAM();
  size_t availableRAM = getAvailableRAM();
  INFO("Total RAM : " << totalRAM);
  INFO("Available RAM : " << availableRAM);

  /**
   * Validating size of Required Memory From RAM,
   * 1) Since we can use half of RAM as shared GPU memory,
   *    Required Memory From RAM should be less than that.
   * 2) And Required Memory From RAM should be less than Available RAM.
   * If Required Memory From RAM is crossing the threshold,
   * allocating 90% available VRAM
   */
  if ((requiredMemFromRAM < (totalRAM / 2)) &&
      (requiredMemFromRAM < availableRAM)) {
    INFO("Required Memory From RAM is Available");
  } else {
    INFO("Required Memory From RAM is Not available, "
         "allocating 90% of available VRAM");
    // Avoided using total available VRAM for stability, to keep buffer memory
    size = freeVRAM * 0.9;
  }
  INFO("Size going to allocate : " << size);

  HIP_CHECK(hipMalloc(&devMem, size));
  REQUIRE(devMem != nullptr);
  performOperations(devMem, size);
  HIP_CHECK(hipFree(devMem));
#endif
}
#endif
