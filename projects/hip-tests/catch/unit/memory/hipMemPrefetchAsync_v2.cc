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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <utils.hh>
#include <numaif.h>
#include <numa.h>

/**
 * Kernel to fill value for each element in the given array
 */
static __global__ void fillDataKernel(int* arr, size_t size, int value) {
  size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < size; i += stride) {
    arr[i] = value;
  }
}

/**
 * Helper function to get the list of devices which supports
 * Managed memory
 */
static std::vector<int> getSupportedDevices() {
  const auto deviceCount = HipTest::getDeviceCount();
  std::vector<int> supportedDevices;
  supportedDevices.reserve(deviceCount + 1);
  for (int i = 0; i < deviceCount; ++i) {
    if (DeviceAttributesSupport(i, hipDeviceAttributeManagedMemory,
                                hipDeviceAttributeConcurrentManagedAccess)) {
      supportedDevices.push_back(i);
    }
  }
  return supportedDevices;
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following scenarios
 *  - 1) With Location type Device
 *  - 2) With Location type Host
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchAsync_v2.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemPrefetchAsync_v2_Device_Host") {
  auto supportedDevices = getSupportedDevices();
  if (supportedDevices.empty()) {
    HipTest::HIP_SKIP_TEST("Test need at least one device with managed memory support");
  }

  HIP_CHECK(hipSetDevice(supportedDevices[0]));

  const int N = 1024;
  const int Nbytes = N * sizeof(int);
  int value = 10;

  int* memPtr = nullptr;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&memPtr), Nbytes, hipMemAttachGlobal));
  REQUIRE(memPtr != nullptr);

  SECTION("With Device") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    for (int deviceId : supportedDevices) {
      hipMemLocation location;
      location.type = hipMemLocationTypeDevice;
      location.id = deviceId;

      HIP_CHECK(hipMemPrefetchAsync_v2(memPtr, Nbytes, location, 0, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      fillDataKernel<<<1, N / 2, 0, stream>>>(memPtr, N, value);
      HIP_CHECK(hipStreamSynchronize(stream));
    }
  }

#if HT_AMD  // In NVIDIA, getting compilation issues for Flags : SWDEV-551244
  SECTION("With Host") {
    hipMemLocation location;
    location.type = hipMemLocationTypeHost;

    HIP_CHECK(hipMemPrefetchAsync_v2(memPtr, Nbytes, location, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Fill data in Host
    for (int i = 0; i < N; i++) {
      memPtr[i] = value;
    }
  }
#endif

  // Validate data
  for (int i = 0; i < N; i++) {
    REQUIRE(memPtr[i] == value);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(memPtr));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following scenarios
 *  - 1) With Location type Host Numa
 *  - 2) With Location type Host Numa Current
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchAsync_v2.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
#if HT_AMD  // In NVIDIA, getting compilation issues for Flags : SWDEV-551244
TEST_CASE("Unit_hipMemPrefetchAsync_v2_HostNuma_HostNumaCurrent") {
  auto supportedDevices = getSupportedDevices();
  if (supportedDevices.empty()) {
    HipTest::HIP_SKIP_TEST("Test need at least one device with managed memory support");
  }

  HIP_CHECK(hipSetDevice(supportedDevices[0]));

  if (numa_available() < 0) {
    HipTest::HIP_SKIP_TEST("NUMA not available on this system");
  }

  int maxNode = numa_max_node();
  REQUIRE(maxNode >= 0);

  const int N = 1024;
  const int Nbytes = N * sizeof(int);
  int value = 10;

  int* memPtr = nullptr;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&memPtr), Nbytes, hipMemAttachGlobal));
  REQUIRE(memPtr != nullptr);

  SECTION("With Host NUMA") {
    hipMemLocation location;

    for (int node = 0; node <= maxNode; ++node) {
      location.type = hipMemLocationTypeHostNuma;
      location.id = node;

      HIP_CHECK(hipMemPrefetchAsync_v2(memPtr, Nbytes, location, 0, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      // Fill data in Host
      for (int i = 0; i < N; i++) {
        memPtr[i] = value;
      }

      // Validate data
      for (int i = 0; i < N; i++) {
        REQUIRE(memPtr[i] == value);
      }

#if 0  // To work this part, fix provided in SWDEV-548802 is required
      // verify placement
      void* page = memPtr;
      int status = -1;
      int ret = move_pages(0, 1, &page, nullptr, &status, 0);
      REQUIRE(ret == 0);
      REQUIRE(status == node);
#endif
    }
  }

  SECTION("With Host Numa Current") {
    hipMemLocation location;
    location.type = hipMemLocationTypeHostNumaCurrent;

    HIP_CHECK(hipMemPrefetchAsync_v2(memPtr, Nbytes, location, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Fill data in Host
    for (int i = 0; i < N; i++) {
      memPtr[i] = value;
    }

    // Validate data
    for (int i = 0; i < N; i++) {
      REQUIRE(memPtr[i] == value);
    }

    // determine current CPU’s NUMA node
    int cpu = sched_getcpu();
    int cur_node = numa_node_of_cpu(cpu);
    REQUIRE(cur_node >= 0);

    // verify that the page is on the current node
    void* page = memPtr;
    int status = -1;
    int ret = move_pages(0, 1, &page, nullptr, &status, 0);
    REQUIRE(ret == 0);
    REQUIRE(status == cur_node);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(memPtr));
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following Negative scenarios
 *  - 1) With dev_ptr as nullptr
 *  - 2) With count 0
 *  - 3) With count larger than actual size
 *  - 4) With invalid device
 *  - 5) With Invalid location type(Invalid, None)
 *  - 6) With Invalid location -1
 *  - 7) With invalid numa node
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchAsync_v2.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemPrefetchAsync_v2_Negative") {
  auto supportedDevices = getSupportedDevices();
  if (supportedDevices.empty()) {
    HipTest::HIP_SKIP_TEST("Test need at least one device with managed memory support");
  }

  HIP_CHECK(hipSetDevice(supportedDevices[0]));

  const int N = 16;
  const int Nbytes = N * sizeof(int);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* memPtr = nullptr;
  HIP_CHECK(hipMallocManaged(&memPtr, Nbytes, hipMemAttachGlobal));

  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;

  SECTION("With dev_ptr as nullptr") {
    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(nullptr, Nbytes, location, 0, stream),
                    hipErrorInvalidValue);
  }

  SECTION("With count 0") {
    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(memPtr, 0, location, 0, stream), hipErrorInvalidValue);
  }

  SECTION("With count larger than actual size") {
    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(memPtr, Nbytes + 10, location, 0, stream),
                    hipErrorInvalidValue);
  }

  SECTION("With invalid device") {
    hipMemLocation dstLocation;
    dstLocation.type = hipMemLocationTypeDevice;
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    dstLocation.id = deviceCount;
    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(memPtr, Nbytes, dstLocation, 0, stream),
                    hipErrorInvalidDevice);
  }

#if 0
  /**
   * Commenting below section due to,
   * In NVIDIA, getting compilation issues for Flags : SWDEV-551244
   * In AMD, not giving expected error code : SWDEV-551254
   */
  SECTION("With Invalid location type Invalid, None") {
    hipMemLocation location;
    location.type = GENERATE(hipMemLocationTypeInvalid,
                             hipMemLocationTypeNone);

    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(memPtr, Nbytes, location, 0, stream),
                    hipErrorInvalidValue);
  }
#endif

#if HT_NVIDIA  // In AMD not giving expected error code : SWDEV-551254
  SECTION("With Invalid location type -1") {
    hipMemLocation location;
    location.type = static_cast<hipMemLocationType>(-1);

    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(memPtr, Nbytes, location, 0, stream),
                    hipErrorInvalidValue);
  }
#endif

#if 0
  /**
   * Commenting below section due to,
   * In NVIDIA, getting compilation issues for HostNuma Flag : SWDEV-551244
   * In AMD, not giving expected error code : SWDEV-551254
   */
  SECTION("With invalid numa node") {
    if (numa_available() >= 0) {
      int maxNode = numa_max_node();
      hipMemLocation dstLocation;
      dstLocation.type = hipMemLocationTypeHostNuma;
      dstLocation.id = maxNode+1;
      HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(memPtr, Nbytes, dstLocation, 0, stream),
                      hipErrorInvalidValue);
    }
  }
#endif

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(memPtr));
}
