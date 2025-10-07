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
#if __linux__
#include <numa.h>
#include <numaif.h>
#endif

/**
 * Kernel to fill value for each element in the given array
 */
static __global__ void fillDataKernel(int *arr, int value) {
  arr[threadIdx.x] = value;
}

/**
 * Kernel to copy data from source array to destination array
 */
static __global__ void copyDataKernel(int *dstArr, int *srcArr) {
  dstArr[threadIdx.x] = srcArr[threadIdx.x];
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
 *  - unit/memory/hipMemAdvise_v2.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemAdvise_v2_Device_Host") {
  auto supportedDevices = getSupportedDevices();
  if (supportedDevices.empty()) {
    HipTest::HIP_SKIP_TEST(
        "Test need at least one device with managed memory support");
  }

  HIP_CHECK(hipSetDevice(supportedDevices[0]));

  constexpr int N = 1024;
  constexpr int Nbytes = N * sizeof(int);
  constexpr int value = 10;
  int *memPtr = nullptr;

  HIP_CHECK(hipMallocManaged(reinterpret_cast<void **>(&memPtr), Nbytes,
                             hipMemAttachGlobal));
  REQUIRE(memPtr != nullptr);

  SECTION("With Device") {
    std::fill_n(memPtr, N, value);

    for (int deviceId : supportedDevices) {
      HIP_CHECK(hipSetDevice(deviceId));

      hipMemLocation location;
      location.type = hipMemLocationTypeDevice;
      location.id = deviceId;

      HIP_CHECK(
          hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));
      int *devArr = nullptr;
      HIP_CHECK(hipMalloc(&devArr, Nbytes));
      REQUIRE(devArr != nullptr);

      copyDataKernel<<<1, N>>>(devArr, memPtr);

      int hostArr[N];
      HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
      HIP_CHECK(hipDeviceSynchronize());

      for (int i = 0; i < N; i++) {
        INFO("For Device " << deviceId << " At index " << i
                           << " Expected value = " << value
                           << " Got value = " << hostArr[i]);
        REQUIRE(hostArr[i] == value);
      }

      HIP_CHECK(hipFree(devArr));
    }
  }

  SECTION("With Host") {
    fillDataKernel<<<1, N>>>(memPtr, value);
    HIP_CHECK(hipDeviceSynchronize());

    hipMemLocation location;
    location.type = hipMemLocationTypeHost;

    HIP_CHECK(
        hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));

    for (int i = 0; i < N; i++) {
      INFO("At index " << i << " Expected value = " << value
                       << " Got value = " << memPtr[i]);
      REQUIRE(memPtr[i] == value);
    }
  }

  HIP_CHECK(hipFree(memPtr));
  // Reset to default device
  HIP_CHECK(hipSetDevice(0));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following scenarios
 *  - 1) With Location type Host Numa
 *  - 2) With Location type Host Numa Current
 * Test source
 * ------------------------
 *  - unit/memory/hipMemAdvise_v2.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
#if __linux__
TEST_CASE("Unit_hipMemAdvise_v2_HostNuma_HostNumaCurrent") {
  auto supportedDevices = getSupportedDevices();
  if (supportedDevices.empty() || numa_available() < 0) {
    HipTest::HIP_SKIP_TEST("Skipping as System does not have managed memory "
                           "supported devices or No Numa nodes in system");
  }

  HIP_CHECK(hipSetDevice(supportedDevices[0]));

  int maxNode = numa_max_node();
  REQUIRE(maxNode >= 0);

  constexpr int N = 1024;
  constexpr int Nbytes = N * sizeof(int);
  constexpr int value = 10;
  int *memPtr = nullptr;

  HIP_CHECK(hipMallocManaged(reinterpret_cast<void **>(&memPtr), Nbytes,
                             hipMemAttachGlobal));
  REQUIRE(memPtr != nullptr);
  fillDataKernel<<<1, N>>>(memPtr, value);
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("With Host NUMA") {
    for (int node = 0; node <= maxNode; ++node) {
      hipMemLocation location;
      location.type = hipMemLocationTypeHostNuma;
      location.id = node;

      HIP_CHECK(
          hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));

      for (int i = 0; i < N; i++) {
        INFO("For Node " << node << " At index " << i << " Expected value = "
                         << value << " Got value = " << memPtr[i]);
        REQUIRE(memPtr[i] == value);
      }
    }
  }

  SECTION("With Host Numa Current") {
    hipMemLocation location;
    location.type = hipMemLocationTypeHostNumaCurrent;

    HIP_CHECK(
        hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));

    for (int i = 0; i < N; i++) {
      INFO("At index " << i << " Expected value = " << value
                       << " Got value = " << memPtr[i]);
      REQUIRE(memPtr[i] == value);
    }
  }

  HIP_CHECK(hipFree(memPtr));
  // Reset to default device
  HIP_CHECK(hipSetDevice(0));
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following Negative scenarios
 *  - 1) With dev_ptr as nullptr
 *  - 2) With count 0
 *  - 3) With count larger than actual size
 *  - 4) With invalid numa node
 *  - 5) With Invalid Advise
 * Test source
 * ------------------------
 *  - unit/memory/hipMemAdvise_v2.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemAdvise_v2_Negative") {
  auto supportedDevices = getSupportedDevices();
  if (supportedDevices.empty()) {
    HipTest::HIP_SKIP_TEST(
        "Test need at least one device with managed memory support");
  }

  HIP_CHECK(hipSetDevice(supportedDevices[0]));

  constexpr int N = 16;
  constexpr int Nbytes = N * sizeof(int);

  void *memPtr = nullptr;
  HIP_CHECK(hipMallocManaged(&memPtr, Nbytes, hipMemAttachGlobal));
  REQUIRE(memPtr != nullptr);

  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;

  SECTION("With dev_ptr as nullptr") {
    HIP_CHECK_ERROR(
        hipMemAdvise_v2(nullptr, Nbytes, hipMemAdviseSetReadMostly, location),
        hipErrorInvalidValue);
  }

  SECTION("With count 0") {
    HIP_CHECK_ERROR(
        hipMemAdvise_v2(memPtr, 0, hipMemAdviseSetReadMostly, location),
        hipErrorInvalidValue);
  }

  SECTION("With count larger than actual size") {
    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes + 10,
                                    hipMemAdviseSetReadMostly, location),
                    hipErrorInvalidValue);
  }

  SECTION("With Invalid location -1") {
    hipMemLocation location;
    location.type = static_cast<hipMemLocationType>(-1);

    HIP_CHECK_ERROR(
        hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location),
        hipErrorInvalidValue);
  }

  SECTION("With Invalid Advise") {
    hipMemLocation location;
    location.type = hipMemLocationTypeDevice;
    location.id = 0;

    hipMemoryAdvise advise = static_cast<hipMemoryAdvise>(-1);

    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes, advise, location),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(memPtr));
  // Reset to default device
  HIP_CHECK(hipSetDevice(0));
}
