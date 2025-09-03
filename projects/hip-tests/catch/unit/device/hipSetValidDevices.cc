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
#include <hip_test_process.hh>

#if __linux__
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define N 1024
#define NBYTES N * sizeof(int)

/**
 * @addtogroup hipSetValidDevices hipSetValidDevices
 * @{
 * @ingroup DeviceTest
 * `hipSetValidDevices(int* device_arr, int len)` -
 * Sets a list of valid devices that can be used by HIP runtime
 */

/**
 * Kernel to double each element in the given array
 */
static __global__ void doubleKernel(int* arr, size_t arrSize) {
  size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < arrSize; i += stride) {
    arr[i] += arr[i];
  }
}

/**
 * Helper function to get Current device
 */
static inline int getCurrentDevice() {
  int currentDevice = -1;
  HIP_CHECK(hipGetDevice(&currentDevice));
  return currentDevice;
}

/**
 * Helper function to perform Memory copy and kernel operations
 */
static void performOperations() {
  int hostMem[N];
  for (int i = 0; i < N; i++) {
    hostMem[i] = 5;
  }
  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
  HIP_CHECK(hipMemcpy(devMem, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

  doubleKernel<<<1, N>>>(devMem, N);

  HIP_CHECK(hipMemcpy(hostMem, devMem, N * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    REQUIRE(hostMem[i] == 10);
  }
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that hipSetValidDevices API can handle invalid parameters
 *    -#  When device array passed is `nullptr` but len is not `0`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -#  When len exceeds the number of devices in the system
 *      - Expected output: return `hipErrorInvalidValue`
 *    -#  When the device Id specified in the list does not exist
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_Negative") {
  auto totalDevices = HipTest::getDeviceCount();
  int device_arr1[] = {0};
  int device_arr2[] = {totalDevices};

  SECTION("Devicearray - nullptr") {
    HIP_CHECK_ERROR(hipSetValidDevices(nullptr, 1), hipErrorInvalidValue);
  }
  SECTION("len > total devices") {
    HIP_CHECK_ERROR(hipSetValidDevices(device_arr1, totalDevices + 2), hipErrorInvalidValue);
  }
  SECTION("DeviceId is not valid") {
    HIP_CHECK_ERROR(hipSetValidDevices(device_arr2, 1), hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the below scenarios
 *  - 1) Call hipSetValidDevices with length < 0 and valid deviceArr of size 1,
 *  -    and expect hipErrorInvalidValue.
 *  - 2) Call hipSetValidDevices with length 1 and with deviceArr
 *  -    {invalid device, valid device}, and expect hipErrorInvalidDevice.
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_Negative_Length_Lessthan_DeviceArrSize") {
  int deviceCount = HipTest::getDeviceCount();

  SECTION("length < 0 and valid dev arr") {
    int length = -1;
    int deviceArr[1] = {0};
    HIP_CHECK_ERROR(hipSetValidDevices(deviceArr, length), hipErrorInvalidValue);
  }

  SECTION("length < dev arr size and deviceArr {Invalid device, 0}") {
    int length = 1;
    int deviceArr[2] = {deviceCount, 0};
    HIP_CHECK_ERROR(hipSetValidDevices(deviceArr, length), hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates the functionality of hipSetValidDevices by default and
 *    also by resetting using hipSetDevice
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_Positive_Basic") {
  int totalDevices = HipTest::getDeviceCount();
  if (totalDevices < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 or more GPUs. Skipping.");
    return;
  }

  // By default, without setting any device, validate that 0th device is being used
  int device;
  HIP_CHECK(hipGetDevice(&device));
  REQUIRE(device == 0);

  // Set the devices 1 and 0 as valid ones using hipSetValidDevices
  int valid_devices1[] = {1, 0};
  HIP_CHECK(hipSetValidDevices(valid_devices1, 2));

  // Fetch the device and validate that the device 1 is being used currently
  // Since the device 1 is set as the first valid device earlier
  HIP_CHECK(hipGetDevice(&device));
  REQUIRE(device == 1);

  if (totalDevices > 2) {
    // Set the device 2 as the current device
    HIP_CHECK(hipSetDevice(2));
    // Fetch the device and validate that the device 2 is being used currently
    // This is to confirm that hipSetDevice sets the device (if the device exists)
    // irrespective of the valid devices set by the app
    HIP_CHECK(hipGetDevice(&device));
    REQUIRE(device == 2);
    // Set 0 as the valid device
    int valid_devices2[] = {0};
    HIP_CHECK(hipSetValidDevices(valid_devices2, 1));
    // Fetch the device and validate that the device 2 is the current device still
    // Since hipSetValidDevices doesn't take effect once hipSetDevice is set
    HIP_CHECK(hipGetDevice(&device));
    REQUIRE(device == 2);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the below scenarios
 *  - 1) Sets the all available devices with hipSetValidDevices
 *  -    one after another
 *  - 2) Sets the all available devices with hipSetValidDevices
 *  -    all at a time
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_WithAllDevicesInSystem") {
  int deviceCount;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  SECTION("Set all devices one after another") {
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
      int length = 1;
      int deviceArr[1] = {deviceId};
      HIP_CHECK(hipSetValidDevices(deviceArr, length));

      REQUIRE(getCurrentDevice() == deviceId);
      performOperations();
    }
  }

  SECTION("Set all devices once in decending order of device ids") {
    std::vector<int> devices;
    for (int deviceId = deviceCount - 1; deviceId >= 0; --deviceId) {
      devices.push_back(deviceId);
    }
    int length = devices.size();
    HIP_CHECK(hipSetValidDevices(devices.data(), length));

    REQUIRE(getCurrentDevice() == (deviceCount - 1));
    performOperations();
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the below scenarios
 *  - 1) Call hipSetValidDevices with length 0 and deviceArr is nullPtr,
 *  -    and expect the default behavior of trying devices sequentially.
 *  - 2) Call hipSetValidDevices with length 0 and valid deviceArr,
 *  -    and expect the default behavior of trying devices sequentially.
 *  - 3) Call hipSetValidDevices with length 1 and deviceArray size 2,
 *  -    and expect to select the first index in deviceArray.
 *  - 4) Call hipSetValidDevices with length 1 and deviceArr contains 1 valid
 *  -    and 1 invalid device, and expect to select the first index in
 *  -    deviceArray even the another device is invalid
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_Positive_Cases") {
  int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping, as this test requires more than 2 GPUs");
    return;
  }

  SECTION("length is 0 and deviceArr is nullPtr") {
    int length = 0;
    int* deviceArr = nullptr;
    HIP_CHECK(hipSetValidDevices(deviceArr, length));

    REQUIRE(getCurrentDevice() == 0);
  }

  SECTION("length is 0 and deviceArr is valid") {
    int length = 0;
    int deviceArr[2] = {1, 0};
    HIP_CHECK(hipSetValidDevices(deviceArr, length));

    REQUIRE(getCurrentDevice() == 0);
  }

  SECTION("length < dev arr size") {
    int length = 1;
    int deviceArr[2] = {1, 0};
    HIP_CHECK(hipSetValidDevices(deviceArr, length));

    REQUIRE(getCurrentDevice() == 1);
  }

  SECTION("Len < dev arr size and dev arr contains invalid devices") {
    int length = 1;
    int deviceArr[2] = {1, deviceCount};
    HIP_CHECK(hipSetValidDevices(deviceArr, length));

    REQUIRE(getCurrentDevice() == 1);
  }
  performOperations();
}

#if __linux__
/**
 * Test Description
 * ------------------------
 *  - This test case checks the behavior of hipSetValidDevices
 *  - in multi process scenario, check the current device at start in
 *  - Parant and child process and set different devices in child and parent
 *  - using hipSetValidDevices and validate.
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_MultiProcess") {
  int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping, as this test requires more than 2 GPUs");
    return;
  }

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid != 0) {  // Parent process
    REQUIRE(getCurrentDevice() == 0);

    int length = 2;
    int deviceArr[2] = {1, 0};
    HIP_CHECK(hipSetValidDevices(deviceArr, length));

    REQUIRE(getCurrentDevice() == 1);
    performOperations();

    int status;
    REQUIRE(wait(&status) >= 0);

  } else {  // Child process
    REQUIRE(getCurrentDevice() == 0);

    int length = 2;
    int device_arr_c[2] = {0, 1};
    HIP_CHECK(hipSetValidDevices(device_arr_c, length));

    REQUIRE(getCurrentDevice() == 0);
    performOperations();

    exit(0);
  }
}
#endif

/**
 * Helper function used in multi threaded scenario to set Valid devices
 */
void launchFunction(int deviceId) {
  REQUIRE(getCurrentDevice() == 0);

  int length = 1;
  int deviceArr[1] = {deviceId};
  HIP_CHECK(hipSetValidDevices(deviceArr, length));

  REQUIRE(getCurrentDevice() == deviceId);
  performOperations();
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the behavior of hipSetValidDevices
 *  - in multi threaded scenarios.
 *  - 1) Launch Multiple threads one after another and set different devices
 *  - 2) Launch Multiple threads all at a time and set different devices
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_MultiThread") {
  int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping, as this test requires more than 2 GPUs");
    return;
  }

  REQUIRE(getCurrentDevice() == 0);

  SECTION("Serial") {
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
      std::thread thread(launchFunction, deviceId);
      thread.join();
    }
  }

  SECTION("Parallel") {
    std::vector<std::thread> threads;
    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
      threads.push_back(std::thread(launchFunction, deviceId));
    }
    for (int t = 0; (t < deviceCount) && (t < threads.size()); t++) {
      threads[t].join();
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following scenario
 *  - 1) Allocate device memory in device 0, fill it
 *  - 2) Set device 1 using hipSetValidDevices
 *  - 3) Allocate device memory in device 1
 *  - 4) Copy data from device 0 to device 1.
 * Test source
 * ------------------------
 *  - unit/device/hipSetValidDevices.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipSetValidDevices_with_hipMemcpyPeer") {
  int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping, as this test requires more than 2 GPUs");
    return;
  }

  REQUIRE(getCurrentDevice() == 0);

  int* dev0_Arr = nullptr;
  HIP_CHECK(hipMalloc(&dev0_Arr, NBYTES));
  REQUIRE(dev0_Arr != nullptr);

  int srcHostMem[N];
  for (int i = 0; i < N; i++) {
    srcHostMem[i] = 5;
  }
  HIP_CHECK(hipMemcpy(dev0_Arr, srcHostMem, N * sizeof(int), hipMemcpyHostToDevice));

  int length = 1;
  int deviceArr[1] = {1};
  HIP_CHECK(hipSetValidDevices(deviceArr, length));

  REQUIRE(getCurrentDevice() == 1);

  int* dev1_Arr = nullptr;
  HIP_CHECK(hipMalloc(&dev1_Arr, NBYTES));
  REQUIRE(dev1_Arr != nullptr);

  int canAccessPeer = -1;
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 1, 0));
  REQUIRE(canAccessPeer == 1);

  HIP_CHECK(hipMemcpyPeer(dev1_Arr, 1, dev0_Arr, 0, N * sizeof(int)));

  int dstHostMem[N];
  for (int i = 0; i < N; i++) {
    dstHostMem[i] = 0;
  }

  HIP_CHECK(hipMemcpy(dstHostMem, dev1_Arr, N * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    REQUIRE(dstHostMem[i] == 5);
  }
}
