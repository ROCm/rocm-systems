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

#include <thread>

#include <hip_test_common.hh>
#include <threaded_zig_zag_test.hh>

/**
 * @addtogroup hipSetDevice hipSetDevice
 * @{
 * @ingroup DeviceTest
 * `hipSetDevice(int deviceId)` -
 * Set default device to be used for subsequent hip API calls from this thread.
 */

/**
 * Test Description
 * ------------------------
 *  - Performs multiple set/get device operations and verifies
 *    that the device that is set is the one that is gotten.
 * Test source
 * ------------------------
 *  - unit/device/hipSetGetDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipSetDevice_BasicSetGet") {
  int numDevices = 0;
  int device{};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices != 0);

  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipGetDevice(&device));
    REQUIRE(device == i);

    // Check for hipDevice_t as well
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, i));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs set/get operations for each detected
 *    device from multiple threads.
 * Test source
 * ------------------------
 *  - unit/device/hipSetGetDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetSetDevice_MultiThreaded") {
  auto maxThreads = std::thread::hardware_concurrency();
  auto deviceCount = HipTest::getDeviceCount();

  auto thread = [&]() {
    for (int i = 0; i < deviceCount; i++) {
      HIP_CHECK_THREAD(hipSetDevice(i));
      int get = -1;
      HIP_CHECK_THREAD(hipGetDevice(&get));
      REQUIRE_THREAD(get == i);

      // check hipDeviceGet
      hipDevice_t device;
      HIP_CHECK_THREAD(hipDeviceGet(&device, i));

      // Alloc some memory and set it
      unsigned int* ptr{nullptr};
      HIP_CHECK_THREAD(hipMalloc(&ptr, sizeof(unsigned int)));
      REQUIRE_THREAD(ptr != nullptr);
      HIP_CHECK_THREAD(hipMemset(ptr, 0x0A, sizeof(unsigned int)));
      int res{0};
      HIP_CHECK_THREAD(hipMemcpy(&res, ptr, sizeof(unsigned int), hipMemcpyDeviceToHost));
      REQUIRE_THREAD(res == 0x0A0A0A0A);
      HIP_CHECK_THREAD(hipFree(ptr));
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(maxThreads);

  for (unsigned i = 0; i < maxThreads; i++) {
    pool.emplace_back(std::thread(thread));
  }

  for (auto& i : pool) {
    i.join();
  }

  HIP_CHECK_THREAD_FINALIZE();
}

/**
 * Test Description
 * ------------------------
 *  - Performs set/get device for separate devices on two
 *    threads and validates device ordinance via memory allocation.
 * Test source
 * ------------------------
 *  - unit/device/hipSetGetDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipSetGetDevice_Positive_Threaded_Basic") {
  class HipSetGetDeviceThreadedTest : public ThreadedZigZagTest<HipSetGetDeviceThreadedTest> {
   public:
    void TestPart1() { HIP_CHECK(hipSetDevice(0)); }
    void TestPart2() {
      HIP_CHECK_THREAD(hipSetDevice(1));
      HIP_CHECK_THREAD(hipMalloc(&ptr, 2 * 1024 * 1024));
    }
    void TestPart3() {
      int device = -1;
      HIP_CHECK_THREAD(hipGetDevice(&device));
      REQUIRE_THREAD(device == 0);
      device = -1;
      // To check if set device worked properly, outside of hipGetDevice
      HIP_CHECK_THREAD(hipPointerGetAttribute(&device, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                              reinterpret_cast<hipDeviceptr_t>(ptr)));
      REQUIRE_THREAD(device == 1);
    }
    void TestPart4() {
      int device = -1;
      HIP_CHECK_THREAD(hipGetDevice(&device));
      REQUIRE_THREAD(device == 1);
      HIP_CHECK_THREAD(hipFree(ptr));
    }

   private:
    void* ptr = nullptr;
  };

  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This rest requires 2 GPUs. Skipping test");
    return;
  }

  HipSetGetDeviceThreadedTest test;
  test.run();
}

/**
 * Test Description
 * ------------------------
 *  - Validates that get/set device APIs can handle invalid parameters
 *    -# Get device when device is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Set device with ordinal number `-1`
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# Set device to the ID which is out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipSetGetDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipSetGetDevice_Negative") {
  SECTION("Get Device - nullptr") { HIP_CHECK_ERROR(hipGetDevice(nullptr), hipErrorInvalidValue); }

  SECTION("Set Device - -1") { HIP_CHECK_ERROR(hipSetDevice(-1), hipErrorInvalidDevice); }

  SECTION("Set Device - NumDevices + 1") {
    HIP_CHECK_ERROR(hipSetDevice(HipTest::getDeviceCount()), hipErrorInvalidDevice);
  }
}

/**
 * End doxygen group hipSetDevice.
 * @}
 */

/**
 * @addtogroup hipGetDevice hipGetDevice
 * @{
 * @ingroup DeviceTest
 * `hipGetDevice(int* deviceId)` -
 * Return the default device id for the calling host thread.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipSetDevice_BasicSetGet
 *  - @ref Unit_hipGetSetDevice_MultiThreaded
 *  - @ref Unit_hipSetGetDevice_Negative
 */

/**
 * End doxygen group hipGetDevice.
 * @}
 */

/**
 * @addtogroup hipDeviceGet hipDeviceGet
 * @{
 * @ingroup DriverTest
 * `hipDeviceGet(hipDevice_t* device, int ordinal)` -
 * Returns a handle to a compute device.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipSetDevice_BasicSetGet
 *  - @ref Unit_hipGetSetDevice_MultiThreaded
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the device is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When device ordinal is out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When device ordinal is negative
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipSetGetDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGet_Negative") {
  // TODO enable after EXSWCPHIPT-104 is fixed
#if HT_NVIDIA
  HIP_CHECK(hipInit(0));
  SECTION("Nullptr as handle") { HIP_CHECK_ERROR(hipDeviceGet(nullptr, 0), hipErrorInvalidValue); }
#endif

  SECTION("Out of bound ordial - positive") {
    hipDevice_t device{};
    auto totalDevices = HipTest::getDeviceCount();
    HIP_CHECK_ERROR(hipDeviceGet(&device, totalDevices), hipErrorInvalidDevice);
  }

  SECTION("Out of bound ordial - negative") {
    hipDevice_t device{};
    HIP_CHECK_ERROR(hipDeviceGet(&device, -1), hipErrorInvalidDevice);
  }
}
