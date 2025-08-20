/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#define NUMBER_OF_THREADS 10
static bool thread_results[NUMBER_OF_THREADS];

/**
 * @addtogroup hipStreamGetDevice hipStreamGetDevice
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device)` -
 * Get the device assocaited with the stream.
 * @}
 */

/**
 * Test Description
 * ------------------------
 *    - NegativeTest case.
 * Pass device as nullptr and verify hipErrorInvalidValue
 * Pass stream as nullptr and verify hipErrorInvalidValue
 * Pass device as nullptr for hipStreamPerThread and verify hipErrorInvalidValue

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamGetDevice.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamGetDevice_Negative") {
  hipStream_t stream;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK_ERROR(hipStreamGetDevice(nullptr, nullptr), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipStreamGetDevice(hipStreamPerThread, nullptr), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipStreamGetDevice(stream, nullptr), hipErrorInvalidValue);
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - NegativeTest case.
 * Test case to validate hipStreamGetDevice on user created stream,
 * Null stream and hipStreamPerThread.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamGetDevice.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamGetDevice_Usecase") {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  REQUIRE(device_count != 0);

  SECTION("Null Stream") {
    CTX_CREATE();

    hipDevice_t device_from_stream, device_from_ordinal;
    HIP_CHECK(hipStreamGetDevice(nullptr, &device_from_stream));

    HIP_CHECK(hipDeviceGet(&device_from_ordinal, 0));
    REQUIRE(device_from_stream == device_from_ordinal);

    CTX_DESTROY();
  }

  SECTION("Stream Per Thread") {
    CTX_CREATE();

    hipDevice_t device_from_stream, device_from_ordinal;
    HIP_CHECK(hipStreamGetDevice(hipStreamPerThread, &device_from_stream));

    HIP_CHECK(hipDeviceGet(&device_from_ordinal, 0));
    REQUIRE(device_from_stream == device_from_ordinal);

    CTX_DESTROY();
  }

  SECTION("Created Stream") {
    for (int i = 0; i < device_count; i++) {
      HIP_CHECK(hipSetDevice(i));

      hipDevice_t device_from_stream, device_from_ordinal;
      hipStream_t stream;

      HIP_CHECK(hipStreamCreate(&stream));
      HIP_CHECK(hipStreamGetDevice(stream, &device_from_stream));

      HIP_CHECK(hipDeviceGet(&device_from_ordinal, i));
      REQUIRE(device_from_stream == device_from_ordinal);
      HIP_CHECK(hipStreamDestroy(stream));
    }
  }
}

/**
 * Test Description
 * ------------------------
 * - NegativeTest case.
 * Test case to multi-threaded scenario for hipStreamGetDevice

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamGetDevice.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

static bool validateStreamGetDevice() {
  int gpu = 0;
  hipDevice_t device_from_stream;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamGetDevice(stream, &device_from_stream));
  HIP_CHECK(hipStreamDestroy(stream));

  REQUIRE(device_from_stream == gpu);
  return true;
}

static void thread_Test(int threadNum) { thread_results[threadNum] = validateStreamGetDevice(); }

static bool test_hipStreamGetDevice_MThread() {
  std::vector<std::thread> tests;

  // Spawn the test threads
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    thread_results[idx] = false;
    tests.push_back(std::thread(thread_Test, idx));
  }
  // Wait for all threads to complete
  for (std::thread& t : tests) {
    t.join();
  }
  // Wait for thread
  bool status = true;
  for (int idx = 0; idx < NUMBER_OF_THREADS; idx++) {
    status = status & thread_results[idx];
  }
  return status;
}

TEST_CASE("Unit_hipStreamGetDevice_MThread") { REQUIRE(true == test_hipStreamGetDevice_MThread()); }

/**
 * Test Description
 * ------------------------
 * - NegativeTest case.
 * Create a stream with gpu1, then set device to gpu0
 * call hipStreamGetDevice on the stream and verify the
 * returned device is same as initial set device.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamGetDevice.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamGetDevice_SetDiffDevice") {
  hipDevice_t device_from_stream;
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }
  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    for (int j = 0; j < device_count; ++j) {
      if (i != j) {
        HIP_CHECK(hipSetDevice(j));
        HIP_CHECK(hipStreamGetDevice(stream, &device_from_stream));
        REQUIRE(device_from_stream == i);
      }
    }
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Test Description
 * ------------------------
 * - NegativeTest case.
 * Set hip Devices to each available gpu and probe
 * hipStreamGetDevice with null stream.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamGetDevice.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 *      Test to be run only on AMD machine as it's failing in CUDA.
 */
#if HT_AMD
TEST_CASE("Unit_hipStreamGetDevice_NullStream") {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  REQUIRE(device_count != 0);

  for (int i = 0; i < device_count; i++) {
    HIP_CHECK(hipSetDevice(i));
    hipDevice_t device_from_stream;
    HIP_CHECK(hipStreamGetDevice(0, &device_from_stream));
    REQUIRE(device_from_stream == i);
  }
}
#endif

/**
 * End doxygen group StreamTest.
 * @}
 */
