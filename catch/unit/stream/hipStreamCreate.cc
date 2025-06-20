/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "streamCommon.hh"

TEST_CASE("Unit_hipStreamCreate_default") {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);         // Check if stream has a valid ptr
  REQUIRE(hip::checkStream(stream));  // check its flags and priority
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipStreamCreate_Negative") {
  REQUIRE(hipErrorInvalidValue == hipStreamCreate(nullptr));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the behaviour of below Stream API's
 *  - during the the stream capture :
 *  - hipStreamCreate, hipStreamDestroy, hipStreamCreateWithFlags,
 *  - hipStreamGetFlags, hipDeviceGetStreamPriorityRange,
 *  - hipStreamCreateWithPriority, hipStreamGetPriority, hipStreamGetDevice
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreate.cc
 */
TEST_CASE("Unit_StreamAPIs_WhileStreamIsCapturing") {
  GENERATE_CAPTURE();

  hipStream_t captureStream{nullptr};
  HIP_CHECK(hipStreamCreate(&captureStream));
  REQUIRE(captureStream != nullptr);
  BEGIN_CAPTURE(captureStream);

  // hipStreamCreate, hipStreamDestroy
  SECTION("Stream Create and Destroy") {
    hipStream_t stream{nullptr};
    HIP_CHECK(hipStreamCreate(&stream));
    REQUIRE(stream != nullptr);
    HIP_CHECK(hipStreamDestroy(stream));
  }

  // hipStreamCreateWithFlags, hipStreamGetFlags
  SECTION("Stream With Flags") {
    const unsigned int flagUnderTest = GENERATE(hipStreamDefault, hipStreamNonBlocking);
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreateWithFlags(&stream, flagUnderTest));

    unsigned int flag{};
    HIP_CHECK(hipStreamGetFlags(stream, &flag));
    REQUIRE(flag == flagUnderTest);

    HIP_CHECK(hipStreamDestroy(stream));
  }

  // hipDeviceGetStreamPriorityRange, hipStreamCreateWithPriority,
  // hipStreamGetPriority, hipStreamGetDevice
  SECTION("Stream With Priority") {
    HIP_CHECK(hipSetDevice(0));

    int priority_low{};
    int priority_high{};
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));

    int priority = priority_high;
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority));

    int priorityToValidate{};
    HIP_CHECK(hipStreamGetPriority(stream, &priorityToValidate));
    REQUIRE(priorityToValidate == priority);

    hipDevice_t device;
    HIP_CHECK(hipStreamGetDevice(stream, &device));
    REQUIRE(device == 0);

    HIP_CHECK(hipStreamDestroy(stream));
  }

  END_CAPTURE(captureStream);
  HIP_CHECK(hipStreamDestroy(captureStream));
}

// Helper function used in callback scenario
static void callBackFunction(hipStream_t stream, hipError_t status, void* userData) {
  REQUIRE(stream != nullptr);
  REQUIRE(status == hipSuccess);
  int* a = reinterpret_cast<int*>(userData);
  *a = 100;
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the behaviour of hipStreamAddCallback API
 *  - during the the stream capture.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreate.cc
 */
TEST_CASE("Unit_StreamAPIs_WhileStreamIsCapturing_hipStreamAddCallback") {
  bool capture = true;
  hipStream_t captureStream{nullptr};
  HIP_CHECK(hipStreamCreate(&captureStream));
  REQUIRE(captureStream != nullptr);
  BEGIN_CAPTURE(captureStream);

  // hipStreamAddCallback
  int data = 0;
  HIP_CHECK_ERROR(
      hipStreamAddCallback(captureStream, callBackFunction, reinterpret_cast<void*>(&data), 0),
      hipErrorStreamCaptureUnsupported);

  // hipStreamEndCapture is failing due to previous error, avoided calling that
  HIP_CHECK(hipStreamDestroy(captureStream));
}
