/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

static __global__ void write_integer(int* memory, int value) { *memory = value; }

TEST_CASE("Unit_hipMallocHost_Positive") {
  int* host_memory = nullptr;

  HIP_CHECK(hipMallocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)));
  REQUIRE(host_memory != nullptr);
  HIP_CHECK(hipHostFree(host_memory));
}

TEST_CASE("Unit_hipMallocHost_DataValidation") {
  int validation_number = 10;
  int* host_memory = nullptr;
  hipEvent_t event = nullptr;

  HIP_CHECK(hipMallocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)));

  write_integer<<<1, 1>>>(host_memory, validation_number);

  SECTION("device sync") { HIP_CHECK(hipDeviceSynchronize()); }

  SECTION("event sync") {
    HIP_CHECK(hipEventCreateWithFlags(&event, 0));
    HIP_CHECK(hipEventRecord(event, nullptr));
    HIP_CHECK(hipEventSynchronize(event));
  }

  SECTION("stream sync") { HIP_CHECK(hipStreamSynchronize(nullptr)); }

  REQUIRE(*host_memory == validation_number);

  if (event != nullptr) {
    HIP_CHECK(hipEventDestroy(event));
  }

  HIP_CHECK(hipHostFree(host_memory));
}

TEST_CASE("Unit_hipMallocHost_Negative") {
  int* host_memory = nullptr;

  SECTION("host memory is nullptr") {
    HIP_CHECK_ERROR(hipMallocHost(nullptr, sizeof(int)), hipErrorInvalidValue);
  }

  SECTION("size is negative") {
    HIP_CHECK_ERROR(hipMallocHost(reinterpret_cast<void**>(&host_memory), -1), hipErrorOutOfMemory);
  }
}

TEST_CASE("Unit_hipMallocHost_Capture") {
  int* host_memory = nullptr;

  hipError_t capture_error = hipSuccess;
  constexpr bool kRelaxedModeAllowed = true;

  BEGIN_CAPTURE_SYNC(capture_error, kRelaxedModeAllowed);
  HIP_CHECK_ERROR(hipMallocHost(reinterpret_cast<void**>(&host_memory), sizeof(int)),
                  capture_error);
  END_CAPTURE_SYNC(capture_error);

  if (capture_error == hipSuccess) {
    REQUIRE(host_memory != nullptr);
    HIP_CHECK(hipHostFree(host_memory));
  }
}
