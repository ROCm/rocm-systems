/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipGreenCtxRecordEvent hipGreenCtxRecordEvent
 * @{
 * @ingroup GreenContextTest
 * `hipGreenCtxRecordEvent` and `hipGreenCtxWaitEvent` APIs
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "hip_greenctx_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Creates green context and records a valid event on it
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipGreenCtxRecordEvent_Sanity") {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipEvent_t event = nullptr;
  hipGreenCtx_t green_ctx = nullptr;
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  HIP_CHECK(hipGreenCtxRecordEvent(green_ctx, event));

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipGreenCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Records an event on a green context and waits on it
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipGreenCtxWaitEvent_Sanity") {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipEvent_t event = nullptr;
  hipGreenCtx_t green_ctx = nullptr;
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  HIP_CHECK(hipGreenCtxRecordEvent(green_ctx, event));
  HIP_CHECK(hipGreenCtxWaitEvent(green_ctx, event));

  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipGreenCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Validates record/wait behavior with real work in green context streams
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipGreenCtxRecordEventFunctional") {
  HIP_CHECK(hipSetDevice(0));
  hipDevResourceDesc_t desc{};
  hipEvent_t event = nullptr;
  hipGreenCtx_t green_ctx = nullptr;
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  HIP_CHECK(hipEventCreate(&event));
  REQUIRE(event != nullptr);

  hipStream_t stream1 = nullptr;
  hipStream_t stream2 = nullptr;
  HIP_CHECK(hipGreenCtxStreamCreate(&stream1, green_ctx, hipStreamNonBlocking, 0x0));
  HIP_CHECK(hipGreenCtxStreamCreate(&stream2, green_ctx, hipStreamNonBlocking, 0x0));
  REQUIRE(stream1 != nullptr);
  REQUIRE(stream2 != nullptr);

  constexpr size_t kNumElements = 1 << 16;
  const size_t kBytes = kNumElements * sizeof(int);
  int* h_a = reinterpret_cast<int*>(malloc(kBytes));
  int* h_b = reinterpret_cast<int*>(malloc(kBytes));
  int* h_c = reinterpret_cast<int*>(malloc(kBytes));
  REQUIRE(h_a != nullptr);
  REQUIRE(h_b != nullptr);
  REQUIRE(h_c != nullptr);

  for (size_t i = 0; i < kNumElements; ++i) {
    h_a[i] = static_cast<int>(i);
    h_b[i] = static_cast<int>(2 * i);
    h_c[i] = 0;
  }

  int* d_a = nullptr;
  int* d_b = nullptr;
  int* d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));
  HIP_CHECK(hipMalloc(&d_c, kBytes));

  HIP_CHECK(hipMemcpyAsync(d_a, h_a, kBytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(d_b, h_b, kBytes, hipMemcpyHostToDevice, stream1));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((kNumElements + kThreads - 1) / kThreads);
  HipTest::vectorADD<<<blocks, kThreads, 0, stream1>>>(d_a, d_b, d_c, kNumElements);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipGreenCtxRecordEvent(green_ctx, event));
  HIP_CHECK(hipGreenCtxWaitEvent(green_ctx, event));

  HIP_CHECK(hipMemcpyAsync(h_c, d_c, kBytes, hipMemcpyDeviceToHost, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));

  for (size_t i = 0; i < kNumElements; ++i) {
    REQUIRE(h_c[i] == h_a[i] + h_b[i]);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipGreenCtxDestroy(green_ctx));
}

/**
 * End doxygen group hipGreenCtxRecordEvent.
 * @}
 */
