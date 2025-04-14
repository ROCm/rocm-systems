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
#include <hip_test_defgroups.hh>
#include <hip_test_kernels.hh>
#include <chrono>
#include <thread>
#define UNUSED(expr) do { (void)(expr); } while (0)
#ifdef __HIP_PLATFORM_AMD__
#define HIPRT_CB
#endif
/**
 * @addtogroup hipStreamAddCallback_spt hipStreamAddCallback_spt
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamAddCallback_spt(hipStream_t stream,
 *                                  hipStreamCallback_t callback,
 *                                  void* userData,
 *                                  unsigned int flags)` -
 * Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.
 */
namespace hipStreamAddCallbackTest_spt {
size_t NSize = 4 * 1024 * 1024;
float *A_h, *C_h;
bool gcbDone = false;
bool gPassed = true;
void *ptr0xff = reinterpret_cast<void *>(0xffffffff);
void *gusrptr;
hipStream_t gstream;
void HIPRT_CB Callback(hipStream_t stream, hipError_t status,
                       void* userData) {
  UNUSED(stream);
  HIP_CHECK(status);
  REQUIRE(userData == NULL);
  gPassed = true;
  for (size_t i = 0; i < NSize; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("Data mismatch :" << i);
      gPassed = false;
      break;
    }
  }
  gcbDone = true;
}
bool testStreamCallbackFunctionality(bool isDefault) {
  float *A_d, *C_d;
  size_t Nbytes = NSize * sizeof(float);
  gcbDone = false;
  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(C_h != nullptr);
  // Fill with Phi + i
  for (size_t i = 0; i < NSize; i++) {
    A_h[i] = 1.618f + i;
  }
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  if (isDefault) {
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice,
                            0));
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                        dim3(threadsPerBlock), 0, 0, A_d, C_d, NSize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                            0));
    HIP_CHECK(hipStreamAddCallback_spt(0, Callback, nullptr, 0));
    while (!gcbDone)
     // Sleep for 100 ms
     std::this_thread::sleep_for(std::chrono::microseconds(100000));
  } else {
    hipStream_t mystream;
    HIP_CHECK(hipStreamCreateWithFlags(&mystream, hipStreamNonBlocking));
    HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice,
                            mystream));
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                      dim3(threadsPerBlock), 0, mystream, A_d, C_d, NSize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                            mystream));
    HIP_CHECK(hipStreamAddCallback_spt(mystream, Callback, nullptr, 0));
    while (!gcbDone)
      // Sleep for 100 ms
      std::this_thread::sleep_for(std::chrono::microseconds(100000));
    HIP_CHECK(hipStreamDestroy(mystream));
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(C_d)));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
  free(C_h);
  free(A_h);
  return gPassed;
}
void Callback_ChkUsrdataPtr(hipStream_t stream, hipError_t status,
                            void* userData) {
  REQUIRE(stream == gstream);
  HIP_CHECK(status);
  gPassed = true;
  if (gusrptr != userData) {
    gPassed = false;
  }
  gcbDone = true;
}
void Callback_ChkStreamValue(hipStream_t stream, hipError_t status,
                             void* userData) {
  REQUIRE(userData == nullptr);
  HIP_CHECK(status);
  gPassed = true;
  if (stream != gstream) {
    gPassed = false;
  }
  gcbDone = true;
}
}  // namespace hipStreamAddCallbackTest_spt
using hipStreamAddCallbackTest_spt::gcbDone;
using hipStreamAddCallbackTest_spt::gPassed;
using hipStreamAddCallbackTest_spt::ptr0xff;
using hipStreamAddCallbackTest_spt::gusrptr;
using hipStreamAddCallbackTest_spt::gstream;
using hipStreamAddCallbackTest_spt::testStreamCallbackFunctionality;
using hipStreamAddCallbackTest_spt::Callback;
using hipStreamAddCallbackTest_spt::Callback_ChkUsrdataPtr;
using hipStreamAddCallbackTest_spt::Callback_ChkStreamValue;
/**
 * Test Description
 * ------------------------
 *  - Validates if userData pointer is passed properly to callback.
 *  - Validates if stream value is passed properly to callback.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamAddCallback_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamAddCallback_spt_ParamTst_Positive") {
  hipStream_t mystream;
  HIP_CHECK(hipStreamCreate(&mystream));
  SECTION("userData pointer value validation") {
    gstream = mystream;
    gusrptr = ptr0xff;
    gPassed = true;
    gcbDone = false;
    HIP_CHECK(hipStreamAddCallback_spt(mystream, Callback_ChkUsrdataPtr,
                                  gusrptr, 0));
    while (!gcbDone) {
      // Sleep for 100 ms
      std::this_thread::sleep_for(std::chrono::microseconds(100000));
    }
    REQUIRE(gPassed);
  }
  SECTION("stream value validation") {
    gstream = mystream;
    gPassed = true;
    gcbDone = false;
    HIP_CHECK(hipStreamAddCallback_spt(mystream, Callback_ChkStreamValue,
                                  nullptr, 0));
    while (!gcbDone) {
      // Sleep for 100 ms
      std::this_thread::sleep_for(std::chrono::microseconds(100000));
    }
    REQUIRE(gPassed);
  }
  HIP_CHECK(hipStreamDestroy(mystream));
}
 /**
 * Test Description
 * ------------------------
 *  - Basic test to validate Negative cases of hipStreamAddCallback_spt.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamAddCallback_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamAddCallback_spt_ParamTst_Negative") {
  hipStream_t mystream;
  HIP_CHECK(hipStreamCreate(&mystream));
  SECTION("callback is nullptr for non-default stream") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback_spt(mystream, nullptr,
                                         nullptr, 0));
  }
  SECTION("callback is nullptr for default stream") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback_spt(0, nullptr,
                                         nullptr, 0));
  }
  SECTION("flag is nonzero for non-default stream") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback_spt(mystream, Callback,
                                         nullptr, 10));
  }
  SECTION("flag is nonzero for default stream") {
    REQUIRE_FALSE(hipSuccess == hipStreamAddCallback_spt(0, Callback,
                                         nullptr, 10));
  }
  HIP_CHECK(hipStreamDestroy(mystream));
}
/**
 * Test Description
 * ------------------------
 *  - Validates hipStreamAddCallback_spt functionality with default stream.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamAddCallback_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamAddCallback_spt_WithDefaultStream") {
  bool TestPassed = true;
  TestPassed = testStreamCallbackFunctionality(true);
  REQUIRE(TestPassed);
}
 /**
 * Test Description
 * ------------------------
 *  - Validates hipStreamAddCallback_spt functionality with defined stream.
 * Test source
 * ------------------------
 *  - /unit/stream/hipStreamAddCallback_spt.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipStreamAddCallback_spt_WithCreatedStream") {
  bool TestPassed = true;
  TestPassed = testStreamCallbackFunctionality(false);
  REQUIRE(TestPassed);
}
/**
* End doxygen group StreamTest.
* @}
*/

