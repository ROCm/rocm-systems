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

#include <cstdlib>
#include <hip_test_common.hh>
#include <csetjmp>
#include <csignal>
#include "hip/hip_runtime_api.h"
#include "hip_test_context.hh"

/**
 * @addtogroup assert assert
 * @{
 * @ingroup DeviceLanguageTest
 * `void assert(int expression)` -
 * Stops the kernel execution if expression is equal to zero.
 */

__global__ void AssertPassKernel(int* x) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  *x = tid;
  // expected always to be true
  assert(tid >= 0);
}

__global__ void AssertFailKernel(int* x) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  *x = tid;
  // expected to fail for the even thread indices
  assert(tid % 2 == 1);
}

bool isAbortOnErrorEnabled() {
  std::string abort_on_error_env = TestContext::getEnvVar("HIP_SKIP_ABORT_ON_GPU_ERROR");

  if (!abort_on_error_env.empty()) {
    try {
      return !std::stoi(abort_on_error_env);
    } catch (const std::invalid_argument&) {
      return true;
    } catch (const std::out_of_range&) {
      return true;
    }
  }
  return false;
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernels with asserts that have an expression equal to 1.
 *  - Expects that SIGABRT is not raised and kernels have executed successfully.
 * Test source
 * ------------------------
 *  - unit/assertion/assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Assert_Positive_Basic_KernelPass") {
  
#ifdef NDEBUG
  HipTest::HIP_SKIP_TEST("Assertions are disabled on this build.");
  return;
#endif

#if HT_AMD
  if (isAbortOnErrorEnabled()) {
    HipTest::HIP_SKIP_TEST(
        "Test incompatible with aborts enabled through HIP_SKIP_ABORT_ON_GPU_ERROR.");
    return;
  }
#endif

  const int num_blocks = 2;
  const int num_threads = 16;
  int* d_a;
  HIP_CHECK(hipMalloc(&d_a, sizeof(int)));

  AssertPassKernel<<<num_blocks, num_threads, 0, 0>>>(d_a);

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(d_a));
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernels with asserts that have an expression equal to 0.
 *  - Test is skipped on AMD if HIP_SKIP_ABORT_ON_GPU_ERROR=0 to avoid call to std::abort() which
 * cannot safely be handled by the test.
 *  - Expects that `hipErrorAssert` is returned from `hipDeviceSynchronize` on NVIDIA.
 *  - Expects that `hipErrorLaunchFailure` is returned from `hipDeviceSynchronize` on AMD. HSA error
 * codes don't have enough granularity to distinguish between assertions and other failures.
 * Test source
 * ------------------------
 *  - unit/assertion/assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Assert_Positive_Basic_KernelFail") {

#ifdef NDEBUG
  HipTest::HIP_SKIP_TEST("Assertions are disabled on this build.");
  return;
#endif

#if HT_AMD
  if (isAbortOnErrorEnabled()) {
    HipTest::HIP_SKIP_TEST(
        "Test incompatible with aborts enabled through HIP_SKIP_ABORT_ON_GPU_ERROR.");
    return;
  }
#endif

  const int num_blocks = 2;
  const int num_threads = 16;
  int* d_a;
  HIP_CHECK(hipMalloc(&d_a, sizeof(int)));

  AssertFailKernel<<<num_blocks, num_threads, 0, 0>>>(d_a);

#if HT_AMD
  HIP_CHECK_ERROR(hipDeviceSynchronize(), hipErrorLaunchFailure);
  HIP_CHECK_ERROR(hipFree(d_a), hipErrorLaunchFailure);
#else
  HIP_CHECK_ERROR(hipDeviceSynchronize(), hipErrorAssert);
  HIP_CHECK_ERROR(hipFree(d_a), hipErrorAssert);
#endif
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
