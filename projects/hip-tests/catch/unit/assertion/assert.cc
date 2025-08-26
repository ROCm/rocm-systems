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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <csetjmp>
#include <csignal>

/**
 * @addtogroup assert assert
 * @{
 * @ingroup DeviceLanguageTest
 * `void assert(int expression)` -
 * Stops the kernel execution if expression is equal to zero.
 */

jmp_buf env_ignore_abort;
volatile int abort_raised_flag = 0;

void on_sigabrt(int signum) {
  signal(signum, SIG_DFL);
  abort_raised_flag = 1;
  longjmp(env_ignore_abort, 1);
}

void try_and_catch_abort(void (*func)()) {
  if (!setjmp(env_ignore_abort)) {
    signal(SIGABRT, &on_sigabrt);
    (*func)();
    signal(SIGABRT, SIG_DFL);
  }
}

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

template <bool should_abort> void LaunchAssertKernel() {
  const int num_blocks = 2;
  const int num_threads = 16;
  int* d_a;
  HIP_CHECK(hipMalloc(&d_a, sizeof(int)));

  if constexpr (should_abort) {
    AssertFailKernel<<<num_blocks, num_threads, 0, 0>>>(d_a);
#if HT_AMD
    HIP_CHECK(hipDeviceSynchronize());
#else
    HIP_CHECK_ERROR(hipDeviceSynchronize(), hipErrorAssert);
#endif
  } else {
    AssertPassKernel<<<num_blocks, num_threads, 0, 0>>>(d_a);
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipFree(d_a));
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
  try_and_catch_abort(&LaunchAssertKernel<false>);
  REQUIRE(abort_raised_flag == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Launches kernels with asserts that have an expression equal to 0.
 *  - Expects that SIGABRT is raised and kernels have been stopped on AMD.
 *    - The HIP runtime also aborts the host code, so this test case uses signal handlers
 *      to avoid host code abortion.
 *  - Expects that `hipErrorAssert` is returned from `hipDeviceSynchronize` on NVIDIA.
 *    - The host code is not aborted.
 * Test source
 * ------------------------
 *  - unit/assertion/assert.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Assert_Positive_Basic_KernelFail") {
  try_and_catch_abort(&LaunchAssertKernel<true>);
#if HT_AMD
  REQUIRE(abort_raised_flag == 1);
#else
  REQUIRE(abort_raised_flag == 0);
#endif
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
