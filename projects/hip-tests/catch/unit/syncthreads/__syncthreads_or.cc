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
#include <resource_guards.hh>

#include "__syncthreads_or_negative_kernels_rtc.hh"
#include "syncthreads_common.hh"

/**
 * @addtogroup __syncthreads_or __syncthreads_or
 * @{
 * @ingroup SyncthreadsTest
 */

/**
 * Test Description
 * ------------------------
 *    - Basic synchronization test for `__syncthreads_or`.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads_or.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___syncthreads_or_Positive_Basic") {
  const auto kGridSize = 2;
  const auto kBlockSize = GENERATE(13, 32, 64, 513);

  LinearAllocGuard<int> out_alloc(LinearAllocs::hipMallocManaged, sizeof(int) * kGridSize);

  HipTest::launchKernel(SyncthreadsKernel<SyncthreadsKind::kOr>, kGridSize, kBlockSize,
                        sizeof(int) * kBlockSize, nullptr, out_alloc.ptr());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kGridSize; ++i) {
    REQUIRE(out_alloc.host_ptr()[i] == kBlockSize * (kBlockSize + 1) / 2);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test `__syncthreads_or` with 0 as the predicate for all threads.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads_or.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___syncthreads_or_Positive_Predicate_Zero") {
  const auto kGridSize = 2;
  const auto kBlockSize = GENERATE(13, 32, 64, 513);

  LinearAllocGuard<int> out_alloc(LinearAllocs::hipMallocManaged,
                                  sizeof(int) * kGridSize * kBlockSize);

  HipTest::launchKernel(SyncthreadsZeroKernel<SyncthreadsKind::kOr>, kGridSize, kBlockSize, 0,
                        nullptr, out_alloc.ptr());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kGridSize * kBlockSize; ++i) {
    REQUIRE(out_alloc.host_ptr()[i] == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test `__syncthreads_or` with 1 as the predicate for all threads.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads_or.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___syncthreads_or_Positive_Predicate_One") {
  const auto kGridSize = 2;
  const auto kBlockSize = GENERATE(13, 32, 64, 513);

  LinearAllocGuard<int> out_alloc(LinearAllocs::hipMallocManaged,
                                  sizeof(int) * kGridSize * kBlockSize);

  HipTest::launchKernel(SyncthreadsOneKernel<SyncthreadsKind::kOr>, kGridSize, kBlockSize, 0,
                        nullptr, out_alloc.ptr());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kGridSize * kBlockSize; ++i) {
    REQUIRE(out_alloc.host_ptr()[i] == 1);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test `__syncthreads_or` with 0 as the predicate for even threads, and 1 as the predicate for
 * odd threads.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads_or.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___syncthreads_or_Positive_Predicate_OddEven") {
  const auto kGridSize = 2;
  const auto kBlockSize = GENERATE(13, 32, 64, 513);

  LinearAllocGuard<int> out_alloc(LinearAllocs::hipMallocManaged,
                                  sizeof(int) * kGridSize * kBlockSize);

  HipTest::launchKernel(SyncthreadsOddEvenKernel<SyncthreadsKind::kOr>, kGridSize, kBlockSize, 0,
                        nullptr, out_alloc.ptr());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kGridSize * kBlockSize; ++i) {
    REQUIRE(out_alloc.host_ptr()[i] == 1);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test `__syncthreads_or` with a negative predicate.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads_or.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___syncthreads_or_Positive_Predicate_Negative") {
  const auto kGridSize = 2;
  const auto kBlockSize = GENERATE(13, 32, 64, 513);

  LinearAllocGuard<int> out_alloc(LinearAllocs::hipMallocManaged,
                                  sizeof(int) * kGridSize * kBlockSize);

  HipTest::launchKernel(SyncthreadsNegativeKernel<SyncthreadsKind::kOr>, kGridSize, kBlockSize, 0,
                        nullptr, out_alloc.ptr());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kGridSize * kBlockSize; ++i) {
    REQUIRE(out_alloc.host_ptr()[i] == 1);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test `__syncthreads_or` with the thread ID as the predicate.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads_or.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___syncthreads_or_Positive_Predicate_Id") {
  const auto kGridSize = 2;
  const auto kBlockSize = GENERATE(13, 32, 64, 513);

  LinearAllocGuard<int> out_alloc(LinearAllocs::hipMallocManaged,
                                  sizeof(int) * kGridSize * kBlockSize);

  HipTest::launchKernel(SyncthreadsIdKernel<SyncthreadsKind::kOr>, kGridSize, kBlockSize, 0,
                        nullptr, out_alloc.ptr());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kGridSize * kBlockSize; ++i) {
    REQUIRE(out_alloc.host_ptr()[i] == 1);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Real-time compiles kernels that pass invalid arguments to `__syncthreads_or`.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads_or.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit___syncthreads_or_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  HIPRTC_CHECK(hiprtcCreateProgram(&program, kSyncthreadsOrSource, "__syncthreads_or_negative.cc",
                                   0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{2};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while (n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}

/**
 * End doxygen group SyncthreadsTest.
 * @}
 */
