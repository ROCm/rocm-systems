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
#include "launch_bounds_negative_kernels_rtc.hh"

/**
 * @addtogroup launch_bounds launch_bounds
 * @{
 * @ingroup DeviceLanguageTest
 * `__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)` -
 * allows the application to provide usage hints that influence the resources (primarily registers)
 * used by the generated code. It is a function attribute that must be attached to a global
 * function.
 */

constexpr int kMaxThreadsPerBlock = 128;
constexpr int kMinWarpsPerMultiprocessor = 2;

__launch_bounds__(kMaxThreadsPerBlock, kMinWarpsPerMultiprocessor) __global__
    void SumKernel(int* sum) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  atomicAdd(sum, tid);
}

template <bool out_of_bounds> void LaunchBoundsWrapper(const int threads_per_block) {
  auto block_size = GENERATE(1, 32, 128);
  int* A_d;
  int* A_h;
  int sum{0};

  A_h = static_cast<int*>(malloc(sizeof(int)));
  memset(A_h, 0, sizeof(int));
  HIP_CHECK(hipMalloc(&A_d, sizeof(int)));
  HIP_CHECK(hipMemcpy(A_d, A_h, sizeof(int), hipMemcpyHostToDevice));
  SumKernel<<<block_size, threads_per_block>>>(A_d);

  if constexpr (out_of_bounds) {
    if (threads_per_block < 0) {
      hipError_t err = hipGetLastError();
      if (err != hipErrorInvalidConfiguration && err != hipErrorInvalidValue) {
        REQUIRE(false);
      }
    } else {
#if HT_AMD
      HIP_CHECK_ERROR(hipGetLastError(), hipErrorLaunchFailure);
#else
      HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidValue);
#endif
    }
  } else {
    HIP_CHECK(hipGetLastError());
  }

  HIP_CHECK(hipMemcpy(A_h, A_d, sizeof(int), hipMemcpyDeviceToHost));

  if constexpr (!out_of_bounds) {
    for (int i = 0; i < threads_per_block * block_size; ++i) {
      sum += i;
    }
    REQUIRE(*A_h == sum);
  }

  free(A_h);
  HIP_CHECK(hipFree(A_d));
}

/**
 * Test Description
 * ------------------------
 *  - Executes simple addition kernel and validates results.
 *  - The number of threads per block used to launch the kernel
 *    are complied with the `__launch_bounds__`:
 *      -# Number of threads per block are less than or equal to the configured maximum value.
 *      -# Different values are assigned and kernel functionality is validated.
 * Test source
 * ------------------------
 *  - unit/launch_bounds/launch_bounds.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Kernel_Launch_bounds_Positive_Basic") {
  auto threads_per_block = GENERATE(1, kMaxThreadsPerBlock / 2, kMaxThreadsPerBlock);
  LaunchBoundsWrapper<false>(threads_per_block);
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the kernels will not be launched if the number of threads
 *    per block is larger than configured with `__launch_bounds__`:
 *    -# Expected output:
 *      - return `hipErrorLaunchFailure` on AMD.
 *      - return `hipErrorInvalidValue` on NVIDIA.
 * Test source
 * ------------------------
 *  - unit/launch_bounds/launch_bounds.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Kernel_Launch_bounds_Negative_OutOfBounds") {
  auto threads_per_block =
      GENERATE(-1 * kMaxThreadsPerBlock, -1, kMaxThreadsPerBlock + 1, 2 * kMaxThreadsPerBlock);
  LaunchBoundsWrapper<true>(threads_per_block);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# Compiles kernels that are not created appropriately:
 *      - Maximum number of threads is 0
 *      - Maximum number of threads is negative
 *      - Minimum number of warps is negative
 *      - Maximum number of threads is not integer value
 *      - Mimimum number of warps is not integer value
 *    -# Expected output: compiler error
 *  - Uses RTC for compilation.
 * Test source
 * ------------------------
 *  - unit/launch_bounds/launch_bounds.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Kernel_Launch_bounds_Negative_Parameters_RTC") {
  hiprtcProgram program{};

#if HT_AMD
  const auto program_source = GENERATE(kMaxThreadsZero, kMaxThreadsNegative, kMinWarpsNegative,
                                       kMaxThreadsNotInt, kMinWarpsNotInt);
#else
  // Aligned with CUDA behavior and expected behavior on NVIDIA
  const auto program_source = GENERATE(kMaxThreadsNotInt, kMinWarpsNotInt);
#endif

  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "launch_bounds_negative.cc", 0,
                                   nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log.
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while (n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  REQUIRE(error_count > 0);
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
