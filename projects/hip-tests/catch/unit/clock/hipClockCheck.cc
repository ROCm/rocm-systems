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

#include <cstring>
#include <numeric>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip/hip_ext.h>

/**
 * @addtogroup clock clock
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for clock, clock64 and wall_clock64 APIs
 */

// Any sort of wait based on clock cycles will be inaccurate give how modern GPUs clock themselves.
// What clock functions should exhibit is forward progress of the clock ticks.
// What we measure here is the start tick should be smaller than the end tick.
// We do some primitive math in the middle.

__device__ float reduce_32_elements(float* in) {
  auto val = in[threadIdx.x];
  val += __shfl_down(val, 16);
  val += __shfl_down(val, 8);
  val += __shfl_down(val, 4);
  val += __shfl_down(val, 2);
  val += __shfl_down(val, 1);
  return val;
}

__global__ void reduce_c64(long long* start, long long* end, float* in /* 32 sized input */,
                           float* out /* single sized output*/) {
  if (threadIdx.x == 0) {
    *start = clock64();
  }

  // do not reorder
  __threadfence();
  auto val = reduce_32_elements(in);
  __threadfence();

  if (threadIdx.x == 0) {
    *out = val;
    *end = clock64();
  }
}

__global__ void reduce_c(long long* start, long long* end, float* in /* 32 sized input */,
                         float* out /* single sized output*/) {
  if (threadIdx.x == 0) {
    *start = clock();
  }

  // do not reorder
  __threadfence();
  auto val = reduce_32_elements(in);
  __threadfence();

  if (threadIdx.x == 0) {
    *out = val;
    *end = clock();
  }
}

__global__ void reduce_wc64(long long* start, long long* end, float* in /* 32 sized input */,
                            float* out /* single sized output*/) {
  if (threadIdx.x == 0) {
    *start = wall_clock64();
  }

  // do not reorder
  __threadfence();
  auto val = reduce_32_elements(in);
  __threadfence();

  if (threadIdx.x == 0) {
    *out = val;
    *end = wall_clock64();
  }
}

void execute_clock_kernels(void (*kernel)(long long*, long long*, float*, float*)) {
  constexpr size_t size = 32; /* Do not change this, the math in kernel is done for 32 elements */
  float *d_in{}, *d_out{}, out{};
  long long *d_clock_start{}, *d_clock_end{}, clock_start{}, clock_end{};
  std::vector<float> in(size, 0.0f);

  for (size_t i = 0; i < size; i++) {
    in[i] = i + 1;
  }
  auto cpu_result = std::accumulate(in.begin(), in.end(), 0.0f);

  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
  HIP_CHECK(hipMalloc(&d_clock_start, sizeof(long long)));
  HIP_CHECK(hipMalloc(&d_clock_end, sizeof(long long)));

  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));
  HIP_CHECK(hipMemset(d_clock_start, 0, sizeof(long long)));
  HIP_CHECK(hipMemset(d_clock_end, 0, sizeof(long long)));

  hipLaunchKernelGGL(kernel, 1, size, 0, nullptr, d_clock_start, d_clock_end, d_in, d_out);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(&clock_start, d_clock_start, sizeof(long long), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&clock_end, d_clock_end, sizeof(long long), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&out, d_out, sizeof(float), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_clock_start));
  HIP_CHECK(hipFree(d_clock_end));

  // Make sure the math happenned correctly
  INFO("sum(1.0f, 2.0f, ..., 32.0f) gpu result: " << out << " cpu: " << cpu_result);
  REQUIRE(out == cpu_result);

  // Measure the clock progress
  // There can be two scenarios:
  // 1) clock_start < clock_end : which we expect
  // 2) clock_start > clock_end : which means clock warped around, but chances of that happening is
  // really low
  INFO("Clock start: " << clock_start << " end: " << clock_end);
  REQUIRE(clock_start < clock_end);
}

TEST_CASE("Unit_hipClock64_Positive_Basic") {
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock64() function on gfx11 devices!");
    return;
  }

  execute_clock_kernels(reduce_c64);
}

/**
 * Test Description
 * ------------------------
 *  - Launches two kernels that run for a specified amount of time passed as a kernel argument by
 * using device function clock. Kernel execution time is calculated through elapsed time between
 * the start and end event, and calculated time is compared with passed time values.
 * Test source
 * ------------------------
 *  - catch/unit/clock/hipClockCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipClock_Positive_Basic") {
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock() function on gfx11 devices!");
    return;
  }

  execute_clock_kernels(reduce_c);
}

TEST_CASE("Unit_hipWallClock64_Positive_Basic") { execute_clock_kernels(reduce_wc64); }

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */