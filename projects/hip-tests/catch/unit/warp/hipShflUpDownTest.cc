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

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip/hip_fp16.h>

const int size = 32;

template <typename T> __global__ void shflDownSum(T* a, int size) {
  T val = a[threadIdx.x];
  for (int i = size / 2; i > 0; i /= 2) {
    val += __shfl_down(val, i, size);
  }
  a[threadIdx.x] = val;
}

template <typename T> __global__ void shflUpSum(T* a, int size) {
  T val = a[threadIdx.x];
  for (int i = size / 2; i > 0; i /= 2) {
    val += __shfl_up(val, i, size);
  }
  a[threadIdx.x] = val;
}

template <typename T> __global__ void shflXorSum(T* a, int size) {
  T val = a[threadIdx.x];
  for (int i = size / 2; i > 0; i /= 2) {
    val += __shfl_xor(val, i, size);
  }
  a[threadIdx.x] = val;
}

static void getFactor(int* fact) { *fact = 101; }
static void getFactor(unsigned int* fact) { *fact = static_cast<unsigned int>(INT32_MAX) + 1; }
static void getFactor(float* fact) { *fact = 2.5; }
static void getFactor(double* fact) { *fact = 2.5; }
static void getFactor(__half* fact) { *fact = 2.5; }
static void getFactor(int64_t* fact) { *fact = 303; }
static void getFactor(uint64_t* fact) { *fact = static_cast<uint64_t>(__LONG_LONG_MAX__) + 1; }

template <typename T> T sum(T* a) {
  T cpuSum = 0;
  T factor;
  getFactor(&factor);
  for (int i = 0; i < size; i++) {
    a[i] = i + factor;
    cpuSum += a[i];
  }
  return cpuSum;
}

template <> __half sum(__half* a) {
  __half cpuSum = 0;
  __half factor;
  getFactor(&factor);
  for (int i = 0; i < size; i++) {
    a[i] = i + __half2float(factor);
    cpuSum = __half2float(cpuSum) + __half2float(a[i]);
  }
  return cpuSum;
}

template <typename T> bool compare(T gpuSum, T cpuSum) {
  if (gpuSum != cpuSum) {
    return true;
  }
  return false;
}

template <> bool compare(__half gpuSum, __half cpuSum) {
  if (__half2float(gpuSum) != __half2float(cpuSum)) {
    return true;
  }
  return false;
}

template <typename T> static void runTestShflUp() {
  const int size = 32;
  T a[size];
  T cpuSum = sum(a);
  T* d_a;
  HIP_CHECK(hipMalloc(&d_a, sizeof(T) * size));
  HIP_CHECK(hipMemcpy(d_a, &a, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflUpSum<T>, 1, size, 0, 0, d_a, size);
  HIP_CHECK(hipMemcpy(&a, d_a, sizeof(T) * size, hipMemcpyDefault));
  REQUIRE((compare(a[size - 1], cpuSum)) == 0);
  HIP_CHECK(hipFree(d_a));
}

template <typename T> static void runTestShflDown() {
  T a[size];
  T cpuSum = sum(a);
  T* d_a;
  HIP_CHECK(hipMalloc(&d_a, sizeof(T) * size));
  HIP_CHECK(hipMemcpy(d_a, &a, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflDownSum<T>, 1, size, 0, 0, d_a, size);
  HIP_CHECK(hipMemcpy(&a, d_a, sizeof(T) * size, hipMemcpyDefault));
  REQUIRE((compare(a[0], cpuSum)) == 0);
  HIP_CHECK(hipFree(d_a));
}

template <typename T> static void runTestShflXor() {
  T a[size];
  T cpuSum = sum(a);
  T* d_a;
  HIP_CHECK(hipMalloc(&d_a, sizeof(T) * size));
  HIP_CHECK(hipMemcpy(d_a, &a, sizeof(T) * size, hipMemcpyDefault));
  hipLaunchKernelGGL(shflXorSum<T>, 1, size, 0, 0, d_a, size);
  HIP_CHECK(hipMemcpy(&a, d_a, sizeof(T) * size, hipMemcpyDefault));
  REQUIRE((compare(a[0], cpuSum)) == 0);
  HIP_CHECK(hipFree(d_a));
}

/**
 * @addtogroup __shfl __shfl
 * @{
 * @ingroup ShflTest
 * `T __shfl_up(T var, unsigned int lane_delta, int width = warpSize)` -
 * Contains warp __shfl_up function
 */

/**
 * Test Description
 * ------------------------
 *    - Test case to verify __shfl_up warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflUpDownTest.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 *    - Gaurding this test against cuda with refernce to mentioned
 * ticket SWDEV-379177
 */

TEST_CASE("Unit_runTestShfl_up") {
  SECTION("runTestShflUp for int") { runTestShflUp<int>(); }
  SECTION("runTestShflUp for float") { runTestShflUp<float>(); }
  SECTION("runTestShflUp for double") { runTestShflUp<double>(); }
  SECTION("runTestShflUp for __half") { runTestShflUp<__half>(); }
  SECTION("runTestShflUp for int64_t") { runTestShflUp<int64_t>(); }
  SECTION("runTestShflUp for unsigned int") { runTestShflUp<unsigned int>(); }
  SECTION("runTestShflUp for uint64_t") { runTestShflUp<uint64_t>(); }
}
/**
 * End doxygen group __shfl.
 * @}
 */

/**
 * @addtogroup __shfl __shfl
 * @{
 * @ingroup ShflTest
 * `T __shfl_down(T var, unsigned int lane_delta, int width = warpSize)` -
 * Contains warp __shfl_down function
 */

/**
 * Test Description
 * ------------------------
 *    - Test case to verify __shfl_down warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflUpDownTest.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 *    - Gaurding this test against cuda with refernce to mentioned
 * ticket SWDEV-379177
 */

TEST_CASE("Unit_runTestShfl_Down") {
  SECTION("runTestShflDown for int") { runTestShflDown<int>(); }
  SECTION("runTestShflDown for float") { runTestShflDown<float>(); }
  SECTION("runTestShflDown for double") { runTestShflDown<double>(); }
  SECTION("runTestShflDown for __half") { runTestShflDown<__half>(); }
  SECTION("runTestShflDown for int64_t") { runTestShflDown<int64_t>(); }
  SECTION("runTestShflDown for unsigned int") { runTestShflDown<unsigned int>(); }
  SECTION("runTestShflDown for uint64_t") { runTestShflDown<uint64_t>(); }
}
/**
 * End doxygen group __shfl.
 * @}
 */

/**
 * @addtogroup __shfl __shfl
 * @{
 * @ingroup ShflTest
 * `T __shfl_xor(T var, int laneMask, int width=warpSize)` -
 * Contains warp __shfl_xor function
 */

/**
 * Test Description
 * ------------------------
 *    - Test case to verify __shfl_xor warp functions for different datatypes.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipShflUpDownTest.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 *    - Gaurding this test against cuda with refernce to mentioned
 * ticket SWDEV-379177
 */

TEST_CASE("Unit_runTestShfl_Xor") {
  SECTION("runTestShflXor for int") { runTestShflXor<int>(); }
  SECTION("runTestShflXor for float") { runTestShflXor<float>(); }
  SECTION("runTestShflXor for double") { runTestShflXor<double>(); }
  SECTION("runTestShflXor for __half") { runTestShflXor<__half>(); }
  SECTION("runTestShflXor for int64_t") { runTestShflXor<int64_t>(); }
  SECTION("runTestShflXor for unsigned int") { runTestShflXor<unsigned int>(); }
  SECTION("runTestShflXor for uint64_t") { runTestShflXor<uint64_t>(); }
}
/**
 * End doxygen group __shfl.
 * @}
 */
