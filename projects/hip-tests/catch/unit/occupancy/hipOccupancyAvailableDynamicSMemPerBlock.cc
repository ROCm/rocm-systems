
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

/**
 * @brief Returns dynamic shared memory available per block when launching
 * numBlocks blocks on SM.
 *
 * @ingroup Occupancy
 * Returns in dynamicSmemSize the maximum size of dynamic shared memory
 * to allow numBlocks blocks per SM.
 *
 * @param [out] dynamicSmemSize Returned maximum dynamic shared memory.
 * @param [in]  f               Kernel function for which occupancy is
 * calculated.
 * @param [in]  numBlocks       Number of blocks to fit on SM
 * @param [in]  blockSize       Size of the block
 *
 * @return #hipSuccess, #hipErrorInvalidDevice, #hipErrorInvalidDeviceFunction,
 * #hipErrorInvalidValue, #hipErrorUnknown
 */

#include <hip_test_common.hh>
#include <iomanip>
#include <iostream>

constexpr size_t SIZE = 1024;

static __global__ void dynamicReverse(int *d, int n) {
  __shared__ int s[SIZE];

  int t = threadIdx.x;  // unique index it starts from 0
  int tr = n - t - 1;  //  it is reverse index of t, if t = 0, then tr = n-1

  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

static __global__ void f1(float *a) { *a = 1.0; }

/**
 * Test Description
 * ------------------------
 *  - This test will verify hipOccupancyAvailableDynamicSMemPerBlock api with
 *    invalid parameters
 * Test source
 * ------------------------
 * - occupancy/hipOccupancyAvailableDynamicSMemPerBlock.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipOccupancyAvailableDynamicSMemPerBlock_Negative") {
  size_t dynamicSmemSize;
  int numBlocks = 1;
  SECTION("Number of blocks are zero") {
    HIP_CHECK_ERROR(
        hipOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize, f1, 0, SIZE),
        hipErrorInvalidValue);
  }

  SECTION("Block size is zero") {
    HIP_CHECK_ERROR(hipOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize,
                                                             f1, numBlocks, 0),
                    hipErrorInvalidValue);
  }

#if HT_AMD
  SECTION("Invalid driver funtion") {
    HIP_CHECK_ERROR(hipOccupancyAvailableDynamicSMemPerBlock(
                        &dynamicSmemSize, NULL, numBlocks, SIZE),
                    hipErrorInvalidDeviceFunction);
  }

  SECTION("dynamicSmemSize is null") {
    HIP_CHECK_ERROR(
        hipOccupancyAvailableDynamicSMemPerBlock(nullptr, f1, numBlocks, SIZE),
        hipErrorInvalidValue);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - This test will verify funtionality of
 * hipOccupancyAvailableDynamicSMemPerBlock api Test source
 * ------------------------
 * - occupancy/hipOccupancyAvailableDynamicSMemPerBlock.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipOccupancyAvailableDynamicSMemPerBlock_Positive") {
  size_t dynamicSmemSize = 0;
  int numBlocks = 1;
  int inputArray[SIZE], expectedOutput[SIZE], actualOutput[SIZE];

  for (int i = 0; i < SIZE; i++) {
    inputArray[i] = i;
    expectedOutput[i] = SIZE - i - 1;  // Expected reversed array
    actualOutput[i] = 0;
  }

  int *deviceArray{};  // Device pointer
  HIP_CHECK(
      hipMalloc(&deviceArray, SIZE * sizeof(int)));  // Allocate device memory

  HIP_CHECK(hipMemcpy(deviceArray, inputArray, SIZE * sizeof(int),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipOccupancyAvailableDynamicSMemPerBlock(
      &dynamicSmemSize, dynamicReverse, numBlocks, SIZE));
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  INFO("Available Dynamic shared memory size : "
       << dynamicSmemSize
       << ", Dynamic shared memory calculated from device properties : "
       << devProp.sharedMemPerBlock - SIZE * sizeof(int));
  REQUIRE(dynamicSmemSize == devProp.sharedMemPerBlock - SIZE * sizeof(int));
  dynamicReverse<<<numBlocks, SIZE, SIZE * sizeof(int)>>>(deviceArray, SIZE);

  HIP_CHECK(hipMemcpy(actualOutput, deviceArray, SIZE * sizeof(int),
                      hipMemcpyDeviceToHost));
  // Verify results
  for (int i = 0; i < SIZE; i++) {
    INFO("Results mismatched : actualOutput[" << i << "]!=expectedOutput[" << i
                                              << "],(" << actualOutput[i] << ","
                                              << expectedOutput[i] << ")");
    REQUIRE(actualOutput[i] == expectedOutput[i]);
  }
}

/**
 * End doxygen group occupancyTest.
 * @}
 */
