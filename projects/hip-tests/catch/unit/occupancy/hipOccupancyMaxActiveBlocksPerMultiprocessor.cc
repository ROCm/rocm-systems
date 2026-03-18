/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Positive_RangeValidation - Test correct execution
of hipOccupancyMaxActiveBlocksPerMultiprocessor for diffrent parameter values
Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Positive_TemplateInvocation - Test correct
execution of hipOccupancyMaxActiveBlocksPerMultiprocessor template for diffrent parameter values
Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Negative_Parameters - Test unsuccessful execution
of hipOccupancyMaxActiveBlocksPerMultiprocessor api when parameters are invalid
*/
#include "occupancy_common.hh"
#include <limits>

static __global__ void f1(float* a) { *a = 1.0; }

template <typename T> static __global__ void f2(T* a) { *a = 1; }

TEST_CASE(Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Negative_Parameters) {
  int numBlocks = 0;
  int blockSize = 0;
  int gridSize = 0;

  // Get potential blocksize
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  // Common negative tests
  MaxActiveBlocksPerMultiprocessorNegative(
      [](int* numBlocks, int blockSize, size_t dynSharedMemPerBlk) {
        return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f1, blockSize,
                                                            dynSharedMemPerBlk);
      },
      blockSize);

  SECTION("Kernel function is NULL") {
    HIP_CHECK_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, NULL, blockSize, 0),
                    hipErrorInvalidDeviceFunction);
  }

  SECTION("Block size is 0 and dynSharedMemPerBlk is max") {
    const hipError_t ret = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, f1, 0, std::numeric_limits<std::size_t>::max());
    REQUIRE(ret != hipSuccess);
  }
}

TEST_CASE(Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Positive_RangeValidation) {
  hipDeviceProp_t devProp;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0") {
    // Get potential blocksize
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f1, blockSize, 0);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
  SECTION("dynSharedMemPerBlk = sharedMemPerBlock") {
    // Get potential blocksize
    HIP_CHECK(
        hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, devProp.sharedMemPerBlock, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, devProp](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f1, blockSize,
                                                              devProp.sharedMemPerBlock);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
}

TEST_CASE(Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Positive_TemplateInvocation) {
  hipDeviceProp_t devProp;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0") {
    // Get potential blocksize
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(&gridSize, &blockSize, f2, 0, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor<void (*)(int*)>(numBlocks, f2,
                                                                              blockSize, 0);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock") {
    // Get potential blocksize
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(&gridSize, &blockSize, f2,
                                                                devProp.sharedMemPerBlock, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, devProp](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor<void (*)(int*)>(
              numBlocks, f2, blockSize, devProp.sharedMemPerBlock);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
}
