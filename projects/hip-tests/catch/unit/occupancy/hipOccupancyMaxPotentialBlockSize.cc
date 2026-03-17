/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipOccupancyMaxPotentialBlockSize_Positive_RangeValidation - Test correct execution of
hipOccupancyMaxPotentialBlockSize for diffrent parameter values
Unit_hipOccupancyMaxPotentialBlockSize_Positive_TemplateInvocation - Test correct execution of
hipOccupancyMaxPotentialBlockSize template for diffrent parameter values
Unit_hipOccupancyMaxPotentialBlockSize_Negative_Parameters - Test unsuccessful execution of
hipOccupancyMaxPotentialBlockSize api when parameters are invalid
*/
#include "occupancy_common.hh"

static __global__ void f1(float* a) { *a = 1.0; }

template <typename T> static __global__ void f2(T* a) { *a = 1; }

TEST_CASE(Unit_hipOccupancyMaxPotentialBlockSize_Negative_Parameters) {
  // Common negative tests
  MaxPotentialBlockSizeNegative([](int* gridSize, int* blockSize) {
    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f1, 0, 0);
  });

#if HT_AMD
#if 0  // EXSWHTEC-219
  SECTION("Kernel function is NULL") {
    int blockSize = 0;
    int gridSize = 0;
    // nvcc doesnt support kernelfunc(NULL) for api
    HIP_CHECK_ERROR(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, NULL, 0, 0),
                    hipErrorInvalidDeviceFunction);
  }
#endif
#endif
}

TEST_CASE(Unit_hipOccupancyMaxPotentialBlockSize_Positive_RangeValidation) {
  hipDeviceProp_t devProp;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f1, 0, 0);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [devProp](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize(
              gridSize, blockSize, f1, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock);
        },
        devProp.maxThreadsPerBlock);
  }
}

TEST_CASE(Unit_hipOccupancyMaxPotentialBlockSize_Positive_TemplateInvocation) {
  hipDeviceProp_t devProp;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(gridSize, blockSize, f2, 0, 0);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [devProp](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(
              gridSize, blockSize, f2, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock);
        },
        devProp.maxThreadsPerBlock);
  }
}
