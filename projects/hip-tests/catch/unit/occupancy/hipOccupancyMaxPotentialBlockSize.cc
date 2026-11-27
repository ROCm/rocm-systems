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
Unit_hipOccupancyMaxPotentialBlockSize_Positive_LaunchBounds - Test that the suggested block size
never exceeds the kernel's __launch_bounds__ or the caller's limit, including bounds and limits that
are not a multiple of the wavefront size, and that the kernel launches with it
*/
#include "occupancy_common.hh"

static __global__ void f1(float* a) { *a = 1.0; }

template <typename T> static __global__ void f2(T* a) { *a = 1; }

// 100 is not a multiple of any wavefront size and 96 is not a multiple of 64.
template <int Bound> static __global__ void __launch_bounds__(Bound) bounded(int* a) {
  a[blockIdx.x * blockDim.x + threadIdx.x] = 1;
}

template <int Bound> static void CheckLaunchBoundsSuggestion() {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  hipFuncAttributes attr;
  HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(bounded<Bound>)));
  REQUIRE(attr.maxThreadsPerBlock == Bound);

  int* out = nullptr;
  HIP_CHECK(hipMalloc(&out, devProp.maxThreadsPerBlock * sizeof(int)));

  // 0 means no limit; a limit above the kernel's maximum must not raise the suggestion above it.
  for (const int limit : {0, Bound - 1, Bound, 2 * Bound, devProp.maxThreadsPerBlock}) {
    int gridSize = 0;
    int blockSize = 0;
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, bounded<Bound>, 0, limit));
    INFO("bound = " << Bound << ", blockSizeLimit = " << limit << ", blockSize = " << blockSize);
    REQUIRE(gridSize > 0);
    REQUIRE(blockSize > 0);
    REQUIRE(blockSize <= Bound);
    if (limit != 0) {
      REQUIRE(blockSize <= limit);
    }

    // The suggestion has to be launchable.
    hipLaunchKernelGGL(bounded<Bound>, dim3(1), dim3(blockSize), 0, 0, out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  // A block larger than the kernel's maximum cannot be resident at all.
  int numBlocks = -1;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, bounded<Bound>, 2 * Bound, 0));
  REQUIRE(numBlocks == 0);

  HIP_CHECK(hipFree(out));
}

HIP_TEST_CASE(Unit_hipOccupancyMaxPotentialBlockSize_Negative_Parameters) {
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

HIP_TEST_CASE(Unit_hipOccupancyMaxPotentialBlockSize_Positive_RangeValidation) {
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

HIP_TEST_CASE(Unit_hipOccupancyMaxPotentialBlockSize_Positive_TemplateInvocation) {
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

HIP_TEST_CASE(Unit_hipOccupancyMaxPotentialBlockSize_Positive_LaunchBounds) {
  SECTION("__launch_bounds__(128)") { CheckLaunchBoundsSuggestion<128>(); }
  SECTION("__launch_bounds__(100)") { CheckLaunchBoundsSuggestion<100>(); }
  SECTION("__launch_bounds__(96)") { CheckLaunchBoundsSuggestion<96>(); }

  SECTION("no bounds, blockSizeLimit not a wavefront multiple") {
    for (const int limit : {100, 96, 33}) {
      int gridSize = 0;
      int blockSize = 0;
      HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, limit));
      INFO("blockSizeLimit = " << limit << ", blockSize = " << blockSize);
      REQUIRE(gridSize > 0);
      REQUIRE(blockSize > 0);
      REQUIRE(blockSize <= limit);
    }
  }
}
