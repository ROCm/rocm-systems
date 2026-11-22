/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipModuleOccupancyMaxPotentialBlockSize_Positive_RangeValidation - Test correct execution of
hipModuleOccupancyMaxPotentialBlockSize for diffrent parameter values
Unit_hipModuleOccupancyMaxPotentialBlockSize_Negative_Parameters - Test unsuccessful execution of
hipModuleOccupancyMaxPotentialBlockSize api when parameters are invalid
Unit_hipModuleOccupancyMaxPotentialBlockSize_Positive_LaunchBounds - Test that the suggested block
size never exceeds the kernel's __launch_bounds__ or the caller's limit, including bounds that are
not a multiple of the wavefront size, and that the kernel launches with it
*/
#include "occupancy_common.hh"

HIP_TEST_CASE(Unit_hipModuleOccupancyMaxPotentialBlockSize_Negative_Parameters) {
  hipModule_t module;
  hipFunction_t function;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  // Common negative tests
  MaxPotentialBlockSizeNegative([&function](int* gridSize, int* blockSize) {
    return hipModuleOccupancyMaxPotentialBlockSize(gridSize, blockSize, function, 0, 0);
  });

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleOccupancyMaxPotentialBlockSize_Positive_RangeValidation) {
  hipDeviceProp_t devProp;
  hipModule_t module;
  hipFunction_t function;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [&function](int* gridSize, int* blockSize) {
          return hipModuleOccupancyMaxPotentialBlockSize(gridSize, blockSize, function, 0, 0);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [&function, devProp](int* gridSize, int* blockSize) {
          return hipModuleOccupancyMaxPotentialBlockSize(
              gridSize, blockSize, function, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock);
        },
        devProp.maxThreadsPerBlock);
  }

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleOccupancyMaxPotentialBlockSize_Positive_LaunchBounds) {
  hipDeviceProp_t devProp;
  hipModule_t module;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));

  int* out = nullptr;
  HIP_CHECK(hipMalloc(&out, devProp.maxThreadsPerBlock * sizeof(int)));

  const auto bound = GENERATE(128, 100, 96);
  const std::string name = "BoundedKernel" + std::to_string(bound);
  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, name.c_str()));

  int maxThreads = 0;
  HIP_CHECK(hipFuncGetAttribute(&maxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
  REQUIRE(maxThreads == bound);

  // 0 means no limit; a limit above the kernel's maximum must not raise the suggestion above it.
  for (const int limit : {0, bound - 1, bound, 2 * bound, devProp.maxThreadsPerBlock}) {
    int gridSize = 0;
    int blockSize = 0;
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function, 0, limit));
    INFO(name << ": blockSizeLimit = " << limit << ", blockSize = " << blockSize);
    REQUIRE(gridSize > 0);
    REQUIRE(blockSize > 0);
    REQUIRE(blockSize <= maxThreads);
    if (limit != 0) {
      REQUIRE(blockSize <= limit);
    }

    // The suggestion has to be launchable.
    void* args[] = {&out};
    HIP_CHECK(
        hipModuleLaunchKernel(function, 1, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  // A block larger than the kernel's maximum cannot be resident at all.
  int numBlocks = -1;
  HIP_CHECK(
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, function, 2 * bound, 0));
  REQUIRE(numBlocks == 0);

  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipModuleUnload(module));
}
