/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_Positive_RangeValidation - Test
correct execution of hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags for diffrent
parameter values
Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_Negative_Parameters - Test
unsuccessful execution of hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags api when
parameters are invalid
*/
#include "occupancy_common.hh"

HIP_TEST_CASE(Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_Negative_Parameters) {
  hipModule_t module;
  hipFunction_t function;
  int numBlocks = 0;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  // Get potential blocksize
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function, 0, 0));

  // Common negative tests
  MaxActiveBlocksPerMultiprocessorNegative(
      [&function](int* numBlocks, int blockSize, size_t dynSharedMemPerBlk) {
        return hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            numBlocks, function, blockSize, dynSharedMemPerBlk, hipOccupancyDefault);
      },
      blockSize);

  SECTION("Flag is invalid") {
    // Only default flag is supported
    HIP_CHECK_ERROR(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                        &numBlocks, function, blockSize, 0, 2),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_Positive_RangeValidation) {
  hipDeviceProp_t devProp;
  hipModule_t module;
  hipFunction_t function;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0") {
    // Get potential blocksize
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function, 0, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, &function](int* numBlocks) {
          return hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
              numBlocks, function, blockSize, 0, hipOccupancyDefault);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
  SECTION("dynSharedMemPerBlk = sharedMemPerBlock") {
    // Get potential blocksize
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function,
                                                      devProp.sharedMemPerBlock, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, devProp, &function](int* numBlocks) {
          return hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
              numBlocks, function, blockSize, devProp.sharedMemPerBlock, hipOccupancyDefault);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }

  HIP_CHECK(hipModuleUnload(module));
}
