/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Positive_RangeValidation - Test correct
execution of hipModuleOccupancyMaxPotentialBlockSizeWithFlags for diffrent parameter values
Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Negative_Parameters - Test unsuccessful
execution of hipModuleOccupancyMaxPotentialBlockSizeWithFlags api when parameters are invalid
*/
#include "occupancy_common.hh"

TEST_CASE(Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Negative_Parameters) {
  hipModule_t module;
  hipFunction_t function;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  HIPCHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));

  // Common negative tests
  MaxPotentialBlockSizeNegative([&function](int* gridSize, int* blockSize) {
    return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, function, 0, 0,
                                                            hipOccupancyDefault);
  });

  SECTION("Flag is invalid") {
    // Only default flag is supported
    HIP_CHECK_ERROR(
        hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize, &blockSize, function, 0, 0, 2),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipModuleUnload(module));
}

TEST_CASE(Unit_hipModuleOccupancyMaxPotentialBlockSizeWithFlags_Positive_RangeValidation) {
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
          return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(gridSize, blockSize, function, 0,
                                                                  0, hipOccupancyDefault);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [&function, devProp](int* gridSize, int* blockSize) {
          return hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
              gridSize, blockSize, function, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock,
              hipOccupancyDefault);
        },
        devProp.maxThreadsPerBlock);
  }

  HIP_CHECK(hipModuleUnload(module));
}
