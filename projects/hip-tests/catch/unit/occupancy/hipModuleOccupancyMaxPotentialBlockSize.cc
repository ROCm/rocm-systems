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
