/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_Positive_RangeValidation - Test correct
execution of hipModuleOccupancyMaxActiveBlocksPerMultiprocessor for diffrent parameter values
Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_Negative_Parameters - Test unsuccessful
execution of hipModuleOccupancyMaxActiveBlocksPerMultiprocessor api when parameters are invalid
*/
#include <hip_test_kernels.hh>
#include "occupancy_common.hh"

HIP_TEST_CASE(Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_Negative_Parameters) {
  hipModule_t module;
  hipFunction_t function;
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
        return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, function, blockSize,
                                                                  dynSharedMemPerBlk);
      },
      blockSize);

  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_Positive_RangeValidation) {
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
          return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, function, blockSize,
                                                                    0);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
  SECTION("dynSharedMemPerBlk = sharedMemPerBlock") {
    // Get potential blocksize
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function,
                                                      devProp.sharedMemPerBlock, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, devProp, &function](int* numBlocks) {
          return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, function, blockSize,
                                                                    devProp.sharedMemPerBlock);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }

  HIP_CHECK(hipModuleUnload(module));
}

/**
 * Test Description
 * ------------------------
 *  - This test case tests the behaviour of all Occupancy APIs
 *  - during the the stream capture.
 * Test source
 * ------------------------
 *  - unit/occupancy/hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.cc
 */
HIP_TEST_CASE(Unit_OccupancyAPIs_StreamCapture) {
  GENERATE_CAPTURE();

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "simple_kernel.code"));
  REQUIRE(module != nullptr);
  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, "SimpleKernel"));
  REQUIRE(function != nullptr);

  BEGIN_CAPTURE(stream);

  int gridSize = 0, blockSize = 0, numBlocks = 0;

  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(
      &gridSize, &blockSize, reinterpret_cast<const void*>(HipTest::vectorADD<int>), 0, 0));
  REQUIRE(gridSize > 0);
  REQUIRE(blockSize > 0);

  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, reinterpret_cast<const void*>(HipTest::vectorADD<int>), blockSize, 0));
  REQUIRE(numBlocks > 0);

  numBlocks = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &numBlocks, reinterpret_cast<const void*>(HipTest::vectorADD<int>), blockSize, 0, 0));
  REQUIRE(numBlocks > 0);

  gridSize = 0, blockSize = 0;
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, function, 0, 0));
  REQUIRE(gridSize > 0);
  REQUIRE(blockSize > 0);

  gridSize = 0;
  blockSize = 0;
  HIP_CHECK(
      hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize, &blockSize, function, 0, 0, 0));
  REQUIRE(gridSize > 0);
  REQUIRE(blockSize > 0);

  numBlocks = 0;
  HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, function, blockSize, 0));
  REQUIRE(numBlocks > 0);

  numBlocks = 0;
  HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, function,
                                                                        blockSize, 0, 0));
  REQUIRE(numBlocks > 0);

  END_CAPTURE(stream);

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipStreamDestroy(stream));
}
