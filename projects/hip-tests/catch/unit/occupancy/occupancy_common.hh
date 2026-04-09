/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>

template <typename F> void MaxPotentialBlockSize(F func, int maxThreadsPerBlock) {
  int gridSize = 0;
  int blockSize = 0;

  // Get potential blocksize
  HIP_CHECK(func(&gridSize, &blockSize));

  // Check if blockSize doesn't exceed maxThreadsPerBlock
  REQUIRE(gridSize > 0);
  REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= maxThreadsPerBlock);
  REQUIRE(gridSize * blockSize < static_cast<int64_t>(std::pow(2, 32)));
}

template <typename F> void MaxPotentialBlockSizeNegative(F func) {
  int blockSize = 0;
  int gridSize = 0;

  // Validate common arguments
  SECTION("gridSize is nullptr") {
    HIP_CHECK_ERROR(func(nullptr, &blockSize), hipErrorInvalidValue);
  }
  SECTION("blockSize is nullptr") {
    HIP_CHECK_ERROR(func(&gridSize, nullptr), hipErrorInvalidValue);
  }
}

template <typename F>
void MaxActiveBlocksPerMultiprocessor(F func, int blockSize, int maxThreadsPerMultiProcessor) {
  int numBlocks = 0;

  // Validate maximum active block pre multiprocessor
  HIP_CHECK(func(&numBlocks));

  // Check if numBlocks and blockSize are within limits
  REQUIRE(numBlocks > 0);
  REQUIRE((numBlocks * blockSize) <= maxThreadsPerMultiProcessor);
}

template <typename F> void MaxActiveBlocksPerMultiprocessorNegative(F func, int blockSize) {
  int numBlocks = 0;

  // Validate common arguments
  SECTION("numBlocks is nullptr") {
    HIP_CHECK_ERROR(func(nullptr, blockSize, 0), hipErrorInvalidValue);
  }
  SECTION("Block size is 0") { HIP_CHECK_ERROR(func(&numBlocks, 0, 0), hipErrorInvalidValue); }
}
