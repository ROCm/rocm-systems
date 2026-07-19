/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// BACKEND-DIFF: The whole hipOccupancyMaxPotentialBlockSize* family this file
// exercises is unavailable on the NVIDIA backend:
// hipOccupancyMaxPotentialBlockSizeVariableSMem and its with-flags sibling have
// no NVIDIA equivalent, and the plain/with-flags templated wrappers resolve to
// cudaOccupancyMaxPotentialBlockSize with a mismatched parameter list under CUDA
// 13.1, so the whole translation unit builds only on AMD. Parity would require
// the NVIDIA-backend occupancy wrappers to match the CUDA template signatures.
//
// PLATFORM-DIFF: hipOccupancyMaxPotentialBlockSizeWithFlags,
// hipOccupancyMaxPotentialBlockSizeVariableSMem, and
// hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags are not exported from the
// Windows HIP runtime (absent from amdhip.def.in), so every case here is an
// unresolved external at link time on Windows. The whole translation unit is
// therefore additionally gated off on Windows (compiling to an empty test binary,
// exactly as it already does on the NVIDIA backend). This platform gate can be
// removed once the Windows runtime exports these symbols.
#if HT_AMD && !defined(_WIN32)

namespace {
__global__ void OccupancyVariableKernel(int* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && output != nullptr) {
    *output = 1;
  }
}

// A block-size-to-dynamic-shared-memory functor. The variable-shared-memory
// occupancy helpers call this for candidate block sizes; returning zero models a
// kernel that uses no dynamic shared memory, keeping the query portable.
struct NoDynamicSmem {
  __host__ __device__ size_t operator()(int /*block_size*/) const { return 0; }
};
}  // namespace

// @asserts: hipOccupancyMaxPotentialBlockSizeWithFlags - returns a non-negative min grid size and positive block size; the reserved flag does not change the contract
HIP_TEST_CASE(Contract_OccupancyVariable_PotentialBlockSizeWithFlags_ReturnsUsableValues) {
  // The with-flags potential-block-size helper must return a usable grid/block
  // pair for a launchable kernel: a non-negative minimum grid size and a
  // positive block size. The flag argument is reserved and does not change the
  // portable contract.
  int min_grid_size = -1;
  int block_size = -1;
  HIP_CHECK(hipOccupancyMaxPotentialBlockSizeWithFlags(&min_grid_size, &block_size,
                                                       OccupancyVariableKernel, 0, 0,
                                                       hipOccupancyDefault));

  REQUIRE(min_grid_size >= 0);
  REQUIRE(block_size > 0);
}

// @asserts: hipOccupancyMaxPotentialBlockSizeWithFlags - under default flags suggests the same block size as the plain hipOccupancyMaxPotentialBlockSize
HIP_TEST_CASE(Contract_OccupancyVariable_PotentialBlockSizeWithFlags_MatchesPlainVariant) {
  // Under default flags the with-flags helper must agree with the plain
  // potential-block-size helper for the same kernel: the reserved flag argument
  // must not change the suggested block size.
  int plain_grid = -1;
  int plain_block = -1;
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&plain_grid, &plain_block, OccupancyVariableKernel, 0,
                                              0));
  REQUIRE(plain_block > 0);

  int flagged_grid = -1;
  int flagged_block = -1;
  HIP_CHECK(hipOccupancyMaxPotentialBlockSizeWithFlags(&flagged_grid, &flagged_block,
                                                       OccupancyVariableKernel, 0, 0,
                                                       hipOccupancyDefault));

  REQUIRE(flagged_block == plain_block);
}

// @asserts: hipOccupancyMaxPotentialBlockSizeVariableSMem - accepts a block-size-to-dynamic-smem functor and returns a non-negative min grid size and positive block size
HIP_TEST_CASE(Contract_OccupancyVariable_VariableSMem_ReturnsUsableValues) {
  // The variable-shared-memory helper must accept a block-size-to-dynamic-smem
  // functor and return a usable grid/block pair. A functor that always requests
  // zero dynamic shared memory keeps the query portable across devices.
  int min_grid_size = -1;
  int block_size = -1;
  HIP_CHECK(hipOccupancyMaxPotentialBlockSizeVariableSMem(
      &min_grid_size, &block_size, OccupancyVariableKernel, NoDynamicSmem{}));

  REQUIRE(min_grid_size >= 0);
  REQUIRE(block_size > 0);
}

// @asserts: hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags - accepts the smem functor and returns a non-negative min grid size and positive block size under the reserved default flag
HIP_TEST_CASE(Contract_OccupancyVariable_VariableSMemWithFlags_ReturnsUsableValues) {
  // The with-flags variable-shared-memory helper must behave like its non-flags
  // sibling for the reserved default flag: it accepts the functor and returns a
  // usable grid/block pair.
  int min_grid_size = -1;
  int block_size = -1;
  HIP_CHECK(hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
      &min_grid_size, &block_size, OccupancyVariableKernel, NoDynamicSmem{}, 0, hipOccupancyDefault));

  REQUIRE(min_grid_size >= 0);
  REQUIRE(block_size > 0);
}
#endif  // HT_AMD && !defined(_WIN32)
