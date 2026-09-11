/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Testcase Scenarios :
Unit_hipOccupancy_Positive_SharedMemPerMultiprocessorConsistency - Validate that the shared memory
fields of hipDeviceProp_t agree with each other. On targets with a WGP (2 CUs sharing one LDS pool)
the per multiprocessor pool is larger than the per block limit, and both per multiprocessor fields
have to report that same pool.
Unit_hipOccupancy_Positive_OccupancyMatchesSharedMemPerMultiprocessor - Validate that the occupancy
reported by hipOccupancyMaxActiveBlocksPerMultiprocessor never claims more LDS than
sharedMemPerMultiprocessor advertises, i.e. that the API and the device properties describe the
same pool in the same unit.
Unit_hipOccupancy_Positive_ZeroLdsMatchesThreadLimit - Validate that a kernel which uses no LDS is
limited only by maxThreadsPerMultiProcessor, i.e. that the occupancy is reported per the same
multiprocessor that multiProcessorCount counts.

Note: the LDS allocation granularity is not exposed through hipDeviceProp_t, so these tests avoid
depending on it. They verify that the occupancy API and the device properties stay consistent with
each other, which no single field can be checked against on its own. Verifying that the pool is
physically correct requires measuring co-residency on the device.
*/

#include "occupancy_common.hh"

#include <climits>
#include <limits>

static __global__ void dynLdsKernel(unsigned int* out) {
  extern __shared__ unsigned int smem[];
  if (threadIdx.x == UINT_MAX) {  // never taken; keeps smem live
    out[0] = smem[0];
  }
}

HIP_TEST_CASE(Unit_hipOccupancy_Positive_SharedMemPerMultiprocessorConsistency) {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  REQUIRE(devProp.sharedMemPerBlock > 0);
  REQUIRE(devProp.sharedMemPerMultiprocessor > 0);

  // A multiprocessor holds at least one maximally sized block, and its pool is a whole number of
  // per block limits (1x in CU mode, 2x when a WGP of 2 CUs is the multiprocessor).
  REQUIRE(devProp.sharedMemPerMultiprocessor >= devProp.sharedMemPerBlock);
  REQUIRE((devProp.sharedMemPerMultiprocessor % devProp.sharedMemPerBlock) == 0);

  // Both per multiprocessor fields describe the same pool and are filled in by separate code
  // paths, so they must not drift apart.
  REQUIRE(devProp.maxSharedMemoryPerMultiProcessor == devProp.sharedMemPerMultiprocessor);

}

HIP_TEST_CASE(Unit_hipOccupancy_Positive_OccupancyMatchesSharedMemPerMultiprocessor) {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  const int blockSize = devProp.warpSize * 2;
  REQUIRE(blockSize <= devProp.maxThreadsPerBlock);

  int prevNumBlocks = std::numeric_limits<int>::max();
  size_t prevLds = 0;

  // Sizing the request as a fraction of the advertised pool keeps this independent of the LDS
  // allocation granularity: rounding up can only ever reduce the block count, never raise it.
  // The fractions are walked from smallest to largest request so that the monotonicity check
  // below sees a strictly growing LDS size.
  for (const int fraction : {16, 8, 4, 3, 2}) {
    const size_t dynLds = devProp.sharedMemPerMultiprocessor / fraction;
    if (dynLds == 0 || dynLds > devProp.sharedMemPerBlock) {
      continue;  // a single block cannot allocate more than the per block limit
    }

    int numBlocks = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, reinterpret_cast<const void*>(dynLdsKernel), blockSize, dynLds));

    INFO("dynLds = " << dynLds << ", numBlocks = " << numBlocks);
    REQUIRE(numBlocks >= 1);

    // The blocks reported as co-resident must fit in the advertised pool. This fails if the
    // occupancy calculation and the reported property disagree about the size of a
    // multiprocessor's LDS.
    REQUIRE(static_cast<size_t>(numBlocks) * dynLds <= devProp.sharedMemPerMultiprocessor);

    // And they must still fit in the thread budget of the same multiprocessor.
    REQUIRE((numBlocks * blockSize) <= devProp.maxThreadsPerMultiProcessor);

    // Occupancy cannot rise as a block asks for more LDS.
    if (prevLds != 0 && dynLds > prevLds) {
      REQUIRE(numBlocks <= prevNumBlocks);
    }
    prevNumBlocks = numBlocks;
    prevLds = dynLds;
  }
}

HIP_TEST_CASE(Unit_hipOccupancy_Positive_ZeroLdsMatchesThreadLimit) {
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  const int blockSize = devProp.warpSize * 2;
  REQUIRE(blockSize <= devProp.maxThreadsPerBlock);

  if ((devProp.maxThreadsPerMultiProcessor % blockSize) != 0) {
    WARN("maxThreadsPerMultiProcessor is not a multiple of the block size; skipping");
    return;
  }

  int numBlocks = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, reinterpret_cast<const void*>(dynLdsKernel), blockSize, 0));

  // With no LDS in play the only limit left is the thread budget, so the two have to agree. They
  // are only comparable if both are expressed per the same multiprocessor, which is what this
  // checks: a kernel scheduled on a smaller unit than multiProcessorCount counts would report
  // proportionally fewer blocks here.
  INFO("blockSize = " << blockSize << ", numBlocks = " << numBlocks);
  REQUIRE((numBlocks * blockSize) == devProp.maxThreadsPerMultiProcessor);
}
