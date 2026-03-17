/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/device_functions.h>


__global__ void gpu_ballot(unsigned int* device_ballot, unsigned Num_Warps_per_Block,
                           unsigned pshift) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int warp_num = threadIdx.x >> pshift;

  // TODO try to remove this
#ifdef __HIP_PLATFORM_AMD__
  atomicAdd(&device_ballot[warp_num + blockIdx.x * Num_Warps_per_Block],
            __popcll(__ballot(tid - 245)));
#else
  unsigned mask = 0xFFFFFFFF;
  atomicAdd(&device_ballot[warp_num + blockIdx.x * Num_Warps_per_Block],
            __popc(__ballot_sync(mask, tid - 245)));
#endif
}

TEST_CASE(Unit_ballot) {
  unsigned warpSize, pshift;
  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  warpSize = devProp.warpSize;

  int w = warpSize;
  pshift = 0;
  while (w >>= 1) ++pshift;

  unsigned int Num_Threads_per_Block = 512;
  unsigned int Num_Blocks_per_Grid = 1;
  unsigned int Num_Warps_per_Block = Num_Threads_per_Block / warpSize;
  unsigned int Num_Warps_per_Grid = (Num_Threads_per_Block * Num_Blocks_per_Grid) / warpSize;
  unsigned int* host_ballot = (unsigned int*)malloc(Num_Warps_per_Grid * sizeof(unsigned int));
  unsigned int* device_ballot;
  HIP_CHECK(hipMalloc((void**)&device_ballot, Num_Warps_per_Grid * sizeof(unsigned int)));
  int divergent_count = 0;
  for (unsigned i = 0; i < Num_Warps_per_Grid; i++) host_ballot[i] = 0;

  HIP_CHECK(hipMemcpy(device_ballot, host_ballot, Num_Warps_per_Grid * sizeof(unsigned int),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(gpu_ballot, dim3(Num_Blocks_per_Grid), dim3(Num_Threads_per_Block), 0, 0,
                     device_ballot, Num_Warps_per_Block, pshift);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(host_ballot, device_ballot, Num_Warps_per_Grid * sizeof(unsigned int),
                      hipMemcpyDeviceToHost));

  for (unsigned i = 0; i < Num_Warps_per_Grid; i++) {
    if ((host_ballot[i] == 0) || (host_ballot[i] / warpSize == warpSize)) {
      INFO("Warp: " << i << " is convergent - predicate true for " << (host_ballot[i] / warpSize)
                    << " threads");
    } else {
      INFO("Warp: " << i << " is divergent - predicate true for " << (host_ballot[i] / warpSize)
                    << " threads");
      divergent_count++;
    }
  }

  HIP_CHECK(hipFree(device_ballot));
  free(host_ballot);
  REQUIRE(divergent_count == 1);
}
