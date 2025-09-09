/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>


/**
 * @addtogroup coalesced_group thread_block_tile
 * @{
 * @ingroup CooperativeGroupTest
 * meta_group_size()
 * meta_group_rank()
 */

using namespace cooperative_groups;

constexpr auto total_elem = 1 << 16;
constexpr auto block_size = 256;
constexpr auto test_size = 32;

static __global__ void kernel_coalesced_grp(int* mgrpSize, int* mgrpRank) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % 2 == 0) {
    coalesced_group threadBlockCGTy = coalesced_threads();
    mgrpSize[id] = threadBlockCGTy.meta_group_size();
    mgrpRank[id] = threadBlockCGTy.meta_group_rank();
  } else {
    coalesced_group threadBlockCGTx = coalesced_threads();
    mgrpSize[id] = threadBlockCGTx.meta_group_size();
    mgrpRank[id] = threadBlockCGTx.meta_group_rank();
  }
}

static __global__ void kernel_tiledgrp_threadblk(int* mgrpSize, int* mgrpRank) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  thread_block_tile<test_size> tiledGr = tiled_partition<test_size>(this_thread_block());
  mgrpSize[id] = tiledGr.meta_group_size();
  mgrpRank[id] = tiledGr.meta_group_rank();
}

/**
 * Test Description
 * ------------------------
 *    - Verify the values returned by meta_group_size() and
 * meta_group_rank() for thread block tile with thread block
 * group as parent.
 * ------------------------
 *    - catch\unit\cooperativeGrps\coalesced_tiled_groups_metagrp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_tiled_groups_metagrp_basic") {
  int *mgrpSize_d = nullptr, *mgrpRank_d = nullptr;
  int *mgrpSize_h = nullptr, *mgrpRank_h = nullptr;
  mgrpSize_h = new int[total_elem];
  REQUIRE(mgrpSize_h != nullptr);
  mgrpRank_h = new int[total_elem];
  REQUIRE(mgrpRank_h != nullptr);

  HIP_CHECK(hipMalloc(&mgrpSize_d, total_elem * sizeof(int)));
  HIP_CHECK(hipMalloc(&mgrpRank_d, total_elem * sizeof(int)));
  SECTION("Parent Group = thread block group") {
    hipLaunchKernelGGL(kernel_tiledgrp_threadblk, total_elem / block_size, block_size, 0, 0,
                       mgrpSize_d, mgrpRank_d);
    HIP_CHECK(hipMemcpy(mgrpRank_h, mgrpRank_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mgrpSize_h, mgrpSize_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < total_elem; i++) {
      REQUIRE(mgrpRank_h[i] >= 0);
      REQUIRE(mgrpRank_h[i] < (block_size / test_size));
      REQUIRE(mgrpSize_h[i] == (block_size / test_size));
    }
  }
  HIP_CHECK(hipFree(mgrpSize_d));
  HIP_CHECK(hipFree(mgrpRank_d));
  delete[] mgrpSize_h;
  delete[] mgrpRank_h;
}

/**
 * Test Description
 * ------------------------
 *    - Verify the values returned by meta_group_size() and
 * meta_group_rank() for Coalesced Groups.
 * ------------------------
 *    - catch\unit\cooperativeGrps\coalesced_tiled_groups_metagrp.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_coalesced_groups_metagrp_basic") {
  int *mgrpSize_d = nullptr, *mgrpRank_d = nullptr;
  int *mgrpSize_h = nullptr, *mgrpRank_h = nullptr;
  mgrpSize_h = new int[total_elem];
  REQUIRE(mgrpSize_h != nullptr);
  mgrpRank_h = new int[total_elem];
  REQUIRE(mgrpRank_h != nullptr);

  HIP_CHECK(hipMalloc(&mgrpSize_d, total_elem * sizeof(int)));
  HIP_CHECK(hipMalloc(&mgrpRank_d, total_elem * sizeof(int)));

  hipLaunchKernelGGL(kernel_coalesced_grp, total_elem / block_size, block_size, 0, 0, mgrpSize_d,
                     mgrpRank_d);
  HIP_CHECK(hipMemcpy(mgrpRank_h, mgrpRank_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(mgrpSize_h, mgrpSize_d, total_elem * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < total_elem; i++) {
    REQUIRE(mgrpRank_h[i] == 0);
    REQUIRE(mgrpSize_h[i] == 1);
  }
  HIP_CHECK(hipFree(mgrpSize_d));
  HIP_CHECK(hipFree(mgrpRank_d));
  delete[] mgrpSize_h;
  delete[] mgrpRank_h;
}

/**
 * End doxygen group CooperativeGroupTest.
 * @}
 */
