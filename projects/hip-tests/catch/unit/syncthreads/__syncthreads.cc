/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include "syncthreads_common.hh"

/**
 * @addtogroup __syncthreads __syncthreads
 * @{
 * @ingroup SyncthreadsTest
 */

/**
 * Test Description
 * ------------------------
 *    - Basic synchronization test for `__syncthreads`.
 *
 * Test source
 * ------------------------
 *    - unit/syncthreads/__syncthreads.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit___syncthreads_Positive_Basic) {
  const auto kGridSize = 2;
  const auto kBlockSize = GENERATE(13, 32, 64, 513);

  LinearAllocGuard<int> out_alloc(LinearAllocs::hipMallocManaged, sizeof(int) * kGridSize);

  HipTest::launchKernel(SyncthreadsKernel<SyncthreadsKind::kDefault>, kGridSize, kBlockSize,
                        sizeof(int) * kBlockSize, nullptr, out_alloc.ptr());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kGridSize; ++i) {
    REQUIRE(out_alloc.host_ptr()[i] == kBlockSize * (kBlockSize + 1) / 2);
  }
}

/**
 * End doxygen group SyncthreadsTest.
 * @}
 */
