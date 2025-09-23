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
TEST_CASE("Unit___syncthreads_Positive_Basic") {
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
