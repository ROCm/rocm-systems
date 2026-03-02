/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "barrier_all_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void BarrierAllTest(int loop, int skip, long long int *start_time,
                               long long int *end_time, TestType type,
                               int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int t_id  = get_flat_block_id();
  int wg_id = get_flat_grid_id();
  int wf_id = t_id / wf_size;


  for (int i = 0; i < loop + skip; i++) {
    if (hipThreadIdx_x == 0 && i == skip) {
      start_time[wg_id] = wall_clock64();
    }

    if (is_block_zero_in_grid()) {
      switch (type) {
        case BarrierAllTestType:
          if(t_id == 0) {
            /**
              * The function `rocshmem_barrier_all` should be called from only
              * one thread within the grid to avoid undefined behavior.
              */
            rocshmem_barrier_all();
          }
          break;
        case WAVEBarrierAllTestType:
          if(wf_id == 0) {
            /**
              * The function `rocshmem_barrier_all_wave` should be called from only
              * one wavefront within the grid to avoid undefined behavior.
              */
            rocshmem_barrier_all_wave();
          }
          break;
        case WGBarrierAllTestType:
          /**
            * The function `rocshmem_barrier_all_wg` should be called from only
            * one workgroup within the grid to avoid undefined behavior.
            */
          rocshmem_barrier_all_wg();
          break;
        default:
          break;
      }
    }
    __syncthreads();
  }

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
BarrierAllTester::BarrierAllTester(TesterArguments args) : Tester(args) {}

BarrierAllTester::~BarrierAllTester() {}

void BarrierAllTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                    size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(BarrierAllTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, _type, wf_size);

  num_msgs = (loop + args.skip);
  num_timed_msgs = loop;
}

void BarrierAllTester::resetBuffers(size_t size) {}

void BarrierAllTester::verifyResults(size_t size) {}
