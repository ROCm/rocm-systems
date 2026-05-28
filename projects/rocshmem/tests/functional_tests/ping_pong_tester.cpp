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

#include "ping_pong_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void PingPongTest(int loop, int skip, long long int *start_time,
                             long long int *end_time, int *r_buf,
                             ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int pe = rocshmem_ctx_my_pe(ctx);

  if (is_thread_zero_in_block()) {

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        start_time[wg_id] = wall_clock64();
      }

      if (pe == 0) {
        rocshmem_ctx_int_p(ctx, &r_buf[hipBlockIdx_x], i + 1, 1);
        rocshmem_int_wait_until(&r_buf[hipBlockIdx_x], ROCSHMEM_CMP_EQ,
                                 i + 1);
      } else {
        rocshmem_int_wait_until(&r_buf[hipBlockIdx_x], ROCSHMEM_CMP_EQ,
                                 i + 1);
        rocshmem_ctx_int_p(ctx, &r_buf[hipBlockIdx_x], i + 1, 0);
      }
    }
    end_time[wg_id] = wall_clock64();

    rocshmem_ctx_quiet(ctx);
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PingPongTester::PingPongTester(TesterArguments args) : Tester(args) {
  r_buf = (int *)rocshmem_malloc(sizeof(int) * args.num_wgs);
}

PingPongTester::~PingPongTester() { rocshmem_free(r_buf); }

void PingPongTester::resetBuffers(size_t size) {
  memset(r_buf, 0, sizeof(int) * args.num_wgs);
}

void PingPongTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(PingPongTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, r_buf,
                     _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void PingPongTester::verifyResults(size_t size) {}
