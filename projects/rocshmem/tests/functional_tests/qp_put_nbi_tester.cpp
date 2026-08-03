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

#include "qp_put_nbi_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include "gda/context_gda_device.hpp"
#include "gda/queue_pair.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * One-way non-blocking put using QP internals directly.  Pipelines multiple
 * put_nbi_single calls and quiets only at batch boundaries.
 *****************************************************************************/
__global__ void QpPutNbiTest(int loop, int skip, long long int *start_time,
                             long long int *end_time, char *source,
                             char *dest, size_t size, int batch,
                             ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  if (threadIdx.x == 0) {
    GDAContext *gda_ctx = reinterpret_cast<GDAContext *>(ctx.ctx_opaque);
    int pe = rocshmem_ctx_my_pe(ctx);
    int target = 1 - pe;

    QueuePair &qp = gda_ctx->qps[target];

    uintptr_t local_base =
        reinterpret_cast<uintptr_t>(gda_ctx->base_heap[pe]);
    uintptr_t remote_base =
        reinterpret_cast<uintptr_t>(gda_ctx->base_heap[target]);

    int wg_id = hipBlockIdx_x;
    int start_slot = (batch - (skip % batch)) % batch;

    // Drain all setup loads from HBM before entering the timed loop.
    __builtin_amdgcn_s_waitcnt(0);

    for (int i = 0; i < loop + skip; i++) {
      int slot = (start_slot + i) % batch;

      if (slot == 0) {
        qp.quiet_single();
        if (i == skip) {
          start_time[wg_id] = wall_clock64();
        }
      }

      uintptr_t d_offset =
          reinterpret_cast<uintptr_t>(dest + size * slot) - local_base;
      void *remote_addr = reinterpret_cast<void *>(remote_base + d_offset);

      qp.put_nbi_single(remote_addr, source, size, true);
    }

    qp.quiet_single();
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
QpPutNbiTester::QpPutNbiTester(TesterArguments args) : Tester(args) {
  s_buf = (char *)alloc_test_buffer(max_msg_size);
  r_buf = (char *)alloc_test_buffer(max_msg_size * batch_size);
}

QpPutNbiTester::~QpPutNbiTester() {
  free_test_buffer(s_buf);
  free_test_buffer(r_buf);
}

void QpPutNbiTester::resetBuffers(size_t size) {
  memset(s_buf, 0xAB, size);
  memset(r_buf, 0, size * batch_size);
}

void QpPutNbiTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(QpPutNbiTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, s_buf, r_buf, size,
                     batch_size, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void QpPutNbiTester::verifyResults([[maybe_unused]] size_t size) {}
