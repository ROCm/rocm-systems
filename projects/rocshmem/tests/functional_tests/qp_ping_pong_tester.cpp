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

#include "qp_ping_pong_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include "gda/context_gda_device.hpp"
#include "gda/queue_pair.hpp"
#include "assembly.hpp"
#include "constmem.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * Bypass the public rocSHMEM API and call QueuePair methods directly.
 *
 * op_type selects the put method:
 *   0 — put_nbi_single (4B immediate inline), spin on data value
 *   1 — put_nbi_single from symmetric buffer (4B), spin on data value
 *   2 — put_nbi_single (variable size) + atomic_nofetch_single for signal
 *
 * Only thread 0 is active (w1z1 pattern).
 *****************************************************************************/
__global__ void QpPingPongTest(int loop, int skip, long long int *start_time,
                               long long int *end_time, int *r_buf,
                               char *data_s_buf, char *data_r_buf,
                               uint64_t *sig_addr, size_t size,
                               unsigned op_type,
                               ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  if (threadIdx.x == 0) {
    GDAContext *gda_ctx = reinterpret_cast<GDAContext *>(ctx.ctx_opaque);
    int pe = constmem.my_pe;
    int target = 1 - pe;

    QueuePair &qp = gda_ctx->qps[target];

    uintptr_t local_base =
        reinterpret_cast<uintptr_t>(gda_ctx->base_heap[pe]);
    uintptr_t remote_base =
        reinterpret_cast<uintptr_t>(gda_ctx->base_heap[target]);

    uintptr_t r_buf_offset =
        reinterpret_cast<uintptr_t>(&r_buf[0]) - local_base;
    void *remote_r_buf = reinterpret_cast<void *>(remote_base + r_buf_offset);

    uintptr_t data_r_offset =
        reinterpret_cast<uintptr_t>(data_r_buf) - local_base;
    void *remote_data_r = reinterpret_cast<void *>(remote_base + data_r_offset);

    uintptr_t sig_offset =
        reinterpret_cast<uintptr_t>(sig_addr) - local_base;
    void *remote_sig = reinterpret_cast<void *>(remote_base + sig_offset);

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        start_time[0] = wall_clock64();
      }

      int val = i + 1;

      if (op_type <= 1) {
        void *src = (op_type == 0) ? static_cast<void *>(&val)
                                   : static_cast<void *>(data_s_buf);
        if (op_type == 1) {
          *reinterpret_cast<int *>(data_s_buf) = val;
        }
        if (pe == 0) {
          qp.put_nbi_single(remote_r_buf, src, sizeof(int), true);
          while (uncached_load(&r_buf[0]) != val) {}
        } else {
          while (uncached_load(&r_buf[0]) != val) {}
          qp.put_nbi_single(remote_r_buf, src, sizeof(int), true);
        }
      } else {
        uint64_t expected = static_cast<uint64_t>(i + 1);
        if (pe == 0) {
          qp.put_nbi_single(remote_data_r, data_s_buf, size, true);
          qp.quiet_single();
          qp.atomic_nofetch_single(remote_sig, 1);
          while (uncached_load(sig_addr) < expected) {}
        } else {
          while (uncached_load(sig_addr) < expected) {}
          qp.put_nbi_single(remote_data_r, data_s_buf, size, true);
          qp.quiet_single();
          qp.atomic_nofetch_single(remote_sig, 1);
        }
      }
    }
    end_time[0] = wall_clock64();

    qp.quiet_single();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
QpPingPongTester::QpPingPongTester(TesterArguments args) : Tester(args) {
  r_buf = (int *)alloc_test_buffer(sizeof(int) * args.num_wgs);
  data_s_buf = (char *)alloc_test_buffer(max_msg_size);
  data_r_buf = (char *)alloc_test_buffer(max_msg_size);
  sig_addr = (uint64_t *)alloc_test_buffer(sizeof(uint64_t));
  rtt_factor = 2;
  bw_factor = 2;
}

QpPingPongTester::~QpPingPongTester() {
  free_test_buffer(r_buf);
  free_test_buffer(data_s_buf);
  free_test_buffer(data_r_buf);
  free_test_buffer(sig_addr);
}

void QpPingPongTester::resetBuffers(size_t size) {
  memset(r_buf, 0, sizeof(int) * args.num_wgs);
  memset(data_s_buf, 0xAB, size);
  memset(data_r_buf, 0, size);
  uint64_t zero = 0;
  memcpy(sig_addr, &zero, sizeof(uint64_t));
}

void QpPingPongTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                    size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(QpPingPongTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, r_buf,
                     data_s_buf, data_r_buf, sig_addr, size,
                     args.op_type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void QpPingPongTester::verifyResults([[maybe_unused]] size_t size) {}
