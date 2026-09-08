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
#include "verify_results_kernels.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * Bypass the public rocSHMEM API and call QueuePair methods directly.
 *
 * op_type selects the put method:
 *   0 — put_nbi_single (4B immediate inline), spin on data value
 *   1 — put_nbi_single from symmetric buffer (4B), spin on data value
 *   2 — put_nbi_single (variable size) + atomic_add_single for signal
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
    int pe = rocshmem_ctx_my_pe(ctx);
    int target = 1 - pe;

    QueuePair &qp = gda_ctx->qps[target];

    uintptr_t local_base =
        reinterpret_cast<uintptr_t>(gda_ctx->base_heap[pe]);
    uintptr_t remote_base =
        reinterpret_cast<uintptr_t>(gda_ctx->base_heap[target]);

    int wg_id = hipBlockIdx_x;

    uintptr_t r_buf_offset =
        reinterpret_cast<uintptr_t>(&r_buf[wg_id]) - local_base;
    void *remote_r_buf = reinterpret_cast<void *>(remote_base + r_buf_offset);

    char *my_data_s = data_s_buf + size * wg_id;
    uintptr_t data_r_offset =
        reinterpret_cast<uintptr_t>(data_r_buf + size * wg_id) - local_base;
    void *remote_data_r = reinterpret_cast<void *>(remote_base + data_r_offset);

    uint64_t *my_sig = &sig_addr[wg_id];
    uintptr_t sig_offset =
        reinterpret_cast<uintptr_t>(my_sig) - local_base;
    void *remote_sig = reinterpret_cast<void *>(remote_base + sig_offset);

    // Drain all setup loads from HBM before entering the timed loop.
#if defined(__GFX12__)
    asm volatile("s_wait_loadcnt 0x0\n s_wait_storecnt 0x0" ::: "memory");
#else
    __builtin_amdgcn_s_waitcnt(0);
#endif

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        start_time[wg_id] = wall_clock64();
      }

      int val = i + 1;

      if (op_type <= 1) {
        void *src = (op_type == 0) ? static_cast<void *>(&val)
                                   : static_cast<void *>(my_data_s);
        if (op_type == 1) {
          *reinterpret_cast<int *>(my_data_s) = val;
        }
        if (pe == 0) {
          qp.put_nbi_single(remote_r_buf, src, sizeof(int), true);
          while (uncached_load(&r_buf[wg_id]) != val) {}
        } else {
          while (uncached_load(&r_buf[wg_id]) != val) {}
          qp.put_nbi_single(remote_r_buf, src, sizeof(int), true);
        }
      } else {
        uint64_t expected = static_cast<uint64_t>(i + 1);
        if (pe == 0) {
          qp.put_nbi_single(remote_data_r, my_data_s, size, false);
          qp.atomic_add_single(remote_sig, 1);
          while (uncached_load(my_sig) < expected) {}
        } else {
          while (uncached_load(my_sig) < expected) {}
          qp.put_nbi_single(remote_data_r, my_data_s, size, false);
          qp.atomic_add_single(remote_sig, 1);
        }
      }
    }
    end_time[wg_id] = wall_clock64();

    qp.quiet_single();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
QpPingPongTester::QpPingPongTester(TesterArguments args) : Tester(args) {
  if (rocshmem_query_backend_type() != BackendType::GDA_BACKEND) {
    if (args.myid == 0) {
      std::cerr << "QpPingPong requires GDA backend (ROCSHMEM_BACKEND=gda)\n";
    }
    exit(1);
  }
  r_buf = (int *)alloc_test_buffer(sizeof(int) * args.num_wgs);
  data_s_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  data_r_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  sig_addr = (uint64_t *)alloc_test_buffer(sizeof(uint64_t) * args.num_wgs);
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
  CHECK_HIP(hipMemset(r_buf, 0, sizeof(int) * args.num_wgs));
  CHECK_HIP(hipMemset(data_s_buf, 'a', size * args.num_wgs));
  CHECK_HIP(hipMemset(data_r_buf, 0, size * args.num_wgs));
  CHECK_HIP(hipMemset(sig_addr, 0, sizeof(uint64_t) * args.num_wgs));
}

void QpPingPongTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                    size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(QpPingPongTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, r_buf,
                     data_s_buf, data_r_buf, sig_addr, size,
                     args.op_type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void QpPingPongTester::verifyResults(size_t size) {
  if (args.op_type <= 1) return;

  size_t check_bytes = size * args.num_wgs;
  *verification_error = false;
  size_t block = std::min((size_t)1024, check_bytes);
  size_t grid = (check_bytes + block - 1) / block;
  hipLaunchKernelGGL(rocshmem::verify_results_kernel_char, grid, block, 0, stream,
                     data_r_buf, check_bytes, check_bytes, 1, 1, 0, 1,
                     verification_error);
  CHECK_HIP(hipStreamSynchronize(stream));

  if (*verification_error) {
    fprintf(stderr, "FAIL: data_r_buf data mismatch (expected 'a') at size=%zu rank=%d\n",
            size, args.myid);
    rocshmem_global_exit(1);
  }
}
