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
#include "verify_results_kernels.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * op_type selects the put method:
 *   0 — rocshmem_ctx_int_p (scalar immediate, 4B fixed)
 *   1 — rocshmem_ctx_putmem_nbi from symmetric source (4B fixed)
 *   2 — rocshmem_ctx_putmem_signal with SIGNAL_ADD (variable msg size)
 *****************************************************************************/
__global__ void PingPongTest(int loop, int skip, long long int *start_time,
                             long long int *end_time, int *r_buf,
                             int *s_buf, char *data_s_buf, char *data_r_buf,
                             uint64_t *sig_addr, size_t size,
                             unsigned op_type, ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int pe = rocshmem_ctx_my_pe(ctx);

  if (is_thread_zero_in_block()) {

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        start_time[wg_id] = wall_clock64();
      }

      int target = 1 - pe;

      if (op_type == 2) {
        char *my_data_s = data_s_buf + size * wg_id;
        char *my_data_r = data_r_buf + size * wg_id;
        uint64_t *my_sig = &sig_addr[wg_id];
        uint64_t expected = static_cast<uint64_t>(i + 1);
        if (pe == 0) {
          rocshmem_ctx_putmem_signal_nbi(ctx, my_data_r, my_data_s, size,
                                     my_sig, 1, ROCSHMEM_SIGNAL_ADD, target);
          rocshmem_ulong_wait_until(
              reinterpret_cast<unsigned long *>(my_sig),
              ROCSHMEM_CMP_EQ, expected);
        } else {
          rocshmem_ulong_wait_until(
              reinterpret_cast<unsigned long *>(my_sig),
              ROCSHMEM_CMP_EQ, expected);
          rocshmem_ctx_putmem_signal_nbi(ctx, my_data_r, my_data_s, size,
                                     my_sig, 1, ROCSHMEM_SIGNAL_ADD, target);
        }
      } else {
        if (pe == 0) {
          if (op_type == 0) {
            rocshmem_ctx_int_p(ctx, &r_buf[hipBlockIdx_x], i + 1, target);
          } else {
            s_buf[hipBlockIdx_x] = i + 1;
            rocshmem_ctx_putmem_nbi(ctx, &r_buf[hipBlockIdx_x],
                                    &s_buf[hipBlockIdx_x], sizeof(int), target);
          }
          rocshmem_int_wait_until(&r_buf[hipBlockIdx_x], ROCSHMEM_CMP_EQ,
                                   i + 1);
        } else {
          rocshmem_int_wait_until(&r_buf[hipBlockIdx_x], ROCSHMEM_CMP_EQ,
                                   i + 1);
          if (op_type == 0) {
            rocshmem_ctx_int_p(ctx, &r_buf[hipBlockIdx_x], i + 1, target);
          } else {
            s_buf[hipBlockIdx_x] = i + 1;
            rocshmem_ctx_putmem_nbi(ctx, &r_buf[hipBlockIdx_x],
                                    &s_buf[hipBlockIdx_x], sizeof(int), target);
          }
        }
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
  r_buf = (int *)alloc_test_buffer(sizeof(int) * args.num_wgs);
  s_buf = (int *)alloc_test_buffer(sizeof(int) * args.num_wgs);
  data_s_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  data_r_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  sig_addr = (uint64_t *)alloc_test_buffer(sizeof(uint64_t) * args.num_wgs);
  rtt_factor = 2;
  bw_factor = 2;
}

PingPongTester::~PingPongTester() {
  free_test_buffer(r_buf);
  free_test_buffer(s_buf);
  free_test_buffer(data_s_buf);
  free_test_buffer(data_r_buf);
  free_test_buffer(sig_addr);
}

void PingPongTester::resetBuffers(size_t size) {
  CHECK_HIP(hipMemset(r_buf, 0, sizeof(int) * args.num_wgs));
  CHECK_HIP(hipMemset(s_buf, 0, sizeof(int) * args.num_wgs));
  CHECK_HIP(hipMemset(data_s_buf, 'a', size * args.num_wgs));
  CHECK_HIP(hipMemset(data_r_buf, 0, size * args.num_wgs));
  CHECK_HIP(hipMemset(sig_addr, 0, sizeof(uint64_t) * args.num_wgs));
}

void PingPongTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(PingPongTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, r_buf, s_buf,
                     data_s_buf, data_r_buf, sig_addr, size,
                     args.op_type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void PingPongTester::verifyResults(size_t size) {
  if (args.op_type != 2) return;

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
