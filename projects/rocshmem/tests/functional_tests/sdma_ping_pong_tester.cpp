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

#include "sdma_ping_pong_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include "context.hpp"
#include "ipc_policy.hpp"
#include "sdma/anvil_device.hpp"
#include "assembly.hpp"
#include "verify_results_kernels.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * Bypass the rocSHMEM API and use anvil directly on the SDMA queue.
 *
 * op_type selects the signaling method:
 *   0,1 — sdma_anvil::put (no quiet), spin on data value
 *   2   — sdma_anvil::put + quiet + GPU shader atomic (separate signal)
 *
 * Only thread 0 is active (single-producer handle).
 *****************************************************************************/
__global__ void SdmaPingPongTest(int loop, int skip,
                                 long long int *start_time,
                                 long long int *end_time,
                                 char *s_buf, char *r_buf,
                                 uint64_t *sig_addr, size_t size,
                                 unsigned op_type,
                                 ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  if (threadIdx.x == 0) {
    Context *base_ctx = reinterpret_cast<Context *>(ctx.ctx_opaque);
    auto &ipc = base_ctx->ipcImpl_;
    auto &sdma = ipc.sdmaImpl_;

    int pe = rocshmem_ctx_my_pe(ctx);
    int my_local = ipc.shm_rank;
    int target_local = 1 - my_local;

    sdma_anvil::SdmaQueueDeviceHandle *handle =
        sdma.deviceHandles_d[target_local * sdma.numChannels + sdma.sdmaChannel];

    int wg_id = hipBlockIdx_x;

    char *my_base = ipc.ipc_bases[my_local];
    char *remote_base = ipc.ipc_bases[target_local];

    char *my_s = s_buf + size * wg_id;
    char *my_r = r_buf + size * wg_id;
    uint64_t r_offset = my_r - my_base;
    void *remote_r_buf = remote_base + r_offset;

    uint64_t *my_sig = &sig_addr[wg_id];
    uint64_t sig_offset_bytes =
        reinterpret_cast<char *>(my_sig) - my_base;
    uint64_t *remote_sig = reinterpret_cast<uint64_t *>(remote_base + sig_offset_bytes);

    int *r_int = reinterpret_cast<int *>(my_r);
    int *s_int = reinterpret_cast<int *>(my_s);

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

      if (op_type <= 1) {
        int val = i + 1;
        if (pe == 0) {
          *s_int = val;
          __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
          sdma_anvil::put(*handle, remote_r_buf, my_s, sizeof(int));
          while (uncached_load(r_int) != val) {}
        } else {
          while (uncached_load(r_int) != val) {}
          *s_int = val;
          __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
          sdma_anvil::put(*handle, remote_r_buf, my_s, sizeof(int));
        }
      } else {
        uint64_t expected = static_cast<uint64_t>(i + 1);
        if (pe == 0) {
          sdma_anvil::put(*handle, remote_r_buf, my_s, size);
          sdma_anvil::quiet(*handle);
          __hip_atomic_fetch_add(remote_sig, 1ULL, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
          sdma_anvil::waitSignal(my_sig, expected);
        } else {
          sdma_anvil::waitSignal(my_sig, expected);
          sdma_anvil::put(*handle, remote_r_buf, my_s, size);
          sdma_anvil::quiet(*handle);
          __hip_atomic_fetch_add(remote_sig, 1ULL, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
    end_time[wg_id] = wall_clock64();

    sdma_anvil::quiet(*handle);
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SdmaPingPongTester::SdmaPingPongTester(TesterArguments args) : Tester(args) {
  s_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  r_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  sig_addr = (uint64_t *)alloc_test_buffer(sizeof(uint64_t) * args.num_wgs);
  rtt_factor = 2;
  bw_factor = 2;
}

SdmaPingPongTester::~SdmaPingPongTester() {
  free_test_buffer(s_buf);
  free_test_buffer(r_buf);
  free_test_buffer(sig_addr);
}

void SdmaPingPongTester::resetBuffers(size_t size) {
  CHECK_HIP(hipMemset(s_buf, 'a', size * args.num_wgs));
  CHECK_HIP(hipMemset(r_buf, 0, size * args.num_wgs));
  CHECK_HIP(hipMemset(sig_addr, 0, sizeof(uint64_t) * args.num_wgs));
}

void SdmaPingPongTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                      size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(SdmaPingPongTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     s_buf, r_buf, sig_addr, size, args.op_type,
                     _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void SdmaPingPongTester::verifyResults(size_t size) {
  if (args.op_type <= 1) return;

  size_t check_bytes = size * args.num_wgs;
  *verification_error = false;
  size_t block = std::min((size_t)1024, check_bytes);
  size_t grid = (check_bytes + block - 1) / block;
  hipLaunchKernelGGL(rocshmem::verify_results_kernel_char, grid, block, 0, stream,
                     r_buf, check_bytes, check_bytes, 1, 1, 0, 1,
                     verification_error);
  CHECK_HIP(hipStreamSynchronize(stream));

  if (*verification_error) {
    fprintf(stderr, "FAIL: r_buf data mismatch (expected 'a') at size=%zu rank=%d\n",
            size, args.myid);
    rocshmem_global_exit(1);
  }
}
