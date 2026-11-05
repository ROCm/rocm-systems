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

#include "sdma_abce_ping_pong_tester.hpp"

// Only the PUBLIC rocSHMEM header -- NOT context.hpp / ipc_policy.hpp, which drag
// in sdma/anvil's sdma_opcodes.h whose SDMA_* opcode CONSTS collide with ABCE's
// sdma_packets.h MACROS when the copy-engine header is in this TU. Peer addresses
// come from the public rocshmem_ptr() instead of anvil IPC internals.
#include <rocshmem/rocshmem.hpp>
#include "assembly.hpp"  // uncached_load: cache-bypassing read so the spin observes SDMA-landed HBM writes
#include "verify_results_kernels.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * 2-PE ping-pong using the ABCE copy engine directly (mirrors the anvil
 * sdma_ping_pong tester). op_type selects the signaling method:
 *   0,1 - SubmitLinearOneLane (no completion), spin on the data value.
 *   2   - SubmitLinearOneLane + drain + GPU shader atomic (separate signal).
 *
 * Only thread 0 is active (single lane -> single engine). The copy-engine handle
 * is a value passed in by the host; peer addresses via the public rocshmem_ptr().
 *****************************************************************************/
__global__ void SdmaAbcePingPongTest(int loop, int skip,
                                     long long int *start_time,
                                     long long int *end_time,
                                     char *s_buf, char *r_buf,
                                     uint64_t *sig_addr, size_t size,
                                     unsigned op_type,
                                     ShmemContextType ctx_type,
                                     hipExtCopyEngineDeviceHandle handle,
                                     uint64_t *drain_signal) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  if (threadIdx.x == 0) {
    int pe = rocshmem_ctx_my_pe(ctx);
    int peer_pe = 1 - pe;  // 2-PE

    int wg_id = hipBlockIdx_x;

    char *my_s = s_buf + size * wg_id;
    char *my_r = r_buf + size * wg_id;
    // Peer's r_buf and signal cell via the public symmetric-ptr translation.
    void *remote_r_buf = rocshmem_ptr(my_r, peer_pe);

    uint64_t *my_sig = &sig_addr[wg_id];
    uint64_t *remote_sig = reinterpret_cast<uint64_t *>(rocshmem_ptr(my_sig, peer_pe));

    int *r_int = reinterpret_cast<int *>(my_r);
    int *s_int = reinterpret_cast<int *>(my_s);

    uint64_t *my_drain = &drain_signal[wg_id];
    // Monotonic Add drain (one SDMA_PKT_ATOMIC / one HBM write pre-gfx125, vs
    // Set's two fence writes); gfx90a+ always has the atomic.
    uint64_t drains = 0;
    auto drain = [&]() {
      ++drains;
      const hipExtCopyEngineSignal completion{reinterpret_cast<uint64_t>(my_drain), 1,
                                              hipExtCopyEngineSignalAdd, 0};
      hipExtCopyEngineSubmitSignalOneLane(&handle, &completion);
      // Cache-bypassing read: the SDMA engine writes this word to HBM; an ordinary
      // cached/atomic load can be satisfied from L2 and never see it.
      while (uncached_load(my_drain) < drains) {
      }
    };

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
          hipExtCopyEngineSubmitLinearOneLane(&handle, remote_r_buf, my_s, sizeof(int));
          while (uncached_load(r_int) != val) {}
        } else {
          while (uncached_load(r_int) != val) {}
          *s_int = val;
          __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
          hipExtCopyEngineSubmitLinearOneLane(&handle, remote_r_buf, my_s, sizeof(int));
        }
      } else {
        uint64_t expected = static_cast<uint64_t>(i + 1);
        if (pe == 0) {
          hipExtCopyEngineSubmitLinearOneLane(&handle, remote_r_buf, my_s, size);
          drain();
          __hip_atomic_fetch_add(remote_sig, 1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          while (uncached_load(my_sig) != expected) {}
        } else {
          while (uncached_load(my_sig) != expected) {}
          hipExtCopyEngineSubmitLinearOneLane(&handle, remote_r_buf, my_s, size);
          drain();
          __hip_atomic_fetch_add(remote_sig, 1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
    end_time[wg_id] = wall_clock64();

    if (op_type > 1) drain();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SdmaAbcePingPongTester::SdmaAbcePingPongTester(TesterArguments args) : Tester(args) {
  s_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  r_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  sig_addr = (uint64_t *)alloc_test_buffer(sizeof(uint64_t) * args.num_wgs);
  drain_signal_ = (uint64_t *)alloc_test_buffer(sizeof(uint64_t) * args.num_wgs);
  rtt_factor = 2;
  bw_factor = 2;
}

SdmaAbcePingPongTester::~SdmaAbcePingPongTester() {
  free_test_buffer(s_buf);
  free_test_buffer(r_buf);
  free_test_buffer(sig_addr);
  free_test_buffer(drain_signal_);
}

void SdmaAbcePingPongTester::ensureHandle() {
  if (have_handle_) return;
  int myDev = 0;
  CHECK_HIP(hipGetDevice(&myDev));
  int peerDev = (myDev == 0) ? 1 : 0;  // 2-PE node-local pair
  int n = 0;
  hipError_t e = hipExtCopyEngineGetDeviceHandles(myDev, myDev, peerDev, 1, &handle_, &n);
  if (e != hipSuccess || n < 1) {
    fprintf(stderr, "SdmaAbcePingPong: GetDeviceHandles(%d->%d) failed: %s (n=%d)\n", myDev, peerDev,
            hipGetErrorString(e), n);
    rocshmem_global_exit(1);
  }
  have_handle_ = true;
}

void SdmaAbcePingPongTester::resetBuffers(size_t size) {
  CHECK_HIP(hipMemset(s_buf, 'a', size * args.num_wgs));
  CHECK_HIP(hipMemset(r_buf, 0, size * args.num_wgs));
  CHECK_HIP(hipMemset(sig_addr, 0, sizeof(uint64_t) * args.num_wgs));
  CHECK_HIP(hipMemset(drain_signal_, 0, sizeof(uint64_t) * args.num_wgs));
}

void SdmaAbcePingPongTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                          size_t size) {
  ensureHandle();
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(SdmaAbcePingPongTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     s_buf, r_buf, sig_addr, size, args.op_type,
                     _shmem_context, handle_, drain_signal_);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void SdmaAbcePingPongTester::verifyResults(size_t size) {
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
