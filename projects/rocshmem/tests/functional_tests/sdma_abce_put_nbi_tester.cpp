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

#include "sdma_abce_put_nbi_tester.hpp"

// Only the PUBLIC rocSHMEM header. Deliberately NOT context.hpp / ipc_policy.hpp:
// those pull in sdma_policy.hpp -> sdma/anvil's sdma_opcodes.h, whose SDMA_*
// opcode CONSTS collide with ABCE's sdma_packets.h MACROS (same names) when the
// ABCE copy-engine header is also in this TU. The ABCE tester needs no anvil SDMA
// at all; it gets the peer address from the public rocshmem_ptr() instead of
// reaching into ipc_bases.
#include <rocshmem/rocshmem.hpp>
#include "assembly.hpp"  // uncached_load: cache-bypassing read so the drain poll sees the engine's HBM write
#include "verify_results_kernels.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * One-way non-blocking put using the ABCE copy engine directly. Pipelines
 * SubmitLinearOneLane calls with no per-copy completion (bare puts, the leanest
 * path -- no extra per-put HBM completion write) and drains a batch by submitting
 * one completion-only signal (lands after every earlier op on the engine) and
 * polling it. This mirrors the anvil sdma_put_nbi tester (put + quiet), with the
 * ABCE drain standing in for sdma_anvil::quiet.
 *
 * The copy-engine handle is a value, passed in by the host. Peer addressing uses
 * the public rocshmem_ptr() (symmetric-heap translation), not anvil IPC internals.
 *****************************************************************************/
__global__ void SdmaAbcePutNbiTest(int loop, int skip,
                                   long long int *start_time,
                                   long long int *end_time,
                                   char *source, char *dest, size_t size,
                                   int batch, ShmemContextType ctx_type,
                                   hipExtCopyEngineDeviceHandle handle,
                                   uint64_t *drain_signal) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  if (threadIdx.x == 0) {
    int my_pe = rocshmem_ctx_my_pe(ctx);
    int peer_pe = 1 - my_pe;  // 2-PE

    int wg_id = hipBlockIdx_x;
    int start_slot = (batch - (skip % batch)) % batch;

    // One drain word per workgroup; a monotonic Add count so the host need not
    // reset it. Add (SDMA_PKT_ATOMIC) is one packet / one HBM write pre-gfx125,
    // vs Set's two fence writes; we target gfx90a+ so the atomic is always
    // available and is the cheaper completion.
    uint64_t *my_drain = &drain_signal[wg_id];
    uint64_t drains = 0;

    auto drain = [&]() {
      ++drains;
      const hipExtCopyEngineSignal completion{reinterpret_cast<uint64_t>(my_drain), 1,
                                              hipExtCopyEngineSignalAdd, 0};
      // A completion-only submit lands after every earlier op on the engine, so
      // the Add fires once this batch's puts have all retired; after |drains|
      // drains the word has accumulated |drains|.
      hipExtCopyEngineSubmitSignalOneLane(&handle, &completion);
      // Cache-bypassing read: the SDMA engine writes this word to HBM; a cached
      // load can be satisfied from L2 and never observe it.
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
      int slot = (start_slot + i) % batch;

      if (slot == 0) {
        drain();
        if (i == skip) {
          start_time[wg_id] = wall_clock64();
        }
      }

      // Peer address of this slot's dest buffer via the public symmetric-ptr API.
      void *remote_dest = rocshmem_ptr(dest + size * slot, peer_pe);

      // Bare put: no completion (constant nullptr, so the compiler keeps nothing
      // in memory and the engine does no extra per-put completion write).
      hipExtCopyEngineSubmitLinearOneLane(&handle, remote_dest, source, size);
    }

    drain();
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SdmaAbcePutNbiTester::SdmaAbcePutNbiTester(TesterArguments args) : Tester(args) {
  s_buf = (char *)alloc_test_buffer(max_msg_size);
  r_buf = (char *)alloc_test_buffer(max_msg_size * batch_size);
  // One drain word per workgroup.
  drain_signal_ = (uint64_t *)alloc_test_buffer(sizeof(uint64_t) * args.num_wgs);
}

SdmaAbcePutNbiTester::~SdmaAbcePutNbiTester() {
  free_test_buffer(s_buf);
  free_test_buffer(r_buf);
  free_test_buffer(drain_signal_);
}

void SdmaAbcePutNbiTester::ensureHandle() {
  if (have_handle_) return;
  // This process owns one GPU (test_driver sets device = local rank). The 2-PE
  // node-local peer is the other local rank's device. Exchange device ordinals
  // over MPI local-rank env so GetDeviceHandles(myDev, myDev, peerDev) targets
  // the engine that reaches the peer's memory.
  int myDev = 0;
  CHECK_HIP(hipGetDevice(&myDev));
  // 2-PE: peer device ordinal = the other rank's device. With one GPU per rank
  // bound by local rank, peerDev = 1 - myDev for a 2-GPU node-local pair.
  int peerDev = (myDev == 0) ? 1 : 0;
  int n = 0;
  hipError_t e = hipExtCopyEngineGetDeviceHandles(myDev, myDev, peerDev, 1, &handle_, &n);
  if (e != hipSuccess || n < 1) {
    fprintf(stderr, "SdmaAbcePutNbi: GetDeviceHandles(%d->%d) failed: %s (n=%d)\n", myDev, peerDev,
            hipGetErrorString(e), n);
    rocshmem_global_exit(1);
  }
  have_handle_ = true;
}

void SdmaAbcePutNbiTester::resetBuffers(size_t size) {
  CHECK_HIP(hipMemset(s_buf, 'a', size));
  CHECK_HIP(hipMemset(r_buf, 0, size * batch_size));
  CHECK_HIP(hipMemset(drain_signal_, 0, sizeof(uint64_t) * args.num_wgs));
}

void SdmaAbcePutNbiTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                        size_t size) {
  ensureHandle();
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(SdmaAbcePutNbiTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     s_buf, r_buf, size, batch_size, _shmem_context,
                     handle_, drain_signal_);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void SdmaAbcePutNbiTester::verifyResults(size_t size) {
  if (args.myid != 1) return;

  size_t check_bytes = size * batch_size;
  *verification_error = false;
  size_t block = std::min((size_t)1024, check_bytes);
  size_t grid = (check_bytes + block - 1) / block;
  hipLaunchKernelGGL(rocshmem::verify_results_kernel_char, grid, block, 0, stream,
                     r_buf, check_bytes, check_bytes, 1, 1, 0, 1,
                     verification_error);
  CHECK_HIP(hipStreamSynchronize(stream));

  if (*verification_error) {
    fprintf(stderr, "FAIL: r_buf data mismatch (expected 'a') at size=%zu\n", size);
    rocshmem_global_exit(1);
  }
}
