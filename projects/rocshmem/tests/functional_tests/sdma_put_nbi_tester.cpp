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

#include "sdma_put_nbi_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include "context.hpp"
#include "ipc_policy.hpp"
#include "sdma/anvil_device.hpp"
#include "constmem.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *
 * One-way blocking put using anvil::put + anvil::quiet directly on the
 * SDMA queue.  PE0 sends, PE1 is passive.  Sweeps message sizes.
 *****************************************************************************/
__global__ void SdmaPutNbiTest(int loop, int skip,
                               long long int *start_time,
                               long long int *end_time,
                               char *source, char *dest, size_t size,
                               ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  if (threadIdx.x == 0) {
    Context *base_ctx = reinterpret_cast<Context *>(ctx.ctx_opaque);
    auto &sdma = base_ctx->ipcImpl_.sdmaImpl_;

    int pe = constmem.my_pe;
    int target = 1 - pe;

    int target_local_pe{-1};
    base_ctx->ipcImpl_.isIpcAvailable(pe, target, &target_local_pe);

    anvil::SdmaQueueDeviceHandle *handle =
        sdma.deviceHandles_d[target_local_pe * sdma.numChannels + 0];

    char *my_base = base_ctx->ipcImpl_.ipc_bases[pe];
    char *remote_base = base_ctx->ipcImpl_.ipc_bases[target];
    uint64_t offset = dest - my_base;
    void *remote_dest = remote_base + offset;

    int wg_id = hipBlockIdx_x;

    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        start_time[wg_id] = wall_clock64();
      }
      anvil::put(*handle, remote_dest, source, size);
      anvil::quiet(*handle);
    }
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SdmaPutNbiTester::SdmaPutNbiTester(TesterArguments args) : Tester(args) {
  s_buf = (char *)alloc_test_buffer(max_msg_size);
  r_buf = (char *)alloc_test_buffer(max_msg_size);
}

SdmaPutNbiTester::~SdmaPutNbiTester() {
  free_test_buffer(s_buf);
  free_test_buffer(r_buf);
}

void SdmaPutNbiTester::resetBuffers(size_t size) {
  memset(s_buf, 0xAB, size);
  memset(r_buf, 0, size);
}

void SdmaPutNbiTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                    size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(SdmaPutNbiTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     s_buf, r_buf, size, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void SdmaPutNbiTester::verifyResults([[maybe_unused]] size_t size) {}
