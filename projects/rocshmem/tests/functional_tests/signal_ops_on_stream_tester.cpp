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

#include <unistd.h>

#include <cstring>

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SignalOpsOnStreamTester::SignalOpsOnStreamTester(TesterArguments args)
    : Tester(args) {
  _print_results = false;
  num_iters = args.skip + args.loop;
  sig_addr = static_cast<uint64_t *>(alloc_test_buffer(sizeof(uint64_t)));
  CHECK_HIP(hipHostMalloc(&observed, num_iters * sizeof(uint64_t)));
}

SignalOpsOnStreamTester::~SignalOpsOnStreamTester() {
  free_test_buffer(sig_addr);
  CHECK_HIP(hipHostFree(observed));
}

void SignalOpsOnStreamTester::resetBuffers([[maybe_unused]] size_t size) {
  *sig_addr = 0;
  std::memset(observed, 0, num_iters * sizeof(uint64_t));
}

void SignalOpsOnStreamTester::launchKernel([[maybe_unused]] dim3 gridSize,
                                           [[maybe_unused]] dim3 blockSize,
                                           [[maybe_unused]] int loop,
                                           [[maybe_unused]] size_t size) {
  int next_pe = (args.myid + 1) % args.numprocs;
  uint64_t expected = 0;

  // PE 0 starts each round and the other PEs forward it once received, so
  // each copy queued after a wait must observe exactly the value waited for.
  // Sends are delayed so that a wait that let the copy run early reads a stale
  // value.
  auto send = [&](int i) {
    usleep(1000);
    if (i % 2 == 0) {
      rocshmem_signal_set_on_stream(sig_addr, expected, next_pe, stream);
    } else {
      rocshmem_signal_add_on_stream(sig_addr, i + 1, next_pe, stream);
    }
  };

  for (int i = 0; i < num_iters; i++) {
    expected += i + 1;

    if (args.myid == 0) {
      send(i);
    }
    rocshmem_signal_wait_until_on_stream(sig_addr, ROCSHMEM_CMP_EQ, expected,
                                         stream);
    CHECK_HIP(hipMemcpyAsync(&observed[i], sig_addr, sizeof(uint64_t),
                             hipMemcpyDeviceToHost, stream));
    if (args.myid != 0) {
      send(i);
    }
  }
}

void SignalOpsOnStreamTester::verifyResults([[maybe_unused]] size_t size) {
  uint64_t expected = 0;
  for (int i = 0; i < num_iters; i++) {
    expected += i + 1;
    if (observed[i] != expected) {
      fprintf(stderr, "PE %d round %d (%s): observed %lu, expected %lu\n",
              args.myid, i, (i % 2 == 0) ? "signal_set" : "signal_add",
              observed[i], expected);
      rocshmem_global_exit(1);
    }
  }
}
