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

#include "quiet_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstring>
#include <cassert>
#include <vector>

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
QuietOnStreamTester::QuietOnStreamTester(TesterArguments args)
    : Tester(args) {
  my_pe = rocshmem_my_pe();
  n_pes = rocshmem_n_pes();

  char *value{nullptr};
  if ((value = getenv("ROCSHMEM_TEST_NUM_STREAMS"))) {
    num_streams = atoi(value);
  } else {
    // Default to 1 stream
    num_streams = 1;
  }

  // Check if we should test with nullptr (default stream)
  use_default_stream = false;
  if ((value = getenv("ROCSHMEM_TEST_USE_DEFAULT_STREAM"))) {
    use_default_stream = (atoi(value) != 0);
    if (use_default_stream) {
      num_streams = 1;  // Only test with one nullptr stream
    }
  }

  streams.resize(num_streams);
  for (int i = 0; i < num_streams; i++) {
    if (use_default_stream) {
      streams[i] = 0;  // Use default stream (0)
    } else {
      CHECK_HIP(hipStreamCreate(&streams[i]));
    }
  }

  // Allocate test buffers - one int per stream
  source_buf = static_cast<int*>(alloc_test_buffer(num_streams * sizeof(int), args.local_buf_type));
  dest_buf = static_cast<int*>(alloc_test_buffer(num_streams * sizeof(int)));
}

QuietOnStreamTester::~QuietOnStreamTester() {
  for (int i = 0; i < num_streams; i++) {
    if (!use_default_stream) {
      CHECK_HIP(hipStreamDestroy(streams[i]));
    }
  }

  free_test_buffer(dest_buf);
  free_test_buffer(source_buf, args.local_buf_type);
}

void QuietOnStreamTester::preLaunchKernel() {
  bw_factor = 1;  // Point-to-point operation
}

void QuietOnStreamTester::postLaunchKernel() {
}

void QuietOnStreamTester::resetBuffers([[maybe_unused]] size_t size) {
  // Initialize source values for each stream
  for (int stream_id = 0; stream_id < num_streams; stream_id++) {
    source_buf[stream_id] = my_pe + 1;
    dest_buf[stream_id] = 0;
  }
}

void QuietOnStreamTester::launchKernel([[maybe_unused]] dim3 gridSize,
                                       [[maybe_unused]] dim3 blockSize,
                                       int loop, [[maybe_unused]] size_t size) {
  int next_pe = (my_pe + 1) % n_pes;

  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }

  // Just do one put+quiet operation per stream for correctness testing
  for (int stream_id = 0; stream_id < num_streams; stream_id++) {
    rocshmem_putmem_on_stream(&dest_buf[stream_id], &source_buf[stream_id],
                              sizeof(int), next_pe, streams[stream_id]);
    rocshmem_quiet_on_stream(streams[stream_id]);
  }

  // Synchronize all streams to ensure operations are complete
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }
}

void QuietOnStreamTester::verifyResults([[maybe_unused]] size_t size) {
  // After quiet, each stream's buffer should have been updated by the previous PE
  int prev_pe = (my_pe - 1 + n_pes) % n_pes;
  int expected = prev_pe + 1;

  for (int stream_id = 0; stream_id < num_streams; stream_id++) {
    if (dest_buf[stream_id] != expected) {
      fprintf(stderr, "PE %d stream %d: Verification failed: expected %d, got %d\n",
              my_pe, stream_id, expected, dest_buf[stream_id]);
      rocshmem_global_exit(1);
    }
  }
}
