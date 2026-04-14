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
  start_events_timed.resize(num_streams);
  stop_events_timed.resize(num_streams);
  for (int i = 0; i < num_streams; i++) {
    if (use_default_stream) {
      streams[i] = nullptr;  // Use default stream (0)
    } else {
      CHECK_HIP(hipStreamCreate(&streams[i]));
    }
    CHECK_HIP(hipEventCreate(&start_events_timed[i]));
    CHECK_HIP(hipEventCreate(&stop_events_timed[i]));
  }

  // Allocate symmetric heap buffers for testing RMA operations
  source_buf = (uint64_t *)rocshmem_malloc(sizeof(uint64_t) * 1024);
  dest_buf = (uint64_t *)rocshmem_malloc(sizeof(uint64_t) * 1024);

  // Initialize source buffer
  for (int i = 0; i < 1024; i++) {
    source_buf[i] = my_pe * 1000 + i;
  }
}

QuietOnStreamTester::~QuietOnStreamTester() {
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipEventDestroy(stop_events_timed[i]));
    CHECK_HIP(hipEventDestroy(start_events_timed[i]));
    // Don't destroy default stream (nullptr)
    if (!use_default_stream) {
      CHECK_HIP(hipStreamDestroy(streams[i]));
    }
  }

  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
}

void QuietOnStreamTester::preLaunchKernel() {
  // Reset destination buffer before each test
  memset(dest_buf, 0, sizeof(uint64_t) * 1024);
}

void QuietOnStreamTester::postLaunchKernel() {
  // Synchronize all streams to ensure events are recorded
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }

  // Get elapsed time for each stream from HIP events
  for (uint32_t stream_id = 0; stream_id < static_cast<uint32_t>(num_streams) && stream_id < static_cast<uint32_t>(num_timers);
       stream_id++) {
    float elapsed_time_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_time_ms,
                                  start_events_timed[stream_id],
                                  stop_events_timed[stream_id]));

    // Convert milliseconds to GPU cycles
    // wall_clk_rate is in kHz, so: cycles = ms * wall_clk_rate
    long long int elapsed_cycles =
        static_cast<long long int>(elapsed_time_ms *
                                   static_cast<float>(wall_clk_rate));

    start_time[stream_id] = 0;
    end_time[stream_id] = elapsed_cycles;
  }

  // Fill remaining timers with zero if num_timers > num_streams
  for (uint32_t i = num_streams; i < static_cast<uint32_t>(num_timers); i++) {
    start_time[i] = 0;
    end_time[i] = 0;
  }
}

void QuietOnStreamTester::resetBuffers([[maybe_unused]] size_t size) {}

void QuietOnStreamTester::launchKernel([[maybe_unused]] dim3 gridSize, [[maybe_unused]] dim3 blockSize,
                                            int loop, [[maybe_unused]] size_t size) {
  // Calculate target PE (next PE in ring)
  int target_pe = (my_pe + 1) % n_pes;

  // Execute warmup iterations (skip)
  for (int i = 0; i < args.skip; i++) {
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
      // Do some RMA operations before quiet
      rocshmem_putmem_on_stream(dest_buf, source_buf, sizeof(uint64_t) * 1024,
                                target_pe, streams[stream_id]);
      rocshmem_quiet_on_stream(streams[stream_id]);
    }
  }

  for (int i = 0; i < loop; i++) {
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
      // Record start event for this stream on first iteration
      if (i == 0) {
        CHECK_HIP(hipEventRecord(start_events_timed[stream_id],
                                 streams[stream_id]));
      }

      // Do some RMA operations before quiet to make the test meaningful
      rocshmem_putmem_on_stream(dest_buf, source_buf, sizeof(uint64_t) * 1024,
                                target_pe, streams[stream_id]);
      rocshmem_quiet_on_stream(streams[stream_id]);

      // Record stop event for this stream on last iteration
      if (i == loop - 1) {
        CHECK_HIP(hipEventRecord(stop_events_timed[stream_id],
                                 streams[stream_id]));
      }
    }
  }

  num_msgs = (loop + args.skip) * num_streams;
  num_timed_msgs = loop * num_streams;
}

void QuietOnStreamTester::verifyResults([[maybe_unused]] size_t size) {
  // Verify that quiet completed all RMA operations
  // The dest_buf should contain data from the previous PE
  int source_pe = (my_pe - 1 + n_pes) % n_pes;
  for (int i = 0; i < 1024; i++) {
    uint64_t expected = source_pe * 1000 + i;
    if (dest_buf[i] != expected) {
      fprintf(stderr, "PE %d: Verification failed at index %d: expected %lu, got %lu\n",
              my_pe, i, expected, dest_buf[i]);
      exit(1);
    }
  }
}
