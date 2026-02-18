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

#include "barrier_all_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstring>
#include <cassert>
#include <vector>

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
BarrierAllOnStreamTester::BarrierAllOnStreamTester(TesterArguments args)
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
}

BarrierAllOnStreamTester::~BarrierAllOnStreamTester() {
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipEventDestroy(stop_events_timed[i]));
    CHECK_HIP(hipEventDestroy(start_events_timed[i]));
    // Don't destroy default stream (nullptr)
    if (!use_default_stream) {
      CHECK_HIP(hipStreamDestroy(streams[i]));
    }
  }
}

void BarrierAllOnStreamTester::preLaunchKernel() {
  // No specific setup needed for barrier
}

void BarrierAllOnStreamTester::postLaunchKernel() {
  // Synchronize all streams to ensure events are recorded
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }

  // Get elapsed time for each stream from HIP events
  for (int stream_id = 0; stream_id < num_streams && stream_id < num_timers;
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
  for (int i = num_streams; i < num_timers; i++) {
    start_time[i] = 0;
    end_time[i] = 0;
  }
}

void BarrierAllOnStreamTester::resetBuffers(size_t size) {}

void BarrierAllOnStreamTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                            int loop, size_t size) {
  // Execute warmup iterations (skip)
  for (int i = 0; i < args.skip; i++) {
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
      rocshmem_barrier_all_on_stream(streams[stream_id]);
    }
  }

  for (int i = 0; i < loop; i++) {
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
      // Record start event for this stream on first iteration
      if (i == 0) {
        CHECK_HIP(hipEventRecord(start_events_timed[stream_id],
                                 streams[stream_id]));
      }

      rocshmem_barrier_all_on_stream(streams[stream_id]);

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

void BarrierAllOnStreamTester::verifyResults(size_t size) {}
