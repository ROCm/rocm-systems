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

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>

#include <cstring>
#include <cassert>

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
SignalWaitUntilOnStreamTester::SignalWaitUntilOnStreamTester(
    TesterArguments args)
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

  // Set target PE (next PE in ring)
  pe_target = (my_pe + 1) % n_pes;

  // Allocate signal addresses on symmetric heap
  sig_addr =
      static_cast<uint64_t *>(rocshmem_malloc(num_streams * sizeof(uint64_t)));
  source_buf =
      static_cast<uint64_t *>(rocshmem_malloc(num_streams * sizeof(uint64_t)));

  if (sig_addr == nullptr || source_buf == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "sig_addr: " << sig_addr << ", source_buf: " << source_buf
              << std::endl;
    rocshmem_global_exit(1);
  }

  streams.resize(num_streams);
  start_events_timed.resize(num_streams);
  stop_events_timed.resize(num_streams);
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamCreate(&streams[i]));
    CHECK_HIP(hipEventCreate(&start_events_timed[i]));
    CHECK_HIP(hipEventCreate(&stop_events_timed[i]));
  }
}

SignalWaitUntilOnStreamTester::~SignalWaitUntilOnStreamTester() {
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipEventDestroy(stop_events_timed[i]));
    CHECK_HIP(hipEventDestroy(start_events_timed[i]));
    CHECK_HIP(hipStreamDestroy(streams[i]));
  }
  rocshmem_free(sig_addr);
  rocshmem_free(source_buf);
}

void SignalWaitUntilOnStreamTester::preLaunchKernel() {
  bw_factor = 1;  // Point-to-point operation
}

void SignalWaitUntilOnStreamTester::postLaunchKernel() {
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

void SignalWaitUntilOnStreamTester::resetBuffers(size_t size) {
  // Clear signal addresses
  std::memset(sig_addr, 0, num_streams * sizeof(uint64_t));
}

void SignalWaitUntilOnStreamTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                                  int loop, size_t size) {
  // Execute warmup + timed iterations
  for (int i = 0; i < args.skip + loop; i++) {
    // Increment signal value for each iteration
    uint64_t signal_value = i + 1;
    
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
      // Record start event after warmup on first timed iteration for all streams
      if (i == args.skip) {
        CHECK_HIP(hipEventRecord(start_events_timed[stream_id],
                                 streams[stream_id]));
      }

      // PE 0 starts the ring by signaling PE 1
      if (my_pe == 0) {
        rocshmem_putmem_signal_on_stream(&sig_addr[stream_id],
                                         &source_buf[stream_id],
                                         sizeof(uint64_t), &sig_addr[stream_id],
                                         signal_value, sig_op, pe_target,
                                         streams[stream_id]);
      } else {
        // All other PEs wait for signal from previous PE
        rocshmem_signal_wait_until_on_stream(&sig_addr[stream_id],
                                             ROCSHMEM_CMP_GE, signal_value,
                                             streams[stream_id]);

        // Forward the signal to next PE (unless we're the last PE)
        if (my_pe != n_pes - 1) {
          rocshmem_putmem_signal_on_stream(&sig_addr[stream_id],
                                           &source_buf[stream_id],
                                           sizeof(uint64_t), &sig_addr[stream_id],
                                           signal_value, sig_op, pe_target,
                                           streams[stream_id]);
        }
      }

      // Record stop event on last timed iteration for all streams
      if (i == args.skip + loop - 1) {
        CHECK_HIP(hipEventRecord(stop_events_timed[stream_id],
                                 streams[stream_id]));
      }
    }

    // Wait for all streams to complete
    for (int j = 0; j < num_streams; j++) {
      CHECK_HIP(hipStreamSynchronize(streams[j]));
    }
    
    // Barrier to ensure all RMA operations completed across all PEs
    rocshmem_barrier_all();
  }

  num_msgs = (loop + args.skip) * num_streams;
  num_timed_msgs = loop * num_streams;
}

void SignalWaitUntilOnStreamTester::verifyResults(size_t size) {
  // Synchronize to ensure all operations completed
  rocshmem_barrier_all();

  // Verify signal values
  // All PEs except PE 0 should have received the final signal value
  uint64_t expected_signal = args.skip + args.loop;

  for (int stream_id = 0; stream_id < num_streams; stream_id++) {
    // PE 0 doesn't receive signals (it initiates), so skip verification
    if (my_pe == 0) {
      continue;
    }

    // Verify signal
    if (sig_addr[stream_id] != expected_signal) {
      std::cerr << "PE " << my_pe << ": Signal verification failed for stream "
                << stream_id << std::endl;
      std::cerr << "Expected signal: " << expected_signal
                << ", Got: " << sig_addr[stream_id] << std::endl;
      rocshmem_global_exit(1);
    }
  }
}

