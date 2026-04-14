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

#include "putmem_signal_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstring>
#include <cassert>
#include <vector>

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PutmemSignalOnStreamTester::PutmemSignalOnStreamTester(TesterArguments args)
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

  // Set target PE to put to (default: next PE in ring)
  pe_target = (my_pe + 1) % n_pes;
  if ((value = getenv("ROCSHMEM_TEST_PUTMEM_TARGET"))) {
    pe_target = atoi(value);
    if (pe_target < 0 || pe_target >= n_pes) {
      std::cerr << "Invalid ROCSHMEM_TEST_PUTMEM_TARGET value. Using next PE."
                << std::endl;
      pe_target = (my_pe + 1) % n_pes;
    }
  }

  int num_bytes_stream = args.max_msg_size;
  int total_bytes = num_bytes_stream * num_streams;
  buf_size = total_bytes;

  source_buf = static_cast<char *>(rocshmem_malloc(buf_size));
  dest_buf = static_cast<char *>(rocshmem_malloc(buf_size));
  sig_addr = static_cast<uint64_t *>(rocshmem_malloc(num_streams * sizeof(uint64_t)));

  if (source_buf == nullptr || dest_buf == nullptr || sig_addr == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "source: " << source_buf << ", dest: " << dest_buf
              << ", sig_addr: " << sig_addr << std::endl;
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

PutmemSignalOnStreamTester::~PutmemSignalOnStreamTester() {
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipEventDestroy(stop_events_timed[i]));
    CHECK_HIP(hipEventDestroy(start_events_timed[i]));
    CHECK_HIP(hipStreamDestroy(streams[i]));
  }
  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
  rocshmem_free(sig_addr);
}

void PutmemSignalOnStreamTester::preLaunchKernel() {
  bw_factor = 1;  // Point-to-point operation
}

void PutmemSignalOnStreamTester::postLaunchKernel() {
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

void PutmemSignalOnStreamTester::resetBuffers(size_t size) {
  // Initialize source buffer on all PEs
  // Each stream has its own portion
  for (int stream_id = 0; stream_id < num_streams; stream_id++) {
    int idx = stream_id * size;
    // Each PE fills its source buffer with a unique value
    int value = (my_pe + 1) * 100 + stream_id;
    std::memset(source_buf + idx, value, size);
  }

  // Clear destination buffer (will receive data from other PEs)
  std::memset(dest_buf, 0, buf_size);
  
  // Clear signal addresses
  std::memset(sig_addr, 0, num_streams * sizeof(uint64_t));
}

void PutmemSignalOnStreamTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                               int loop, size_t size) {
  uint64_t signal_value = 1;

  // Execute warmup iterations (skip)
  for (int i = 0; i < args.skip; i++) {
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
      char *stream_source = source_buf + stream_id * size;
      char *stream_dest = dest_buf + stream_id * size;
      rocshmem_putmem_signal_on_stream(stream_dest, stream_source, size,
                                       &sig_addr[stream_id], signal_value,
                                       sig_op, pe_target, streams[stream_id]);
    }
  }

  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }

  // Reset signal addresses after warmup and synchronize across PEs
  std::memset(sig_addr, 0, num_streams * sizeof(uint64_t));
  rocshmem_barrier_all();

  for (int i = 0; i < loop; i++) {
    for (int stream_id = 0; stream_id < num_streams; stream_id++) {
      // Record start event for this stream on first iteration
      if (i == 0) {
        CHECK_HIP(hipEventRecord(start_events_timed[stream_id],
                                 streams[stream_id]));
      }

      char *stream_source = source_buf + stream_id * size;
      char *stream_dest = dest_buf + stream_id * size;
      rocshmem_putmem_signal_on_stream(stream_dest, stream_source, size,
                                       &sig_addr[stream_id], signal_value,
                                       sig_op, pe_target, streams[stream_id]);

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

void PutmemSignalOnStreamTester::verifyResults(size_t size) {
  // Synchronize to ensure all operations completed
  rocshmem_barrier_all();

  // Verify correctness: after putmem_signal, my dest buffer should have
  // the data that was put from the PE that targets me
  // We need to find which PE writes to me: pe_source where (pe_source + 1) % n_pes == my_pe
  int pe_source = (my_pe - 1 + n_pes) % n_pes;
  
  for (int stream_id = 0; stream_id < num_streams; stream_id++) {
    int idx = stream_id * size;
    // Expected value is from pe_source
    int expected_value = (pe_source + 1) * 100 + stream_id;

    // Verify data
    for (size_t k = 0; k < size; k++) {
      if (static_cast<unsigned char>(dest_buf[idx + k]) !=
          static_cast<unsigned char>(expected_value)) {
        std::cerr << "PE " << my_pe << ": Data verification failed for stream "
                  << stream_id << " at byte " << k << std::endl;
        std::cerr << "Expected value from PE " << pe_source << ": " 
                  << expected_value
                  << ", Got: " << static_cast<int>(dest_buf[idx + k])
                  << std::endl;
        rocshmem_global_exit(1);
      }
    }

    // Verify signal
    uint64_t expected_signal = 1;
    if (sig_addr[stream_id] != expected_signal) {
      std::cerr << "PE " << my_pe << ": Signal verification failed for stream "
                << stream_id << std::endl;
      std::cerr << "Expected signal: " << expected_signal
                << ", Got: " << sig_addr[stream_id] << std::endl;
      rocshmem_global_exit(1);
    }
  }
}

