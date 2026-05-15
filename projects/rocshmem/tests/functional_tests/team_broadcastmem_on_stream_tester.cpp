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

#include "team_broadcastmem_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstring>
#include <cassert>
#include <vector>

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamBroadcastmemOnStreamTester::TeamBroadcastmemOnStreamTester(TesterArguments args)
    : Tester(args) {
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_TEST_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  } else {
    // Default to number of work groups
    num_teams = args.num_wgs;
  }

  // Set root PE to 0 by default, can be modified via environment variable
  if ((value = getenv("ROCSHMEM_TEST_BROADCAST_ROOT"))) {
    pe_root = atoi(value);
    if (pe_root < 0 || pe_root >= n_pes) {
      std::cerr << "Invalid ROCSHMEM_TEST_BROADCAST_ROOT value. Using PE 0."
                << std::endl;
      pe_root = 0;
    }
  }

  int num_bytes_wg = args.max_msg_size;
  int total_bytes = num_bytes_wg * num_teams;
  buf_size = total_bytes;

  source_buf = static_cast<char *>(rocshmem_malloc(buf_size));
  dest_buf = static_cast<char *>(rocshmem_malloc(buf_size));

  if (source_buf == nullptr || dest_buf == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "source: " << source_buf << ", dest: " << dest_buf
              << std::endl;
    rocshmem_global_exit(1);
  }

  team_world_dup.resize(num_teams);

  streams.resize(num_teams);
  start_events_timed.resize(num_teams);
  stop_events_timed.resize(num_teams);
  for (int i = 0; i < num_teams; i++) {
    CHECK_HIP(hipStreamCreate(&streams[i]));
    CHECK_HIP(hipEventCreate(&start_events_timed[i]));
    CHECK_HIP(hipEventCreate(&stop_events_timed[i]));
  }
}

TeamBroadcastmemOnStreamTester::~TeamBroadcastmemOnStreamTester() {
  for (int i = 0; i < num_teams; i++) {
    CHECK_HIP(hipEventDestroy(stop_events_timed[i]));
    CHECK_HIP(hipEventDestroy(start_events_timed[i]));
    CHECK_HIP(hipStreamDestroy(streams[i]));
  }
  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
}

void TeamBroadcastmemOnStreamTester::preLaunchKernel() {
  bw_factor = 1;  // Broadcast is one-to-all

  for (int team_i = 0; team_i < num_teams; team_i++) {
    team_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &team_world_dup[team_i]);
    if (team_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      std::cerr << "Team " << team_i << " is invalid!" << std::endl;
      abort();
    }
  }
}

void TeamBroadcastmemOnStreamTester::postLaunchKernel() {
  // Synchronize all streams to ensure events are recorded
  for (int i = 0; i < num_teams; i++) {
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }

  // Get elapsed time for each work group from HIP events
  for (int wg_id = 0; wg_id < num_teams && wg_id < num_timers; wg_id++) {
    float elapsed_time_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_time_ms, start_events_timed[wg_id],
                                  stop_events_timed[wg_id]));

    // Convert milliseconds to GPU cycles
    // wall_clk_rate is in kHz, so: cycles = ms * wall_clk_rate
    long long int elapsed_cycles = static_cast<long long int>(
        elapsed_time_ms * static_cast<float>(wall_clk_rate));

    start_time[wg_id] = 0;
    end_time[wg_id] = elapsed_cycles;
  }

  // Fill remaining timers with zero if num_timers > num_teams
  for (int i = num_teams; i < num_timers; i++) {
    start_time[i] = 0;
    end_time[i] = 0;
  }

  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(team_world_dup[team_i]);
  }
}

void TeamBroadcastmemOnStreamTester::resetBuffers(size_t size) {
  // Initialize source buffer on all PEs
  // Each work group has its own portion
  for (int wg_id = 0; wg_id < num_teams; wg_id++) {
    int idx = wg_id * size;
    if (my_pe == pe_root) {
      // Root PE fills its source buffer with broadcast value
      int value = (pe_root + 1) * 100 + wg_id;
      std::memset(source_buf + idx, value, size);
    } else {
      // Non-root PEs source buffer (not used in broadcast)
      std::memset(source_buf + idx, 0xFF, size);
    }
  }

  // Initialize destination buffer on all PEs
  // Root PE keeps its initial dest value (broadcast doesn't copy to root's
  // dest) Non-root PEs set to 0 (will receive broadcast data)
  for (int wg_id = 0; wg_id < num_teams; wg_id++) {
    int idx = wg_id * size;
    if (my_pe == pe_root) {
      // Root PE's dest buffer stays with a different value
      int root_dest_value = 0xAA;
      std::memset(dest_buf + idx, root_dest_value, size);
    } else {
      std::memset(dest_buf + idx, 0, size);
    }
  }
}

void TeamBroadcastmemOnStreamTester::launchKernel(dim3 gridSize,
                                                  dim3 blockSize,
                                                  int loop,
                                                  size_t size) {
  // Execute warmup iterations (skip)
  for (int i = 0; i < args.skip; i++) {
    for (int wg_id = 0; wg_id < num_teams; wg_id++) {
      char *wg_source = source_buf + wg_id * size;
      char *wg_dest = dest_buf + wg_id * size;
      rocshmem_broadcastmem_on_stream(team_world_dup[wg_id], wg_dest,
                                      wg_source, size, pe_root, streams[wg_id]);
    }
  }

  for (int i = 0; i < num_teams; i++) {
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  }

  for (int i = 0; i < loop; i++) {
    for (int wg_id = 0; wg_id < num_teams; wg_id++) {
      // Record start event for this work group on first iteration
      if (i == 0) {
        CHECK_HIP(hipEventRecord(start_events_timed[wg_id], streams[wg_id]));
      }

      char *wg_source = source_buf + wg_id * size;
      char *wg_dest = dest_buf + wg_id * size;
      rocshmem_broadcastmem_on_stream(team_world_dup[wg_id], wg_dest,
                                      wg_source, size, pe_root, streams[wg_id]);

      // Record stop event for this work group on last iteration
      if (i == loop - 1) {
        CHECK_HIP(hipEventRecord(stop_events_timed[wg_id], streams[wg_id]));
      }
    }
  }

  num_msgs = (loop + args.skip) * num_teams;
  num_timed_msgs = loop * num_teams;
}

void TeamBroadcastmemOnStreamTester::verifyResults(size_t size) {
  // Verify correctness: after broadcast, non-root PEs receive the broadcast
  // data Root PE's dest buffer is NOT modified (per OpenSHMEM/rocSHMEM spec)
  for (int wg_id = 0; wg_id < num_teams; wg_id++) {
    int idx = wg_id * size;
    int expected_value;

    if (my_pe == pe_root) {
      // Root PE's dest buffer should remain unchanged (0xAA)
      expected_value = 0xAA;
    } else {
      // Non-root PEs should have received the broadcast value
      expected_value = (pe_root + 1) * 100 + wg_id;
    }

    for (size_t k = 0; k < size; k++) {
      if (static_cast<unsigned char>(dest_buf[idx + k]) !=
          static_cast<unsigned char>(expected_value)) {
        std::cerr << "PE " << my_pe << ": Verification failed for WG "
                  << wg_id << " at byte " << k << std::endl;
        std::cerr << "Expected value: " << expected_value
                  << ", Got: " << static_cast<int>(dest_buf[idx + k])
                  << std::endl;
        rocshmem_global_exit(1);
      }
    }
  }
}

