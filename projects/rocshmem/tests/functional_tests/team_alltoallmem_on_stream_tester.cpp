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

#include "team_alltoallmem_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstring>
#include <cassert>
#include <vector>

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamAlltoallmemOnStreamTester::TeamAlltoallmemOnStreamTester(TesterArguments args)
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

  int num_bytes_wg = args.max_msg_size * n_pes;
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

TeamAlltoallmemOnStreamTester::~TeamAlltoallmemOnStreamTester() {
  for (int i = 0; i < num_teams; i++) {
    CHECK_HIP(hipEventDestroy(stop_events_timed[i]));
    CHECK_HIP(hipEventDestroy(start_events_timed[i]));
    CHECK_HIP(hipStreamDestroy(streams[i]));
  }
  rocshmem_free(source_buf);
  rocshmem_free(dest_buf);
}

void TeamAlltoallmemOnStreamTester::preLaunchKernel() {
  bw_factor = n_pes * n_pes;

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

void TeamAlltoallmemOnStreamTester::postLaunchKernel() {
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

void TeamAlltoallmemOnStreamTester::resetBuffers(size_t size) {
  // Initialize source buffer: each PE fills its portion with its PE number
  // For alltoall, PE i sends block j to PE j
  // Support multiple work groups (teams)
  int idx = 0;

  for (int wg_id = 0; wg_id < num_teams; wg_id++) {
    for (int pe = 0; pe < n_pes; pe++) {
      // Each block in source buffer is filled with (my_pe * n_pes + pe)
      // This makes it easy to verify correctness
      int value = my_pe * n_pes + pe;
      idx = (wg_id * n_pes + pe) * size;
      std::memset(source_buf + idx, value, size);
    }
  }

  // Clear destination buffer
  std::memset(dest_buf, 0, buf_size);
}

void TeamAlltoallmemOnStreamTester::launchKernel(dim3 gridSize,
                                                 dim3 blockSize,
                                                 int loop,
                                                 size_t size) {
  // Execute warmup iterations (skip)
  for (int i = 0; i < args.skip; i++) {
    for (int wg_id = 0; wg_id < num_teams; wg_id++) {
      char *wg_source = source_buf + wg_id * n_pes * size;
      char *wg_dest = dest_buf + wg_id * n_pes * size;
      rocshmem_alltoallmem_on_stream(team_world_dup[wg_id], wg_dest,
                                     wg_source, size, streams[wg_id]);
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

      char *wg_source = source_buf + wg_id * n_pes * size;
      char *wg_dest = dest_buf + wg_id * n_pes * size;
      rocshmem_alltoallmem_on_stream(team_world_dup[wg_id], wg_dest,
                                     wg_source, size, streams[wg_id]);

      // Record stop event for this work group on last iteration
      if (i == loop - 1) {
        CHECK_HIP(hipEventRecord(stop_events_timed[wg_id], streams[wg_id]));
      }
    }
  }

  num_msgs = (loop + args.skip) * num_teams;
  num_timed_msgs = loop * num_teams;
}

void TeamAlltoallmemOnStreamTester::verifyResults(size_t size) {
  // Verify correctness: after alltoall, PE i should receive from PE j
  // the block that PE j sent to PE i
  // PE j sends block i (containing value j * n_pes + i) to PE i
  // Support multiple work groups (teams)
  int idx = 0;

  for (int wg_id = 0; wg_id < num_teams; wg_id++) {
    for (int j = 0; j < n_pes; j++) {
      int expected_value = j * n_pes + my_pe;
      idx = (wg_id * n_pes + j) * size;

      for (size_t k = 0; k < size; k++) {
        if (static_cast<unsigned char>(dest_buf[idx + k]) !=
            static_cast<unsigned char>(expected_value)) {
          std::cerr << "PE " << my_pe << ": Verification failed for WG "
                    << wg_id << ", block from PE " << j << " at byte " << k
                    << std::endl;
          std::cerr << "Expected value: " << expected_value
                    << ", Got: " << static_cast<int>(dest_buf[idx + k])
                    << std::endl;
          rocshmem_global_exit(1);
        }
      }
    }
  }
}

