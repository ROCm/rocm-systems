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

#include "reduce_on_stream_tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstring>
#include <vector>
#include <stdlib.h>

#define CHECKROCSHMEM(x)                                   \
do                                                         \
{                                                          \
  if ((x) != ROCSHMEM_SUCCESS)                             \
    throw std::runtime_error("CHECKROCSHMEM failed: " #x); \
} while (0);

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
ReduceOnStreamTester::ReduceOnStreamTester(TesterArguments args)
    : Tester(args) {
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  num_streams = args.num_wgs;

  buf_size = args.max_msg_size * n_pes * num_streams;

  source_buf = static_cast<int *>(alloc_test_buffer(buf_size * sizeof(int), args.local_buf_type));
  dest_buf   = static_cast<int *>(alloc_test_buffer(buf_size * sizeof(int)));

  team_world_dup.resize(num_streams);
  ctxs.resize(num_streams);
  streams.resize(num_streams);
  start_events_timed.resize(num_streams);
  stop_events_timed.resize(num_streams);

  
  for (int i = 0; i < num_streams; i++) {

    CHECK_HIP(hipStreamCreate(&streams[i]));
    CHECK_HIP(hipEventCreate(&start_events_timed[i]));
    CHECK_HIP(hipEventCreate(&stop_events_timed[i]));
  }
  
}

ReduceOnStreamTester::~ReduceOnStreamTester() {
  for (int i = 0; i < num_streams; i++) {
    CHECK_HIP(hipEventDestroy(stop_events_timed[i]));
    CHECK_HIP(hipEventDestroy(start_events_timed[i]));
    CHECK_HIP(hipStreamDestroy(streams[i]));
  }
  free_test_buffer(source_buf, args.local_buf_type);
  free_test_buffer(dest_buf);
}

void ReduceOnStreamTester::preLaunchKernel() {
  bw_factor = n_pes;

  for (int i = 0; i < num_streams; i++) {
    CHECKROCSHMEM(rocshmem_ctx_create(0, &ctxs[i]));
    team_world_dup[i] = ROCSHMEM_TEAM_INVALID;
    CHECKROCSHMEM(rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                &team_world_dup[i]));
    if (team_world_dup[i] == ROCSHMEM_TEAM_INVALID) {
      std::cerr << "Team " << i << " is invalid!" << std::endl;
      abort();
    }
  }
}

void ReduceOnStreamTester::postLaunchKernel() {
  for (int i = 0; i < num_streams; i++)
    CHECK_HIP(hipStreamSynchronize(streams[i]));
  
  for (int i =0; i < num_streams; i++)
    rocshmem_ctx_destroy(ctxs[i]);

  for (int i = 0; i < num_streams && i < static_cast<int>(num_timers); i++) {
    float elapsed_time_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_time_ms, start_events_timed[i],
                                  stop_events_timed[i]));

    long long int elapsed_cycles = static_cast<long long int>(
        elapsed_time_ms * static_cast<float>(wall_clk_rate));

    start_time[i] = 0;
    end_time[i] = elapsed_cycles;
  }

  for (int i = num_streams; i < static_cast<int>(num_timers); i++) {
    start_time[i] = 0;
    end_time[i] = 0;
  }

  for (int i = 0; i < num_streams; i++)
    rocshmem_team_destroy(team_world_dup[i]);
}

void ReduceOnStreamTester::resetBuffers([[maybe_unused]] size_t size) {
  for (int i = 0; i < static_cast<int>(buf_size); i++)
    source_buf[i] = 1;

  std::memset(dest_buf, 0, buf_size * sizeof(int));
}

void ReduceOnStreamTester::launchKernel([[maybe_unused]] dim3 gridSize,
                                        [[maybe_unused]] dim3 blockSize,
                                        int loop,
                                        size_t size) {
  for (int i = 0; i < args.skip; i++) {
    for (int s = 0; s < num_streams; s++) {
      int *wg_source = source_buf + s * n_pes * size;
      int *wg_dest   = dest_buf   + s * n_pes * size;
      CHECKROCSHMEM(rocshmem_ctx_int_sum_reduce_on_stream(ctxs[s],
                                            team_world_dup[s],
                                            wg_dest, wg_source, size,
                                            streams[s]));
    }
  }

  for (int s = 0; s < num_streams; s++)
    CHECK_HIP(hipStreamSynchronize(streams[s]));

  for (int i = 0; i < loop; i++) {
    for (int s = 0; s < num_streams; s++) {
      if (i == 0)
        CHECK_HIP(hipEventRecord(start_events_timed[s], streams[s]));

      int *wg_source = source_buf + s * n_pes * size;
      int *wg_dest   = dest_buf   + s * n_pes * size;
      CHECKROCSHMEM(rocshmem_ctx_int_sum_reduce_on_stream(ctxs[s],
                                            team_world_dup[s],
                                            wg_dest, wg_source, size,
                                            streams[s]));

      if (i == loop - 1)
        CHECK_HIP(hipEventRecord(stop_events_timed[s], streams[s]));
    }
  }

  num_msgs = (loop + args.skip) * num_streams;
  num_timed_msgs = loop * num_streams;
}

void ReduceOnStreamTester::verifyResults(size_t size) {
  for (int s = 0; s < num_streams; s++) {
    int *wg_dest = dest_buf + s * n_pes * size;
    for (size_t i = 0; i < size; i++) {
      if (wg_dest[i] != n_pes) {
        fprintf(stderr, "Data validation error at stream %d idx %zu: "
                "expected %d got %d\n", s, i, n_pes, wg_dest[i]);
        exit(-1);
      }
    }
  }
}
