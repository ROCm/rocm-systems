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

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void TeamSyncTest(int loop, int skip, long long int *start_time,
                             long long int *end_time,
                             ShmemContextType ctx_type, TestType type,
                             int wf_size, rocshmem_team_t *teams) {
  __shared__ rocshmem_ctx_t ctx;
  int t_id  = get_flat_block_id();
  int wg_id = get_flat_grid_id();
  int wf_id = t_id / wf_size;

  rocshmem_wg_team_create_ctx(teams[wg_id], ctx_type, &ctx);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && hipThreadIdx_x == 0) {
      start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case TeamSyncTestType:
        if(t_id == 0) {
          rocshmem_ctx_sync(ctx, teams[wg_id]);
        }
        break;
      case TeamWAVESyncTestType:
        if(wf_id == 0) {
          rocshmem_ctx_sync_wave(ctx, teams[wg_id]);
        }
        break;
      case TeamWGSyncTestType:
        rocshmem_ctx_sync_wg(ctx, teams[wg_id]);
        break;
      default:
        break;
    }
    __syncthreads();
  }

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamSyncTester::TeamSyncTester(TesterArguments args) : Tester(args) {
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  CHECK_HIP(hipMalloc(&team_sync_world_dup,
                      sizeof(rocshmem_team_t) * num_teams));
}

TeamSyncTester::~TeamSyncTester() {
  CHECK_HIP(hipFree(team_sync_world_dup));
}

void TeamSyncTester::preLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    team_sync_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                 &team_sync_world_dup[team_i]);
    if (team_sync_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      printf("Team %d is invalid!\n", team_i);
      abort();
    }
  }
}

void TeamSyncTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(TeamSyncTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     _shmem_context, _type, wf_size,
                     team_sync_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void TeamSyncTester::postLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(team_sync_world_dup[team_i]);
  }
}

void TeamSyncTester::resetBuffers(size_t size) {}

void TeamSyncTester::verifyResults(size_t size) {}
