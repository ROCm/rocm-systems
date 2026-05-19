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

#include "team_ctx_infra_tester.hpp"

#include <rocshmem/rocshmem.hpp>

#include <cstdlib>
#include <cassert>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
  __global__ void TeamCtxInfraSimpleTest(ShmemContextType ctx_type,
                                         rocshmem_team_t team,
                                         int expected_pe, int expected_n_pes) {
    __shared__ rocshmem_ctx_t ctx;

    rocshmem_wg_team_create_ctx(team, ctx_type, &ctx);

    int num_pes = rocshmem_ctx_n_pes(ctx);
    int my_pe = rocshmem_ctx_my_pe(ctx);

    int team_pe = rocshmem_team_my_pe(team);
    int team_size = rocshmem_team_n_pes(team);

    if (my_pe != expected_pe || team_pe != expected_pe) {
      printf("PE doesn't match. Expected %d got (pe: %d, team_pe: %d)\n",
        expected_pe, my_pe, team_pe);
      abort();
    }

    if (num_pes != expected_n_pes || team_size != expected_n_pes) {
      printf("Team size doesn't match. Expected %d got (size: %d, "
        "team_size: %d)\n", expected_n_pes, num_pes, team_size);
      abort();
    }

    __syncthreads();

    rocshmem_ctx_quiet(ctx);
    rocshmem_wg_ctx_destroy(&ctx);
  }

 __global__ void TeamCtxInfraTest(ShmemContextType ctx_type,
                                rocshmem_team_t *team,
                                int num_teams) {
  __shared__ rocshmem_ctx_t ctx1, ctx2, ctx3;
  extern __shared__ rocshmem_ctx_t ctx[];


  /**
   * Test 1: Assert team infos of different ctxs
   * from the same team are the same.
   */

  rocshmem_wg_team_create_ctx(team[0], ctx_type, &ctx1);
  if (nullptr == ctx1.ctx_opaque) {
    printf("Create ctx1 on team[0] returned an invalid context!\n");
    abort();
  }
  rocshmem_wg_team_create_ctx(team[0], ctx_type, &ctx2);
  if (nullptr == ctx2.ctx_opaque) {
    printf("Create ctx2 on team[0] returned an invalid context!\n");
    abort();
  }
  rocshmem_wg_ctx_destroy(&ctx1);
  rocshmem_wg_team_create_ctx(team[0], ctx_type, &ctx3);
  if (nullptr == ctx3.ctx_opaque) {
    printf("Create ctx3 on team[0] returned an invalid context!\n");
    abort();
  }

  __syncthreads();

  if (ctx3.team_opaque != ctx2.team_opaque) {
    printf("Incorrect for teams of ctx2 and ctx3 to be different!\n");
    abort();
  }

  rocshmem_wg_ctx_destroy(&ctx2);
  rocshmem_wg_ctx_destroy(&ctx3);

  __syncthreads();

  /**
   * Test 2: Assert team infos of different ctxs
   * from different teams are different.
   */
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_wg_team_create_ctx(team[team_i], ctx_type, &ctx[team_i]);
    if (nullptr == ctx[team_i].ctx_opaque) {
      printf("Create ctx on team[%d] returned an invalid context!\n", team_i);
      abort();
    }
  }

  if (ctx[0].team_opaque == ctx[num_teams - 1].team_opaque) {
    printf("Incorrect for teams of ctx[0] and ctx[num_teams-1] to be equal to each other\n");
    abort();
  }

  __syncthreads();

  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_wg_ctx_destroy(&ctx[team_i]);
  }

}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamCtxInfraTester::TeamCtxInfraTester(TesterArguments args) : Tester(args) {
  _splitType = args.team_type;

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
    if (num_teams < 1) {
      printf("ROCSHMEM_MAX_NUM_TEAMS must be >= 1; got %d\n", num_teams);
      abort();
    }
  }

  CHECK_HIP(hipMalloc(&team_world_dup,
                      sizeof(rocshmem_team_t) * num_teams));
}

TeamCtxInfraTester::~TeamCtxInfraTester() {
  CHECK_HIP(hipFree(team_world_dup));
}

void TeamCtxInfraTester::resetBuffers([[maybe_unused]] size_t size) {}

void TeamCtxInfraTester::preLaunchKernel() {
  int n_pes = rocshmem_team_n_pes(_parentTeam);
  int my_pe = rocshmem_team_my_pe(_parentTeam);

  if (_splitType == ROCSHMEM_TEST_TEAM_DUP) {
    if (num_teams < 2) {
      printf("ROCSHMEM_TEST_TEAM_DUP requires num_teams >= 2; got %d\n", num_teams);
      abort();
    }

    if (auto maximum_num_contexts_str = getenv("ROCSHMEM_MAX_NUM_CONTEXTS")) {
      int max_ctx = atoi(maximum_num_contexts_str);
      if (max_ctx <= num_teams) {
        printf("ROCSHMEM_MAX_NUM_CONTEXTS=%d is smaller than num_teams %d, invalid test setup!\n", max_ctx, num_teams);
        assert(max_ctx > num_teams);
        abort();
      }
    }

    for (int team_i = 0; team_i < num_teams; team_i++) {
      team_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
      rocshmem_team_split_strided(_parentTeam, 0, 1, n_pes, nullptr, 0,
                                  &team_world_dup[team_i]);
      if (team_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
        printf("Created team %d is invalid!\n", team_i);
        abort();
      }
    }

    /* Assert the failure of a new team creation. */
    rocshmem_team_t new_team = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(_parentTeam, 0, 1, n_pes, nullptr, 0,
                                &new_team);
    if (new_team != ROCSHMEM_TEAM_INVALID) {
      printf("Created new team should have been invalid!\n");
      abort();
    }
  }
  else if (_splitType == ROCSHMEM_TEST_TEAM_SINGLE) {
    rocshmem_team_split_strided(_parentTeam, my_pe, 1, 1, nullptr, 0,
                                &team_world_dup[0]);
    _expected_pe = rocshmem_team_my_pe(team_world_dup[0]);
    _expected_n_pes = rocshmem_team_n_pes(team_world_dup[0]);

    if (_expected_n_pes != 1) {
      printf("ROCSHMEM_TEST_TEAM_SINGLE: n_pes %d expected: 1\n", _expected_n_pes);
      abort();
    }

    if (_expected_pe != 0) {
      printf("ROCSHMEM_TEST_TEAM_SINGLE: my_pe %d expected: 0\n", _expected_pe);
      abort();
    }
  } else if (_splitType == ROCSHMEM_TEST_TEAM_BLOCK) {
    int mid_pe = n_pes / 2; // integer division
    int start_pe  = my_pe < mid_pe ? 0 : mid_pe;
    int end_pe = my_pe < mid_pe ? (mid_pe - 1) : (n_pes - 1);
    int num_pes = end_pe - start_pe + 1;
    int new_pe =  my_pe < mid_pe ? my_pe : (my_pe - start_pe);

    rocshmem_team_split_strided(_parentTeam, start_pe, 1, num_pes, nullptr, 0,
                                &team_world_dup[0]);
    _expected_pe = rocshmem_team_my_pe(team_world_dup[0]);
    _expected_n_pes = rocshmem_team_n_pes(team_world_dup[0]);

    if (_expected_n_pes != num_pes) {
      printf("ROCSHMEM_TEST_TEAM_BLOCK: n_pes %d expected: %d\n", _expected_n_pes, num_pes);
      abort();
    }

    if (_expected_pe != new_pe) {
      printf("ROCSHMEM_TEST_TEAM_BLOCK: my_pe %d expected: %d\n", _expected_pe, new_pe);
      abort();
    }
  } else if (_splitType == ROCSHMEM_TEST_TEAM_ODDEVEN) {
    int start_pe = (my_pe % 2) == 0 ? 0 : 1;
    int num_pes = n_pes / 2;
    if (((n_pes % 2) != 0) && ((my_pe % 2) == 0))
      num_pes++;
    int new_pe = (my_pe / 2);

    rocshmem_team_split_strided(_parentTeam, start_pe, 2, num_pes, nullptr, 0,
                                &team_world_dup[0]);
    _expected_pe = rocshmem_team_my_pe(team_world_dup[0]);
    _expected_n_pes = rocshmem_team_n_pes(team_world_dup[0]);

    if (_expected_n_pes != num_pes) {
      printf("ROCSHMEM_TEST_TEAM_ODDEVEN: n_pes %d expected: %d\n", _expected_n_pes, num_pes);
      abort();
    }

    if (_expected_pe != new_pe) {
      printf("ROCSHMEM_TEST_TEAM_ODDEVEN: my_pe %d expected: %d\n", _expected_pe, new_pe);
      abort();
    }
  } else if (_splitType == ROCSHMEM_TEST_TEAM_SHARED) {
    if (ROCSHMEM_TEAM_SHARED == ROCSHMEM_TEAM_INVALID) {
      printf("ROCSHMEM_TEAM_SHARED is TEAM_INVALID (IPC disabled), skipping\n");
      _skip_shared = true;
      return;
    }
    team_world_dup[0] = ROCSHMEM_TEAM_SHARED;
    _expected_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_SHARED);
    _expected_n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_SHARED);
  } else if (_splitType == ROCSHMEM_TEST_TEAM_SUBSET_PARENT) {
    // First create a subset parent team via parity-based partition:
    // even PEs create team {0,2,4,...}, odd PEs create team {1,3,5,...}
    int start_pe = (my_pe % 2) == 0 ? 0 : 1;
    int num_pes = n_pes / 2;
    if (((n_pes % 2) != 0) && ((my_pe % 2) == 0))
      num_pes++;

    rocshmem_team_split_strided(_parentTeam, start_pe, 2, num_pes, nullptr, 0,
                                &subset_parent_team);

    if (subset_parent_team == ROCSHMEM_TEAM_INVALID) {
      printf("ROCSHMEM_TEST_TEAM_SUBSET_PARENT: Failed to create subset parent team!\n");
      abort();
    }

    // Now split this subset parent team into two halves
    int subset_n_pes = rocshmem_team_n_pes(subset_parent_team);
    int subset_my_pe = rocshmem_team_my_pe(subset_parent_team);
    int mid_pe = subset_n_pes / 2;
    int subset_start_pe  = subset_my_pe < mid_pe ? 0 : mid_pe;
    int subset_end_pe = subset_my_pe < mid_pe ? (mid_pe - 1) : (subset_n_pes - 1);
    int subset_num_pes = subset_end_pe - subset_start_pe + 1;
    int new_pe =  subset_my_pe < mid_pe ? subset_my_pe : (subset_my_pe - subset_start_pe);

    rocshmem_team_split_strided(subset_parent_team, subset_start_pe, 1, subset_num_pes, nullptr, 0,
                                &team_world_dup[0]);
    _expected_pe = rocshmem_team_my_pe(team_world_dup[0]);
    _expected_n_pes = rocshmem_team_n_pes(team_world_dup[0]);

    if (_expected_n_pes != subset_num_pes) {
      printf("ROCSHMEM_TEST_TEAM_SUBSET_PARENT: n_pes %d expected: %d\n", _expected_n_pes, subset_num_pes);
      abort();
    }

    if (_expected_pe != new_pe) {
      printf("ROCSHMEM_TEST_TEAM_SUBSET_PARENT: my_pe %d expected: %d\n", _expected_pe, new_pe);
      abort();
    }
  }
}

void TeamCtxInfraTester::launchKernel(dim3 gridSize, dim3 blockSize, [[maybe_unused]] int loop,
                                      [[maybe_unused]] size_t size) {
  size_t shared_bytes = 0;

  if (_skip_shared) return;

  if (_splitType == ROCSHMEM_TEST_TEAM_DUP) {
    shared_bytes = sizeof(rocshmem_ctx_t) * num_teams;

    hipLaunchKernelGGL(TeamCtxInfraTest, gridSize, blockSize, shared_bytes,
                       stream, _shmem_context, team_world_dup, num_teams);
  } else if (_splitType == ROCSHMEM_TEST_TEAM_SINGLE ||
             _splitType == ROCSHMEM_TEST_TEAM_BLOCK  ||
             _splitType == ROCSHMEM_TEST_TEAM_ODDEVEN ||
             _splitType == ROCSHMEM_TEST_TEAM_SHARED ||
             _splitType == ROCSHMEM_TEST_TEAM_SUBSET_PARENT ) {
    hipLaunchKernelGGL(TeamCtxInfraSimpleTest, gridSize, blockSize, shared_bytes,
                       stream, _shmem_context, team_world_dup[0], _expected_pe, _expected_n_pes);
  }
}

void TeamCtxInfraTester::postLaunchKernel() {
  if (_splitType == ROCSHMEM_TEST_TEAM_SHARED || _skip_shared) return;

  int teams_to_destroy = _splitType == ROCSHMEM_TEST_TEAM_DUP ? num_teams : 1;
  for (int team_i = 0; team_i < teams_to_destroy; team_i++) {
    rocshmem_team_destroy(team_world_dup[team_i]);
  }

  // Destroy the subset parent team if it was created
  if (_splitType == ROCSHMEM_TEST_TEAM_SUBSET_PARENT &&
      subset_parent_team != ROCSHMEM_TEAM_INVALID) {
    rocshmem_team_destroy(subset_parent_team);
    subset_parent_team = ROCSHMEM_TEAM_INVALID;
  }
}

void TeamCtxInfraTester::verifyResults([[maybe_unused]] size_t size) {}
