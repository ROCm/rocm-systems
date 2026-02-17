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

/* this constant should equal ROCSHMEM_MAX_NUM_TEAMS-1 */
#define NUM_TEAMS 39

rocshmem_team_t team_world_dup[NUM_TEAMS];

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

    if (my_pe != expected_pe) {
      printf("PE doesn't match. Expected %d got %d\n", expected_pe, my_pe);
      abort();
    }

    if (num_pes != expected_n_pes) {
      printf("Team size doesn't match. Expected %d got %d\n", expected_n_pes, num_pes);
      abort();
    }

    __syncthreads();

    rocshmem_ctx_quiet(ctx);
    rocshmem_wg_ctx_destroy(&ctx);
  }

 __global__ void TeamCtxInfraTest(ShmemContextType ctx_type,
                                 rocshmem_team_t *team) {
  __shared__ rocshmem_ctx_t ctx1, ctx2, ctx3;
  __shared__ rocshmem_ctx_t ctx[NUM_TEAMS];


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
  for (int team_i = 0; team_i < NUM_TEAMS; team_i++) {
    rocshmem_wg_team_create_ctx(team[team_i], ctx_type, &ctx[team_i]);
    if (nullptr == ctx[team_i].ctx_opaque) {
      printf("Create ctx on team[%d] returned an invalid context!\n", team_i);
      abort();
    }
  }

  if (ctx[0].team_opaque == ctx[NUM_TEAMS - 1].team_opaque) {
    printf("Incorrect for teams of ctx[0] and ctx[NUM_TEAMS-1] to be equal to each other\n");
    abort();
  }

  __syncthreads();

  for (int team_i = 0; team_i < NUM_TEAMS; team_i++) {
    rocshmem_wg_ctx_destroy(&ctx[team_i]);
  }

}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamCtxInfraTester::TeamCtxInfraTester(TesterArguments args) : Tester(args) {
  _splitType = args.team_type;
}

TeamCtxInfraTester::~TeamCtxInfraTester() {}

void TeamCtxInfraTester::resetBuffers(size_t size) {}

void TeamCtxInfraTester::preLaunchKernel() {
  int n_pes = rocshmem_team_n_pes(_parentTeam);
  int my_pe = rocshmem_team_my_pe(_parentTeam);

  if (_splitType == ROCSHMEM_TEST_TEAM_DUP) {
    // validate we can run the test
    if (auto maximum_num_contexts_str = getenv("ROCSHMEM_MAX_NUM_CONTEXTS")) {
      int max_ctx = atoi(maximum_num_contexts_str);
      if (max_ctx <= NUM_TEAMS) {
        printf("ROCSHMEM_MAX_NUM_CONTEXTS=%d is smaller than NUM_TEAMS %d, invalid test setup!\n", max_ctx, NUM_TEAMS);
        assert(max_ctx > NUM_TEAMS);
        abort();
      }
    }

    for (int team_i = 0; team_i < NUM_TEAMS; team_i++) {
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
  }
}

void TeamCtxInfraTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                      size_t size) {
  size_t shared_bytes = 0;

  /* Copy array of teams to device */
  rocshmem_team_t *teams_on_device;

  if (_splitType == ROCSHMEM_TEST_TEAM_DUP) {
    CHECK_HIP(hipMalloc(&teams_on_device, sizeof(rocshmem_team_t) * NUM_TEAMS));
    CHECK_HIP(hipMemcpy(teams_on_device, team_world_dup,
                        sizeof(rocshmem_team_t) * NUM_TEAMS, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(TeamCtxInfraTest, gridSize, blockSize, shared_bytes,
                       stream, _shmem_context, teams_on_device);
  } else if (_splitType == ROCSHMEM_TEST_TEAM_SINGLE ||
             _splitType == ROCSHMEM_TEST_TEAM_BLOCK  ||
             _splitType == ROCSHMEM_TEST_TEAM_ODDEVEN ) {
    CHECK_HIP(hipMalloc(&teams_on_device, sizeof(rocshmem_team_t)));
    CHECK_HIP(hipMemcpy(teams_on_device, team_world_dup,
                        sizeof(rocshmem_team_t), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(TeamCtxInfraSimpleTest, gridSize, blockSize, shared_bytes,
                       stream, _shmem_context, teams_on_device[0], _expected_pe, _expected_n_pes);
  }

  CHECK_HIP(hipFree(teams_on_device));
}

void TeamCtxInfraTester::postLaunchKernel() {
  int num_teams = _splitType == ROCSHMEM_TEST_TEAM_DUP ? NUM_TEAMS : 1;
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(team_world_dup[team_i]);
  }
}

void TeamCtxInfraTester::verifyResults(size_t size) {}
