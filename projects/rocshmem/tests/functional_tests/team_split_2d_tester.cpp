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

#include "team_split_2d_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamSplit2DTester::TeamSplit2DTester(TesterArguments args) : Tester(args) 
{
  my_pe = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD);
  n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  if (n_pes % xrange) {
    fprintf(stderr, 
      "Test needs world size divisible by %d, cannot continue test \n",
      xrange);
    exit(-1);
  }

  team_world_dup = ROCSHMEM_TEAM_INVALID;
  x_team = ROCSHMEM_TEAM_INVALID;
  y_team = ROCSHMEM_TEAM_INVALID;

  if (rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0, 
                                    &team_world_dup) != ROCSHMEM_SUCCESS){
    fprintf(stderr, 
            "ERROR: Unable to start test. Error in rocshmem_team_split_strided\n");
    exit(-1);
  }
}

TeamSplit2DTester::~TeamSplit2DTester() {
  rocshmem_team_destroy(team_world_dup);
}

void TeamSplit2DTester::resetBuffers([[maybe_unused]] size_t size) {}

void TeamSplit2DTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                     size_t size) {}

void TeamSplit2DTester::verifyResults([[maybe_unused]] size_t size) {
  if (test_failed) {
     fprintf(stderr,
             "PE %d FAIL: rocshmem_team_split_2d failed %d checks\n",
             my_pe, test_failed);
     exit(-1);
   }
   if (my_pe == 0) {
     printf("PASS: rocshmem_team_split_2d succeeded with %d PEs\n", n_pes);
   }
}

void TeamSplit2DTester::preLaunchKernel() {
  int ret = 0;
  x_team = ROCSHMEM_TEAM_INVALID;
  y_team = ROCSHMEM_TEAM_INVALID;
  ret = rocshmem_team_split_2d(team_world_dup, xrange, nullptr, 0, &x_team, 
                               nullptr, 0, &y_team);
  if (ret != 0){
    fprintf(stderr,
            "PE %d FAIL: rocshmem_team_split_2d returned %d (expected 0)\n",
            my_pe, ret);
    test_failed++;
  }
}

void TeamSplit2DTester::postLaunchKernel() {
  if (x_team == ROCSHMEM_TEAM_INVALID){
    fprintf(stderr, "PE %d FAIL: x_team is invalid\n", my_pe);
    test_failed++;
  }
  if (y_team == ROCSHMEM_TEAM_INVALID){
    fprintf(stderr, "PE %d FAIL: y_team is invalid\n", my_pe);
    test_failed++;
  }
  
  int x_size = rocshmem_team_n_pes(x_team);
  if (x_size != xrange){
    fprintf(stderr,
            "PE %d FAIL: xaxis_team size = %d, expected %d\n",
            my_pe, x_size, xrange);
    test_failed++;
  }

  int y_size = rocshmem_team_n_pes(y_team);
  if (y_size != n_pes / xrange){
    fprintf(stderr,
            "PE %d FAIL: yaxis_team size = %d, expected %d\n",
            my_pe, y_size, n_pes / xrange);
    test_failed++;
  }
  rocshmem_team_destroy(x_team);
  rocshmem_team_destroy(y_team);
}
