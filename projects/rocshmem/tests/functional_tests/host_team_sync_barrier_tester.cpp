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

#include "host_team_sync_barrier_tester.hpp"

#include <cstdlib>
#include <iostream>

using namespace rocshmem;

HostTeamSyncBarrierTester::HostTeamSyncBarrierTester(TesterArguments args)
    : Tester(args) {
  _type = HostTeamSyncBarrierTestType;
  _print_results = false;
}

HostTeamSyncBarrierTester::~HostTeamSyncBarrierTester() {}

void HostTeamSyncBarrierTester::resetBuffers([[maybe_unused]] size_t size) {}

void HostTeamSyncBarrierTester::launchKernel([[maybe_unused]] dim3 gridSize,
                                             [[maybe_unused]] dim3 blockSize,
                                             [[maybe_unused]] int loop,
                                             [[maybe_unused]] size_t size) {}

void HostTeamSyncBarrierTester::verifyResults([[maybe_unused]] size_t size) {
  int me = rocshmem_my_pe();
  int n = rocshmem_n_pes();
  if (n < 2) {
    if (me == 0) {
      std::cerr << "host team sync/barrier test requires >= 2 PEs\n";
    }
    abort();
  }

  // Split WORLD into even ranks, so odd ranks exercise INVALID no-op behavior.
  int even_size = (n + 1) / 2;
  rocshmem_team_t even_team = ROCSHMEM_TEAM_INVALID;
  int rc = rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 2, even_size,
                                       nullptr, 0, &even_team);
  if (rc != ROCSHMEM_SUCCESS) {
    std::cerr << "rocshmem_team_split_strided(even) failed rc=" << rc << "\n";
    abort();
  }

  if ((me % 2) == 0) {
    if (even_team == ROCSHMEM_TEAM_INVALID) {
      std::cerr << "even rank received TEAM_INVALID for even_team\n";
      abort();
    }
  } else if (even_team != ROCSHMEM_TEAM_INVALID) {
    std::cerr << "odd rank unexpectedly received valid even_team\n";
    abort();
  }

  // All ranks call with their local handle (valid or INVALID).
  rocshmem_team_sync(even_team);
  rocshmem_barrier(even_team);

  // Stream-ordered variants: even ranks enqueue a real team collective on the
  // stream; odd ranks pass ROCSHMEM_TEAM_INVALID and must no-op (nothing is
  // enqueued).  Synchronizing the stream then re-joining barrier_all confirms
  // the enqueued kernels completed without a hang.
  rocshmem_team_sync_on_stream(even_team, stream);
  rocshmem_barrier_on_stream(even_team, stream);
  CHECK_HIP(hipStreamSynchronize(stream));
  rocshmem_barrier_all();

  // Nested split inside the even team: team of size 1 containing even-team PE 0.
  rocshmem_team_t leader_team = ROCSHMEM_TEAM_INVALID;
  if ((me % 2) == 0) {
    rc = rocshmem_team_split_strided(even_team, 0, 1, 1, nullptr, 0,
                                     &leader_team);
    if (rc != ROCSHMEM_SUCCESS) {
      std::cerr << "rocshmem_team_split_strided(leader) failed rc=" << rc
                << "\n";
      abort();
    }
  }

  rocshmem_team_sync(leader_team);
  rocshmem_barrier(leader_team);

  // Stream-ordered on the single-PE leader team (and INVALID no-op for
  // non-leaders).
  rocshmem_team_sync_on_stream(leader_team, stream);
  rocshmem_barrier_on_stream(leader_team, stream);
  CHECK_HIP(hipStreamSynchronize(stream));

  if ((me % 2) == 0) {
    rocshmem_team_destroy(leader_team);
  }
  rocshmem_team_destroy(even_team);

  rocshmem_barrier_all();

  if (me == 0) {
    std::cout << "PASS: host team sync/barrier (blocking + on_stream) "
                 "runtime behavior validated\n";
  }
}
