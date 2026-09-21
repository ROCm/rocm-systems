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

#ifndef _HOST_TEAM_SYNC_BARRIER_TESTER_HPP
#define _HOST_TEAM_SYNC_BARRIER_TESTER_HPP

#include "tester.hpp"

/**
 * Host-side functional validation for the team-scoped barrier/sync APIs:
 *   - blocking:       rocshmem_barrier(team), rocshmem_team_sync(team)
 *   - stream-ordered: rocshmem_barrier_on_stream(team, stream),
 *                     rocshmem_team_sync_on_stream(team, stream)
 */
class HostTeamSyncBarrierTester : public Tester {
 public:
  explicit HostTeamSyncBarrierTester(TesterArguments args);
  ~HostTeamSyncBarrierTester() override;

 protected:
  void resetBuffers(size_t size) override;
  void launchKernel(dim3 gridSize, dim3 blockSize, int loop, size_t size) override;
  void verifyResults(size_t size) override;
};

#endif  // _HOST_TEAM_SYNC_BARRIER_TESTER_HPP
