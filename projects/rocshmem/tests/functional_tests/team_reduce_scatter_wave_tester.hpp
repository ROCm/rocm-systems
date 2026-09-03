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

#ifndef _TEAM_REDUCE_SCATTER_WAVE_TESTER_HPP_
#define _TEAM_REDUCE_SCATTER_WAVE_TESTER_HPP_

#include <functional>
#include <utility>

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
template <typename T1, ROCSHMEM_OP T2>
class TeamReduceScatterWaveTester : public Tester {
 public:
  explicit TeamReduceScatterWaveTester(
      TesterArguments args, std::function<void(T1 &, T1 &)> f1,
      std::function<std::pair<bool, std::string>(const T1 &, const T1 &)> f2);
  virtual ~TeamReduceScatterWaveTester();

  std::string typeName() const override { return type_name<T1>(); }

 protected:
  virtual void resetBuffers(uint64_t size) override;

  virtual void preLaunchKernel() override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) override;

  virtual void postLaunchKernel() override;

  virtual void verifyResults(uint64_t size) override;

  T1 *s_buf;  // source: n_pes * size elements per wave
  T1 *r_buf;  // dest:   size elements per wave

 private:
  int my_pe = 0;
  int n_pes = 0;

  /**
   * This constant should equal ROCSHMEM_MAX_NUM_TEAMS - 1.
   * The default value for the maximum number of teams is 40.
   */
  int num_teams = 39;
  rocshmem_team_t *team_reduce_scatter_wave_world_dup;

  std::function<void(T1 &, T1 &)> init_buf;
  std::function<std::pair<bool, std::string>(const T1 &, const T1 &)>
      verify_buf;
};

#include "team_reduce_scatter_wave_tester.cpp"

#endif  // _TEAM_REDUCE_SCATTER_WAVE_TESTER_HPP_
