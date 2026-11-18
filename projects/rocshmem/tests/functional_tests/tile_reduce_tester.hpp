/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef TILE_REDUCE_TESTER_HPP
#define TILE_REDUCE_TESTER_HPP

#include <functional>
#include <utility>
#include <string>

#include "tester.hpp"

#include <rocshmem/rocshmem.hpp>
#include "../../src/context_incl.hpp"
#include <rocshmem/rocshmem_TILE_impl.hpp>

/******************************************************************************
 * HOST TESTER CLASS
 *
 * Templated on element type T and reduction operation Op.
 * Times a single type+op combination per instantiation, matching the pattern
 * used by TeamReductionTester.  Verification is performed on-device via
 * error_flag; the host only checks the flag in verifyResults.
 ******************************************************************************/
template <typename T, ROCSHMEM_OP Op>
class TileReduceTester : public Tester {
 public:
  explicit TileReduceTester(
      TesterArguments args,
      std::function<void(T &, T &)> f1,
      std::function<bool(T, int, int)> f2);

  virtual ~TileReduceTester();

 protected:
  virtual void resetBuffers(size_t size) override;
  virtual void preLaunchKernel() override;
  virtual void launchKernel(dim3 gridSize, dim3 blockSize,
                             int loop, size_t size) override;
  virtual void postLaunchKernel() override;
  virtual void verifyResults(size_t size) override;

  // Tile dimensions — set from max_msg_size at construction time.
  static constexpr int DEFAULT_TILE_ROWS = 8;
  static constexpr int DEFAULT_TILE_COLS = 8;

  int tile_extent_0 = DEFAULT_TILE_ROWS;
  int tile_extent_1 = DEFAULT_TILE_COLS;

  // Symmetric buffers: source tile(s) and result tile(s).
  T *s_buf = nullptr;
  T *r_buf = nullptr;

  rocshmem_team_t *teams = nullptr;
  int num_teams = 0;

  int *error_flag = nullptr;

 private:
  std::function<void(T &, T &)> init_buf;
  std::function<bool(T, int, int)> verify_buf;  // (value, n_pes, linear_idx)
};

#include "tile_reduce_tester.cpp"

#endif  // TILE_REDUCE_TESTER_HPP
