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

#include "share_strategy.hpp"
#include "constants.hpp"

#include <hip/hip_runtime.h>

namespace rocshmem {

__device__ uint64_t Block::lane_id() {
  /*
   * amd_detail/device_functions.h::__lane_id()
   */
  return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

__device__ uint64_t Block::number_active_lanes() {
  /*
   * The __ballot(1) built-in instruction conducts an active lane roll
   * call storing the roll call result in a bit vector. Using its index
   * within the warp, each active lane contributes a '1' to the bit vector
   * Inactive lanes cannot contribute to the bit vector so their lane
   * values contain a '0'.
   *
   * For example, assume lanes 2, 5, and 62 active out of 64 lanes (0...63).
   * The result of the ballot would be the following:
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [0 0 1 0 0 1 0 ... 1 0]
   *
   * The __ballot(1) result is fed to __popcll.
   *
   * The __popcll instruction conducts a population count; it checks
   * the number of '1' values in a bit vector.
   *
   * Using the previous example, the result would be a 3 (since indices
   * 2, 5, and 62) contain a '1'.
   */
  return __popcll(__ballot(1));
}

__device__ bool Block::is_lowest_active_lane() {
  return active_logical_lane_id() == 0;
}

__device__ uint64_t Block::lowest_active_lane() {
  /*
   * The __ballot(1) built-in instruction conducts an active lane roll
   * call storing the roll call result in a bit vector. Using its index
   * within the warp, each active lane contributes a '1' to the bit vector
   * Inactive lanes cannot contribute to the bit vector so their lane
   * values contain a '0'.
   *
   * For example, assume lanes 2, 5, and 62 active out of 64 lanes (0...63).
   * The result of the ballot would be the following:
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [0 0 1 0 0 1 0 ... 1 0]
   *
   * The __ballot(1) result is fed to __ffsll.
   *
   * The __ffsll instruction finds the index of the least significant
   * bit set to '1' (in the input bit vector).
   *
   * In the previous example, the return result would be 2 (since index
   * 2 is the first bit with a '1').
   *
   * The '- 1' at the end is necessary because the return index is 1-based
   * instead of 0-based ([1...64] vs [0...63]).
   */
  return __ffsll(__ballot(1)) - 1;
}

__device__ uint64_t Block::active_logical_lane_id() {
  /*
   * The __ballot(1) built-in instruction conducts an active lane roll
   * call storing the roll call result in a bit vector. Using its index
   * within the warp, each active lane contributes a '1' to the bit vector
   * Inactive lanes cannot contribute to the bit vector so their lane
   * values contain a '0'.
   *
   * For example, assume lanes 2, 5, and 62 active out of 64 lanes (0...63).
   * The result of the ballot would be the following:
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [0 0 1 0 0 1 0 ... 1 0]
   */
  uint64_t ballot = __ballot(1);

  /*
   * The physical_lane_id is the warp lane index of the thread executing
   * this code. The word 'physical' here denotes that the lane_id will
   * be the actual hardware lane_id.
   */
  uint64_t my_physical_lane_id = lane_id();

  /*
   * Create a full bitset for subsequent operations.
   *
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [1 1 1 1 1 1 1 ... 1 1]
   */
  uint64_t all_ones_mask = -1;

  /*
   * Left-shift to zero-out the mask elements up to our lane_id.
   *
   * As an example, assume our lane_id is '5':
   *
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [0 0 0 0 0 1 1 ... 1 1]
   */
  uint64_t lane_mask = all_ones_mask << my_physical_lane_id;

  /*
   * Invert the lane_mask.
   *
   * Continue with lane_id '5' example:
   *
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [1 1 1 1 1 0 0 ... 0 0]
   */
  uint64_t inverted_mask = ~lane_mask;

  /*
   * Bit-wise And the inverted_mask and the ballot.
   *
   * The result contains a bitset with all active_lanes preceding this
   * thread in the ballot (all active threads with lower lane_ids).
   *
   * Continue with lane_id '5' example:
   *
   * ballot
   * ------------------------------
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [0 0 1 0 0 1 0 ... 1 0]
   *
   * inverted_mask
   * ------------------------------
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [1 1 1 1 1 0 0 ... 0 0]
   *
   * lower_active_lanes
   * ------------------------------
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [0 0 1 0 0 0 0 ... 0 0]
   */
  uint64_t lower_active_lanes = ballot & inverted_mask;

  /*
   * Conduct a population count on lower_active_lanes.
   *
   * The result gives an index into our logical_lane_id.
   *
   * Continue with lane_id '5' example:
   *
   * lower_active_lanes
   * ------------------------------
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   * value: [0 0 1 0 0 0 0 ... 0 0]
   *
   * my_logical_lane_id
   * ------------------------------
   *        [- - X - - - - ... - -] <- population_count = 1
   *
   * index:  0 0 0 0 0 0 0 ... 6 6
   *         0 1 2 3 4 5 6 ... 2 3
   *        [- - 0 - - 1 - ... 2 -] <- my_logical_lane_id = 1
   */
  uint64_t my_logical_lane_id = __popcll(lower_active_lanes);

  return my_logical_lane_id;
}

__device__ uint64_t Block::broadcast_up(uint64_t fetch_value) {
  for (unsigned i = 0; i < WF_SIZE; i++) {
    uint64_t temp = __shfl_up(fetch_value, i);
    if (temp) {
      fetch_value = temp;
    }
  }
  return fetch_value;
}

__device__ void ShareStrategy::syncthreads() const {
  switch (_sse) {
    case ShareStrategyEnum::PRIVATE:
      return;
    case ShareStrategyEnum::BLOCK:
      __syncthreads();
      return;
    case ShareStrategyEnum::DEVICE:
      abort();
      return;
    case ShareStrategyEnum::UNUSED:
      abort();
      return;
  }
}

}  // namespace rocshmem
