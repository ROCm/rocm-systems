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

#ifndef LIBRARY_SRC_TEAM_TRACKER_HPP_
#define LIBRARY_SRC_TEAM_TRACKER_HPP_

/**
 * @file team_tracker.hpp
 * Defines the TeamTracker class
 */

#include <vector>

#include "rocshmem/rocshmem.hpp"
#include "envvar.hpp"

namespace rocshmem {

class Team;

/**
 * @class TeamTracker team_tracker.hpp
 *
 * @brief Container class for team information
 */
class TeamTracker {
 public:
  /**
   * @brief Primary constructor
   */
  TeamTracker();

  /**
   * @brief Add team from the list of user-created teams
   *
   * @param[in] team which needs to be added to tracker
   *
   * @param void
   */
  void track(rocshmem_team_t team);

  /**
   * @brief Remove team from the list of user-created teams
   *
   * @param[in] team which needs to be removed from tracker
   *
   * @return void
   */
  void untrack(rocshmem_team_t team);

  /**
   * @brief Remove all teams from the list of user-created teams
   *
   * @return void
   */
  template <typename FN_T>
  void destroy_all(FN_T&& team_destroy) {
    while (!teams_.empty()) {
      team_destroy(teams_.back());
      teams_.pop_back();
    }
  }

  /**
   * @brief Get the number of teams created by the user
   *
   * @return number of teams currently being tracked
   */
  int get_num_user_teams() { return teams_.size(); }

  /**
   * @brief Get maximum number of teams supported by tracker
   *
   * @return number of teams supported by tracker
   */
  __host__ __device__ int get_max_num_teams() { return max_num_teams_; } // TODO fix narrowing

  /**
   * @brief Get team world pointer
   *
   * @return team world pointer
   */
  __host__ Team* get_team_world() { return team_world_; }

  /**
   * @brief Set team world pointer
   *
   * @param[in] team_world pointer
   *
   * @return void
   */
  __host__ void set_team_world(Team* team_world) { team_world_ = team_world; }

 private:
  /**
   * @brief List of teams created by the user.
   */
  std::vector<rocshmem_team_t> teams_{};

  /**
   * @brief The maximum number of teams the user can create.
   *
   * This constraint is required since the library needs to
   * pre-allocate resources (e.g. LDS, working arrays, etc.)
   * for teams.
   */
  size_t max_num_teams_{envvar::max_num_teams};

  /**
   * @brief Pointer to implementation of ROCSHMEM_TEAM_WORLD
   */
  Team* team_world_{nullptr};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TEAM_TRACKER_HPP_
