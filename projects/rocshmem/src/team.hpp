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

#ifndef LIBRARY_SRC_TEAM_HPP_
#define LIBRARY_SRC_TEAM_HPP_

#include "rocshmem/rocshmem.hpp"
#include "rocshmem/rocshmem_mpi.hpp"
#include "backend_type.hpp"

namespace rocshmem {

class Backend;
class Team;
class ROTeam;
class IPCTeam;
class GDATeam;

class TeamInfo {
 public:
  /**
   * @brief Secondary constructor
   */
  __host__ __device__ TeamInfo() = default;

  /**
   * @brief Primary constructor
   */
  __host__ __device__ TeamInfo(Team* parent_team, int pe_start, int stride,
                               int size);

  /**
   * @brief The team from which this team was created.
   */
  Team* parent_team{nullptr};

  /**
   * @brief My position within the team.
   */
  int pe_start{-1};

  /**
   * @brief The stride used calculate team members.
   */
  int stride{-1};

  /**
   * @brief The log2 stride used to calculate team members.
   */
  double log_stride{-1};

  /**
   * @brief The size of this team.
   */
  int size{-1};
};

class Team {
 public:
  /**
   * @brief Constructor.
   *
   * @param handle The handle to the backend
   * @param team_info_wrt_parent information about this team wrt parent
   * @param team_info_wrt_world information about this team wrt TEAM_WORLD
   * @param num_pes number of PEs in this team
   * @param _my_pe the index of this PE in the team
   * @param _mpi_comm MPI Communicator representing the team
   */
  Team(Backend* handle, TeamInfo* team_info_wrt_parent,
       TeamInfo* team_info_wrt_world, int num_pes, int my_pe,
       MPI_Comm mpi_comm);

  /**
   * @brief Destructor.
   */
  virtual ~Team();

  /**
   * @brief Returns the corresponding PE in team world.
   *
   * @param[in] PE in my team.
   *
   * @return The PE of the process in team world.
   */
  __host__ __device__ int get_pe_in_world(int pe);

  /**
   * @brief Checks if a PE in team_world is in my team.
   *
   * @param[in] pe_in_world Index of a PE in team_world.
   *
   * @return The PE of the process in my team. -1 if not in my team.
   */
  __host__ __device__ int get_pe_in_my_team(int pe_in_world);

  /**
   * @brief Info about team world size.
   */
  int world_size{0};

  /**
   * @brief Info about this PE with respect to team_world.
   */
  int my_pe_in_world{-1};

  /**
   * @brief Info about team with respect to team_world.
   */
  TeamInfo* tinfo_wrt_world{nullptr};

  /**
   * @brief This team's info with respect to parent team.
   */
  TeamInfo* tinfo_wrt_parent{nullptr};

  /**
   * @brief The numbers of PEs within the team.
   */
  int num_pes{-1};

  /**
   * @brief My PE within the team.
   */
  int my_pe{-1};

  /**
   * @brief This teams mpi communicator.
   */
  MPI_Comm mpi_comm{MPI_COMM_NULL};

  /**
   * @brief The backend type.
   *
   * @note This is required to do some reinterpret_casts.
   */
  BackendType type{BackendType::RO_BACKEND};
};

__host__ __device__ Team* get_internal_team(rocshmem_team_t team);

ROTeam* get_internal_ro_team(rocshmem_team_t team);

IPCTeam* get_internal_ipc_team(rocshmem_team_t team);

GDATeam* get_internal_gda_team(rocshmem_team_t team);

__host__ __device__ int team_translate_pe(rocshmem_team_t src_team, int src_pe,
                                          rocshmem_team_t dst_team);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TEAM_HPP_
