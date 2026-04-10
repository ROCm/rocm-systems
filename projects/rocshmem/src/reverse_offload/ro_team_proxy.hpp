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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_RO_TEAM_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_RO_TEAM_PROXY_HPP_

#include "device_proxy.hpp"
#include "ro_net_team.hpp"
#include "mpi_instance.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class ROTeamProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, ROTeam>;

 public:
  /*
   * Placement new the memory which is allocated by proxy_
   */
  ROTeamProxy(Backend* backend, MPI_Comm comm, int pe, int npes,
              size_t num_elems = 1)
    : proxy_{num_elems} {

    mpilib_ftable_.Comm_dup(comm, &team_world_comm_);

    TeamInfo wrt_parent(nullptr, 0, 1, npes);
    TeamInfo wrt_world(nullptr, 0, 1, npes);

    new (proxy_.get()) ROTeam(backend, wrt_parent, wrt_world,
                              npes, pe, team_world_comm_);
  }

  ROTeamProxy(const ROTeamProxy& other) = delete;

  ROTeamProxy& operator=(const ROTeamProxy& other) = delete;

  ROTeamProxy(ROTeamProxy&& other) = default;

  ROTeamProxy& operator=(ROTeamProxy&& other) = default;

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~ROTeamProxy() {
    proxy_.get()->~ROTeam();

    mpilib_ftable_.Comm_free(&team_world_comm_);
  }

  /*
   * @brief Provide access to the memory referenced by the proxy
   */
  __host__ __device__ ROTeam* get() { return proxy_.get(); }

 private:
  /**
   * @brief Holds duplicated mpi world communicator.
   */
  MPI_Comm team_world_comm_;

  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};
};

using ROTeamProxyT = ROTeamProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_RO_TEAM_PROXY_HPP_
