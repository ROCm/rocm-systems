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

#include "ipc_team.hpp"

#include "constants.hpp"
#include "backend_type.hpp"
#include "backend_ipc.hpp"

namespace rocshmem {

IPCTeam::IPCTeam(Backend *backend, TeamInfo *team_info_parent,
                 TeamInfo *team_info_world, int num_pes, int my_pe,
                 MPI_Comm mpi_comm, int pool_index)
    : Team(backend, team_info_parent, team_info_world, num_pes, my_pe,
           mpi_comm) {
  type = BackendType::IPC_BACKEND;
  const IPCBackend *b = static_cast<const IPCBackend *>(backend);

  pool_index_ = pool_index;

  barrier_pSync = &(b->barrier_pSync_pool[pool_index * ROCSHMEM_BARRIER_SYNC_SIZE]);
  reduce_pSync = &(b->reduce_pSync_pool[pool_index * ROCSHMEM_REDUCE_SYNC_SIZE]);
  bcast_pSync = &(b->bcast_pSync_pool[pool_index * ROCSHMEM_BCAST_SYNC_SIZE]);
  alltoall_pSync = &(b->alltoall_pSync_pool[pool_index * ROCSHMEM_ALLTOALL_SYNC_SIZE]);

  pWrk = reinterpret_cast<char *>(b->pWrk_pool) + ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) * pool_index;
}

IPCTeam::~IPCTeam() {}

}  // namespace rocshmem
