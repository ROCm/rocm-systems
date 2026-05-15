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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_TRANSPORT_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_TRANSPORT_HPP_

#include <cassert>

#include "rocshmem/rocshmem.hpp"
#include "backend_proxy.hpp"
#include "ro_net_team.hpp"
#include "mpi_instance.hpp"

namespace rocshmem {

class ROBackend;
class ROHostContext;

class Transport {
 public:
  virtual ~Transport() = default;

  virtual void initTransport(int num_queues, BackendProxyT *proxy) = 0;

  virtual void finalizeTransport() = 0;

  virtual void createNewTeam(ROBackend *backend_handle, Team *parent_team,
                             TeamInfo *team_info_wrt_parent,
                             TeamInfo *team_info_wrt_world, int num_pes,
                             int my_pe_in_new_team, MPI_Comm team_comm,
                             rocshmem_team_t *new_team) = 0;

  virtual void barrier(int wg_id, volatile char *status, bool blocking,
                       MPI_Comm team, bool quiet) = 0;

  virtual void team_reduction(void *dst, void *src, int size, int win_id,
                              int wg_id, MPI_Comm team, ROCSHMEM_OP op,
                              ro_net_types type, volatile char *status,
                              bool blocking) = 0;

  virtual void team_broadcast(void *dst, void *src, int size, int win_id,
                              int wg_id, MPI_Comm team, int PE_root,
                              ro_net_types type, volatile char *status,
                              bool blocking) = 0;

  virtual void alltoall(void *dst, void *src, int size, int win_id, int wg_id,
                        MPI_Comm team, void *ata_buffptr, ro_net_types type,
                        volatile char *status, bool blocking) = 0;

  virtual void fcollect(void *dst, void *src, int size, int win_id, int wg_id,
                        MPI_Comm team, void *ata_buffptr, ro_net_types type,
                        volatile char *status, bool blocking) = 0;

  virtual void putMem(void *dst, void *src, int size, int pe, int win_id,
                      int wg_id, volatile char *status, bool blocking,
                      bool inline_data = false) = 0;

  virtual void getMem(void *dst, void *src, int size, int pe, int win_id,
                      int wg_id, volatile char *status, bool blocking) = 0;

  virtual void amoFOP(void *dst, void *src, void *val, int pe, int win_id,
                      int wg_id, volatile char *status, bool blocking,
                      ROCSHMEM_OP op, ro_net_types type) = 0;

  virtual void amoFCAS(void *dst, void *src, void *val, int pe, int win_id,
                       int wg_id, volatile char *status, bool blocking,
                       void *cond, ro_net_types type) = 0;

  virtual bool readyForFinalize() = 0;

  virtual void quiet(int wg_id, volatile char *status) = 0;

  virtual void progress() = 0;

  virtual int numOutstandingRequests() = 0;

  virtual MPI_Comm get_world_comm() = 0;

  int getMyPe() const {
    assert(my_pe != -1);
    return my_pe;
  }

  int getNumPes() const {
    assert(num_pes != -1);
    return num_pes;
  }

  virtual void global_exit(int status) = 0;

  virtual void insertRequest(const queue_element_t *element, int queue_id) = 0;

 protected:
  int my_pe{-1};

  int num_pes{-1};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_TRANSPORT_HPP_
