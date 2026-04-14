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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_MPI_TRANSPORT_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_MPI_TRANSPORT_HPP_

#include <map>
#include <mutex>  // NOLINT
#include <queue>
#include <vector>
#include <thread>

#include "queue.hpp"
#include "transport.hpp"

namespace rocshmem {

class HostInterface;

class MPITransport : public Transport {
public:
  explicit MPITransport(MPI_Comm com, Queue* queue);

  virtual ~MPITransport();

  void initTransport(int num_queues, BackendProxyT *proxy) override;

  void finalizeTransport() override;

  void createNewTeam(ROBackend *backend, Team *parent_team,
                     TeamInfo *team_info_wrt_parent,
                     TeamInfo *team_info_wrt_world, int num_pes,
                     int my_pe_in_new_team, MPI_Comm team_comm,
                     rocshmem_team_t *new_team) override;

  void barrier(int contextId, volatile char *status, bool blocking,
               MPI_Comm team, bool quiet) override;

  void team_reduction(void *dst, void *src, int size, int win_id,
                      int contextId, MPI_Comm team, ROCSHMEM_OP op,
                      ro_net_types type, volatile char *status,
                      bool blocking) override;

  void team_broadcast(void *dst, void *src, int size, int win_id,
                      int contextId, MPI_Comm team, int PE_root,
                      ro_net_types type, volatile char *status,
                      bool blocking) override;

  void alltoall(void *dst, void *src, int size, int win_id, int contextId,
                MPI_Comm team, void *ata_buffptr, ro_net_types type,
                volatile char *status, bool blocking) override;

  void fcollect(void *dst, void *src, int size, int win_id, int contextId,
                MPI_Comm team, void *ata_buffptr, ro_net_types type,
                volatile char *status, bool blocking) override;

  void putMem(void *dst, void *src, int size, int pe, int win_id,
              int contextId, volatile char *status, bool blocking,
              bool inline_data = false) override;

  void amoFOP(void *dst, void *src, void *val, int pe, int win_id,
              int contextId, volatile char *status, bool blocking,
              ROCSHMEM_OP op, ro_net_types type) override;

  void amoFCAS(void *dst, void *src, void *val, int pe, int win_id,
               int contextId, volatile char *status, bool blocking, void *cond,
               ro_net_types type) override;

  void getMem(void *dst, void *src, int size, int pe, int win_id,
              int contextId, volatile char *status, bool blocking) override;

  void quiet(int contextId, volatile char *status) override;

  void progress() override;

  int numOutstandingRequests() override;

  void insertRequest(const queue_element_t *element, int queue_id) override;

  bool readyForFinalize() override { return !transport_up; }

  void global_exit(int status) override;

  MPI_Comm get_world_comm() override { return ro_net_comm_world; }

  HostInterface *host_interface{nullptr};

private:
  struct CommKey {
    CommKey(int _start, int _logPstride, int _size)
        : start(_start), logPstride(_logPstride), size(_size) {}

    bool operator<(const CommKey &key) const {
      return start < key.start ||
             (start == key.start && logPstride < key.logPstride) ||
             (start == key.start && logPstride == key.logPstride &&
              size < key.size);
    }

    int start{-1};
    int logPstride{-1};
    int size{-1};
  };

  struct RequestProperties {
    RequestProperties(volatile char *_status, int _contextId, bool _blocking,
                      void *_src, bool _inline_data)
        : status(_status),
          contextId(_contextId),
          blocking(_blocking),
          src(_src),
          inline_data(_inline_data) {}

    RequestProperties(volatile char *_status, int _contextId, bool _blocking)
        : status(_status),
          contextId(_contextId),
          blocking(_blocking),
          src(nullptr),
          inline_data(false) {}

    volatile char* status{nullptr};
    int contextId{-1};
    bool blocking{};
    void *src{nullptr};
    bool inline_data{};
  };

  struct Request {
    MPI_Request request;
    RequestProperties properties;
  };

  MPI_Comm createComm(int start, int logPstride, int size);

  void threadProgressEngine();

  void submitRequestsToMPI();

  MPI_Op get_mpi_op(ROCSHMEM_OP op);

  Queue *queue{nullptr};

  std::unique_ptr<MPI_Request[]> raw_requests();

  // Unordered vector of in-flight MPI Requests. Can complete out of order.
  std::vector<Request> requests{};

  std::vector<std::vector<volatile char *> > waiting_quiet{};

  std::vector<int> outstanding{};

  MPI_Comm ro_net_comm_world{MPI_COMM_NULL};

  std::map<CommKey, MPI_Comm> comm_map{};

  std::queue<queue_element_t> q{};

  std::queue<int> q_wgid{};

  std::mutex queue_mutex{};

  std::atomic<bool> transport_up{false};

  BackendProxyT *backend_proxy{nullptr};

  std::thread progress_thread{};

  std::array<int, 128> testsome_indices;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_MPI_TRANSPORT_HPP_
