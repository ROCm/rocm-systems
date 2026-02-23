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

#ifndef LIBRARY_SRC_IPC_BACKEND_HPP_
#define LIBRARY_SRC_IPC_BACKEND_HPP_

#include "backend_bc.hpp"
#include "containers/free_list_impl.hpp"
#include "hdp_proxy.hpp"
#include "memory/hip_allocator.hpp"
#include "context_incl.hpp"
#include "ipc_context_proxy.hpp"
#include "ipc_policy.hpp"
#include "bootstrap/bootstrap.hpp"

namespace rocshmem {

class IPCBackend : public Backend {
  const unsigned MAX_NUM_BLOCKS{65536};

 public:
  /**
   * @copydoc Backend::Backend(unsigned)
   */
  explicit IPCBackend(MPI_Comm comm);
  explicit IPCBackend(TcpBootstrap *bootstr);

  /**
   * @copydoc Backend::~Backend()
   */
  virtual ~IPCBackend();

  __device__ bool create_ctx(int64_t options, rocshmem_ctx_t *ctx);

  /**
   * @brief Destroy a `rocshmem_ctx_t` context and returns it back to the
   * context free list.
   */
  __device__ void destroy_ctx(rocshmem_ctx_t *ctx);

  /**
   * @copydoc Backend::ctx_create
   */
  void ctx_create(int64_t options, void **ctx) override;

  /**
   * @copydoc Backend::ctx_destroy
   */
  void ctx_destroy(Context *ctx) override;

  /**
   * @brief Helper to initialize IPC interface.
   */
  void initIPC();

  /**
   * @brief Helper to initialize IPC interface, non-MPI based version.
   */
  void initIPC(TcpBootstrap *bootstrap);

  /**
   * @brief Allocation and initialization of backend contexts.
   */
  void setup_ctxs();

  /**
   * @brief Abort the application.
   *
   * @param[in] status Exit code.
   *
   * @return void.
   *
   * @note This routine terminates the entire application.
   */
  void global_exit(int status) override;

  /**
   * @copydoc Backend::create_new_team
   */
  void create_new_team(Team *parent_team, TeamInfo *team_info_wrt_parent,
                       TeamInfo *team_info_wrt_world, int num_pes,
                       int my_pe_in_new_team, MPI_Comm team_comm,
                       rocshmem_team_t *new_team) override;

  /**
   * @copydoc Backend::team_destroy(rocshmem_team_t)
   */
  void team_destroy(rocshmem_team_t team) override;

  /**
   * @brief Accessor for work/sync bases
   *
   * @return Vector containing the addresses of the work/sync bases
   */
  char** get_wrk_sync_bases() { return wrk_sync_pool_bases_; }

  /**
   * @brief The host-facing interface that will be used
   * by all contexts of the IPCBackend
   */
  std::shared_ptr<HostInterface> host_interface{nullptr};

  /**
   * @brief Scratchpad for the internal barrier algorithms.
   */
  int64_t *barrier_sync{nullptr};

  /**
   * @brief Handle for raw memory for barrier sync
   */
  long *barrier_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for reduce sync
   */
  long *reduce_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for broadcast sync
   */
  long *bcast_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for alltoall sync
   */
  long *alltoall_pSync_pool{nullptr};

  /**
   * @brief Handle for raw memory for work
   */
  void *pWrk_pool{nullptr};

  /**
   * @brief Handle for raw memory for alltoall
   */
  void *pAta_pool{nullptr};

  /**
   * @brief Handle for raw memory for fence/quiet
  */
  int *fence_pool{nullptr};

  /**
   * @brief Check whether all PEs are on a single node
   */
   static int backend_can_run(MPI_Comm comm, TcpBootstrap *bootstrap);

 protected:
   /**
   * @copydoc Backend::dump_backend_stats()
   */
  void dump_backend_stats() override;

  /**
   * @copydoc Backend::reset_backend_stats()
   */
  void reset_backend_stats() override;

  /**
   * @brief Allocates uncacheable host memory for the hdp policy.
   *
   * @note Internal data ownership is managed by the proxy
   */
  HdpProxy<HIPHostAllocator> hdp_proxy_{};

  /**
   * @brief Holds a copy of the default context for host functions
   */
  std::unique_ptr<IPCHostContext> default_host_ctx{nullptr};

  /**
   * @brief Allocate and initialize team world.
   */
  void setup_team_world();

  /**
   * @brief Initialize the resources required to support teams
   */
  void teams_init();

  /**
   * @brief Destruct the resources required to support teams
   */
  void teams_destroy();

  /**
   * @brief Allocate and initialize barrier operation addresses on
   * symmetric heap.
   *
   * When this method completes, the barrier_sync member will be available
   * for use.
   */
  void rocshmem_collective_init();

  /**
   * @brief Allocate buffer for fence/quiet operation
   */
  void setup_fence_buffer();

 private:
  /**
   * @brief Common code invoked from the different constructors
   */
  void init();

  /**
   * @brief Proxy for the default context
   *
   * @note Internal data ownership is managed by the proxy
   */
  IPCDefaultContextProxyT default_context_proxy_;  // init handled in constructor

  /**
   * @brief An array of @ref ROContexts that backs the context FreeList.
   */
  IPCContext *ctx_array{nullptr};

  /**
   * @brief A free-list containing contexts.
   */
  FreeListProxy<HIPAllocator, IPCContext *> ctx_free_list{};

  /**
   * @brief The bitmask representing the availability of teams in the pool
   */
  char *team_pool_bitmask_{nullptr};

  /**
   * @brief Bitmask to store the reduced result of bitmasks on pariticipating
   * PEs
   *
   * With no thread-safety for this bitmask, multithreaded creation of teams is
   * not supported.
   */
  char *team_reduced_bitmask_{nullptr};

  /**
   * @brief Size of the bitmask
   */
  int team_bitmask_size_{-1};

  /**
   * Fine grained memory allocator for buffers used in collectives Routines
   */
  MemoryAllocator *fine_grained_allocator_{nullptr};

  /**
   * @brief Collective routines work/sync buffer size
   */
  size_t wrk_sync_pool_size_{};

  /**
   * @brief Collective routines work/sync buffer base ptr
   */
  char* const wrk_sync_pool_{nullptr};

  /**
   * @brief Temporary buffer pointer pointing to the same address as
   * wrk_sync_pool_, used to calculate the starting addresses of
   * different work and sync buffers.
  */
  char *wrk_sync_pool_top_{nullptr};

  /**
   * @brief Array containing the addresses of the work/sync buffer bases
   * of other PEs
  */
  char** wrk_sync_pool_bases_{nullptr};

  /**
   * @brief Initialize memory required for work/sync buffers and open IPC
   * handle on PE's wrk_sync_pool.
   */
  void setup_wrk_sync_buffers();

  /**
   * @brief Close IPC memory handles for work/sync buffers and deallocate
   * work/sync buffer.
  */
  void cleanup_wrk_sync_buffer();

  /**
   * @brief
   */
  void Allreduce_char_BAND (char* inbuf, char *outbuf, size_t num_bytes, Team *team);

};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_BACKEND_HPP_
