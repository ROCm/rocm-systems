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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP_

#include <memory>
#include <vector>

#include "backend_bc.hpp"
#include "containers/free_list_impl.hpp"
#include "hdp_proxy.hpp"
#include "memory/hip_allocator.hpp"
#include "backend_proxy.hpp"
#include "block_handle.hpp"
#include "context_proxy.hpp"
#include "mpi_transport.hpp"
#include "profiler.hpp"
#include "queue.hpp"
#include "ro_team_proxy.hpp"
#include "team_info_proxy.hpp"
#include "window_proxy.hpp"

namespace rocshmem {

class HostInterface;
class ROHostContext;

/**
 * @class ROBackend backend.hpp
 * @brief Reverse Offload Transports specific backend.
 *
 * The Reverse Offload (RO) backend class forwards device network requests to
 * the host (which allows the device to initiate network requests).
 * The word, "Reverse", denotes that the device is doing the offloading to
 * the host (which is an inversion of the normal behavior).
 */
class ROBackend : public Backend {
  using RetBufferProxyT = DeviceProxy<HIPAllocator, uint64_t>;
  using StatusProxyT =
          DeviceProxy<HIPDefaultFinegrainedAllocator, char>;

 public:
  /**
   * @copydoc Backend::Backend(unsigned)
   */
  explicit ROBackend(MPI_Comm comm);

  /**
   * @copydoc Backend::~Backend()
   */
  virtual ~ROBackend();

  /**
   * @brief Verify whether RO Backend could run
   *
   * @return ROSCHMEM_SUCCESS if RO backend can most likely be used
   *         ROCSHMEM_ERROR otherwise
   */
  static int backend_can_run(void);

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
   * @brief Free all resources associated with the backend.
   *
   * The memory allocated to the handle param is deallocated during this
   * method. The handle should be treated as a nullptr after the call.
   *
   * The destructor treats this method as a helper function to destroy
   * this object.
   *
   * @todo The method needs to be broken into smaller pieces and most
   * of these internal resources need to be moved into subclasses using
   * RAII.
   */
  void ro_net_free_runtime();

  /**
   * @brief The host-facing interface that will be used
   * by all contexts of the ROBackend
   */
  HostInterface *host_interface{nullptr};

  /**
   * @brief Handle to device memory fields.
   */
  BackendProxyT backend_proxy{};

  /**
   * @brief Handle to block resources
   */
  BlockHandleProxyT block_handle_proxy_;

  /**
   * @brief Handle to block resources
   */
  DefaultBlockHandleProxyT default_block_handle_proxy_;

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
   * @brief Service thread routine which spins on a number of queues until
   * the host calls net_finalize.
   *
   * @todo Fix the assumption that only one gpu device exists in the
   * node.
   */
  void ro_net_poll();

  /**
   * @brief Helper to initialize IPC interface.
   */
  void initIPC();

  /**
   * @brief Allocation and initialization of backend contexts.
   */
  void setup_ctxs();

  /**
   * @brief Allocation and initialization of buffers for default context.
   */
  void setup_default_ctx_buffers();

  /**
   * @brief Handle for the transport class object.
   *
   * See the transport class for more details.
   */
  MPITransport *transport_;

  /**
   * @brief Proxy for the team info used by the device.
   *
   * See the transport class for more details.
   */
  ROTeamProxyT *team_world_proxy_;

  /**
   * @brief Workers used to poll on the device network request queues.
   */
  std::thread worker_thread{};

  /**
   * @brief Holds a copy of the default context for host functions
   */
  std::unique_ptr<ROHostContext> default_host_ctx{nullptr};

 public:
  /**
   * @brief Pool of contexts for RO_NET
   */
  WindowProxyT *ro_window_proxy_;

 protected:
  /**
   * @brief Allocates uncacheable host memory for the hdp policy.
   *
   * @note Internal data ownership is managed by the proxy
   */
  HdpProxy<HIPHostAllocator> hdp_proxy_{};

  /**
   * @brief Handle to device profiler memory
   *
   * @note Internal data ownership is managed by the proxy
   */
  ProfilerProxyT profiler_proxy_;  // init handled in constructor

 public:
  /**
   * @brief Handle to network queues.
   */
  Queue queue_;

 protected:
  /**
   * @brief Proxy for the default context
   *
   * @note Internal data ownership is managed by the proxy
   */
  DefaultContextProxyT default_context_proxy_;  // init handled in constructor

  /**
   * @brief Controls how many thread blocks are monitored by polling thread.
   */
  size_t poll_block_count_{1};

 private:
  /**
   * @brief An array of @ref ROContexts that backs the context FreeList.
   */
  ROContext *ctx_array{nullptr};

  /**
   * @brief A free-list containing contexts.
   */
  FreeListProxy<HIPAllocator, ROContext *> ctx_free_list{};

  /**
   * @brief AtomicWFQueue containing status flag buffers for default context
   */
  AtomicWFQueueProxy<HIPAllocator, volatile char*> default_ctx_status_{};

  /**
   * @brief AtomicWFQueue containing rocshmem_g return buffers for default
   * context
   */
  AtomicWFQueueProxy<HIPAllocator, uint64_t*> default_ctx_g_ret_buffer_{};

  /**
   * @brief AtomicWFQueue containing rocshmem return buffers for default
   * context
   */
  AtomicWFQueueProxy<HIPAllocator, uint64_t*> default_ctx_atomic_ret_buffer_{};

  /**
   * @brief Holds maximum threads per work-group
   */
  int max_wg_size_{};

  /**
   * @brief Holds wavefront size
   */
  int wf_size_{};

  /**
   * @brief Holds the queue size for each context
  */
  size_t queue_size_{512};

  /**
   * @brief Number of MPI windows used for device contexts in RO Backend
   */
  size_t num_windows_{32};

  /**
   * @brief Return buffer for rocshmem_g API
   */
  RetBufferProxyT g_ret_buffer_;
  RetBufferProxyT g_ret_buffer_default_ctx_;

  /**
   * @brief Return buffer for rocshmem atomic return APIs
   */
  RetBufferProxyT atomic_ret_buffer_;
  RetBufferProxyT atomic_ret_buffer_default_ctx_;

  /**
   * This buffer is used by the GPU to wait on a blocking operation. The initial
   * value is 0. When a GPU enqueues a blocking operation, it waits for this
   * value to resolve to 1, which is set by the CPU when the blocking
   * operation completes. The GPU then resets status back to zero. There is
   * a separate status variable for each work-item in a RO Context
   */
  StatusProxyT status_;
  StatusProxyT status_default_ctx_;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP_
