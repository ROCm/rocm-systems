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

#ifndef LIBRARY_SRC_GDA_BACKEND_HPP_
#define LIBRARY_SRC_GDA_BACKEND_HPP_

#include <dlfcn.h>
#include <infiniband/verbs.h>

#include "backend_bc.hpp"
#include "containers/free_list_impl.hpp"
#include "hdp_proxy.hpp" //TODO useless?
#include "memory/hip_allocator.hpp"
#include "context_incl.hpp"
#include "gda_context_proxy.hpp"
#include "queue_pair.hpp"
#include "bootstrap/bootstrap.hpp"
#include "debug_gda.hpp"
#include "gda/ionic/provider_gda_ionic.hpp"
#include "gda/bnxt/provider_gda_bnxt.hpp"
#include "gda/mlx5/provider_gda_mlx5.hpp"

namespace rocshmem {

class GDAContext;
class GDAHostContext;
class QueuePair;
class HostInterface;

enum GDAProvider {
  UNSET,
  IONIC,
  BNXT,
  MLX5
};

inline constexpr uint32_t GDA_IONIC_VENDOR_ID = 0x1DD8;
inline constexpr uint32_t GDA_MLX5_VENDOR_ID  = 0x02c9; //PCI-ID is 15b3
inline constexpr uint32_t GDA_BNXT_VENDOR_ID  = 0x14E4;

class GDABackend : public Backend {
 private:
  typedef struct dest_info {
    int lid;
    int qpn;
    int psn;
    union ibv_gid gid;
  } dest_info_t;

  const char *requested_nic = nullptr;
  struct ibv_device *device = nullptr;
  struct ibv_context *context = nullptr;;
  struct ibv_device_attr device_attr;
  struct ibv_pd *pd_orig = nullptr;
  enum GDAProvider gda_provider = GDAProvider::UNSET;

  struct ibv_port_attr portinfo;
  union ibv_gid gid;
  int port = 1;
  int gid_index = 0;

  uint32_t *heap_rkey = nullptr;
  struct ibv_mr *heap_mr = nullptr;

  std::string debug_str;

  uint32_t inline_threshold = 8;
  QueuePair *host_qps = nullptr;
  QueuePair *gpu_qps = nullptr;
  std::vector<ibv_qp*> qps;
  std::vector<ibv_cq*> cqs;
  std::vector<dest_info_t> dest_info;

  /* GDA_BNXT START */
  std::vector<struct bnxt_host_qp> bnxt_qps;
  std::vector<struct bnxt_host_cq> bnxt_scqs;
  std::vector<struct bnxt_host_cq> bnxt_rcqs;
  /* GDA_BNXT END */

  /* GDA_IONIC & GDA_MLX5 START */
  struct ibv_pd *pd_parent = nullptr;
  /* GDA_IONIC & GDA_MLX5 END */

  /* GDA_IONIC START */
  struct ibv_pd *pd_uxdma[2];
  void *gpu_db_page = nullptr;
  uint64_t *gpu_db_cq = nullptr;
  uint64_t *gpu_db_sq = nullptr;
  /* GDA_IONIC END */

 /**
   * @brief Choose nic device according to locality/user preferences
   */
  void select_nic();

  /**
   * @brief return user-preferred GDA provider (or NONE if not specified)
   */
  static GDAProvider requested_provider();

 public:
  friend GDAContext;

  /**
   * @copydoc Backend::Backend(unsigned)
   */
  explicit GDABackend(MPI_Comm comm);
  explicit GDABackend(TcpBootstrap *bootstr);

  /**
   * @copydoc Backend::~Backend()
   */
  virtual ~GDABackend();

  /**
   * @brief Check if a device's vendor ID matches the expected vendor for a provider
   *
   * @param provider The GDA provider to check against
   * @param device_attr The device attributes containing the vendor ID
   * @param device_name The device name (for debug messages)
   * @return true if the device vendor matches the provider, false otherwise
   */
  static bool device_matches_provider_vendor(GDAProvider provider,
                                             const struct ibv_device_attr &device_attr,
                                             const char *device_name);

  /**
   * @brief Check if there are active InfiniBand/RDMA interfaces available
   *        that match the specified provider's vendor ID.
   *
   * @param provider The GDA provider to check for (BNXT, IONIC, or MLX5)
   * @return true if at least one active port on a matching vendor device is found,
   *         false otherwise
   */
  static bool has_active_ib_interface(GDAProvider provider);

  /**
   * @brief Verify whether GDA Backend could run
   *
   * @return ROSCHMEM_SUCCESS if GDA Backend can most likely be used
   *         ROCSHMEM_ERROR otherwise
   */
  static int backend_can_run(void);

  /**
   * @brief
   */
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
  char** get_wrk_sync_bases() { return wrk_sync_pool_bases_; } //TODO UNUSED

  /**
   * @brief The host-facing interface that will be used
   * by all contexts of the GDABackend
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
  std::unique_ptr<GDAHostContext> default_host_ctx{nullptr};

  /**
   * @brief Allocate and initialize team world.
   */
  void setup_team_world();

  /**
   * @brief Initialize the resources required to support teams
   */
  void setup_teams();

  /**
   * @brief Destruct the resources required to support teams
   */
  void cleanup_teams();

  /**
   * @brief Allocation and initialization of backend contexts.
   */
  void setup_ctxs();
  void cleanup_ctxs();
  void setup_host_ctx();
  void setup_default_ctx();


  /**
   * @brief Allocation and initialization of resources required to
   * support IPC handover.
   */
  void setup_ipc();
  void cleanup_ipc();

  /**
   * @brief Allocate and initialize barrier operation addresses on
   * symmetric heap.
   *
   * When this method completes, the barrier_sync member will be available
   * for use.
   */
  void setup_collectives();

  /**
   * @brief Allocate buffer for fence/quiet operation
   */
  void setup_fence_buffer();

  void setup_heap_memory_rkey();
  void cleanup_heap_memory_rkey();

  void initialize_gpu_qp(QueuePair* qp, int conn_num);
  void bnxt_initialize_gpu_qp(QueuePair* qp, int conn_num);
  void ionic_initialize_gpu_qp(QueuePair* qp, int conn_num);
  void mlx5_initialize_gpu_qp(QueuePair* qp, int conn_num);

  /**
   * @brief Setup InfiniBand Resources
   */
  void setup_ibv();

  /**
   * @brief Cleanup InfiniBand Resources
   */
  void cleanup_ibv();

  /**
   * @brief Detect and load the available direct verbs libraries
   */
  void open_dv_libs();

  /**
   * @ brief Close opened direct verbs libraries
   */
  void close_dv_libs();

  /**
   * @brief Open InfiniBand Device and create common structures
   */
  void open_ib_device();

  /**
   * @brief Validated the rocSHMEM will run with the currently open InfiniBand Device
   */
  void validate_ib_device();

  /**
   * @brief Selects the best GID index
   */
  void select_gid_index();

  /**
   * @brief Create all CQs and QPs
   */
  void create_queues();

  /**
   * @brief Create all CQs with a of length ncqes
   */
  void create_cqs(int ncqes);
  void bnxt_create_cqs(int ncqes);
  void ionic_create_cqs(int ncqes);

  /**
   * @brief Create all QPs with a SQ of length sq_length
   */
  void create_qps(int sq_length);
  void bnxt_create_qps(int sq_length);

  /**
   * @brief Reorders QPs to that we map rocSHMEM contexts to the correct QPs
   */
  void alternate_qp_ports();

  /**
   * @brief Exchange QP information for connection
   */
  void exchange_qp_dest_info();

  /**
   * @brief Modify all QPs from RESET to INIT state
   */
  void modify_qps_reset_to_init();

  /**
   * @brief Modify all QPs from INIT to RTR state
   */
  void modify_qps_init_to_rtr();

  /**
   * @brief Modify all QPs from RTR to RTs state
   */
  void modify_qps_rtr_to_rts();

  /**
   * @brief Converts an ibv_mtu to an integer
   */
  int ibv_mtu_to_int(enum ibv_mtu mtu);

  static void* pd_alloc_host(ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type);
  static void* pd_alloc_device_uncached(ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type);

  static void pd_release(ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type);

  void create_parent_domain();
  void ionic_setup_parent_domain(struct ibv_parent_domain_init_attr* pattr);

  void setup_gpu_qps();
  void cleanup_gpu_qps();

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
  GDADefaultContextProxyT default_context_proxy_;  // init handled in constructor

  /**
   * @brief An array of @ref ROContexts that backs the context FreeList.
   */
  GDAContext *ctx_array{nullptr};

  /**
   * @brief A free-list containing contexts.
   */
  FreeListProxy<HIPAllocator, GDAContext *> ctx_free_list{};

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
  HIPDefaultFinegrainedAllocator fine_grained_allocator_ {};

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
  char** wrk_sync_pool_bases_{nullptr};//TODO UNUSED, maybe used again later when we decouple the sync from the main heap

  /**
   * @brief Initialize memory required for work/sync buffers and open GDA
   * handle on PE's wrk_sync_pool.
   */
  void setup_wrk_sync_buffer();

  /**
   * @brief Close GDA memory handles for work/sync buffers and deallocate
   * work/sync buffer.
  */
  void cleanup_wrk_sync_buffer();

  /**
   * @brief rte all-to-all
   */
  void Alltoall_char_inplace (char *inoutbuf, size_t num_bytes, rocshmem_team_t team);

  /**
   * @brief rte allreduce for teams
   */
  void Allreduce_char_BAND (char* inbuf, char *outbuf, size_t num_bytes, Team *team);

  /**
   * @brief rte barrier for initialization
   */
  void rte_barrier();

  /**
   * @brief structures holding the function pointers to the direct verbs functionality
   * of each network driver.
   */
  bnxtdv_funcs_t bnxt_re_dv;

  /**
   * @brief handle used for the dlopen of the BCOM library
   */
  void *bnxtdv_handle_{nullptr};

  /**
   * @brief initialize function table for BCOM direct verbs support
   */
  int bnxt_dv_dl_init();

  /**
   * @brief open bnxt dv lib
   */
  static void* bnxt_dv_dlopen();

  /**
   * @brief structures holding the function pointers to the direct verbs functionality
   * of each network driver.
   */
  mlx5dv_funcs_t mlx5dv;

  /**
   * @brief handle used for the dlopen of the MLX5 library
   */
  void *mlx5dv_handle_{nullptr};

  /**
   * @brief initialize function table for MLNX direct verbs support
   */
  int mlx5_dv_dl_init();

  /**
   * @brief open mlx5 dv lib
   */
  static void* mlx5_dv_dlopen();

  /**
   * @brief structures holding the function pointers to the direct verbs functionality
   * of each network driver.
   */
  ionicdv_funcs_t ionic_dv;

  /**
   * @brief handle used for the dlopen of the IONIC library
   */
  void *ionicdv_handle_{nullptr};

  /**
   * @brief initialize function table for IONIC direct verbs support
   */
  int ionic_dv_dl_init();

  /**
   * @brief open ionic dv lib
   */
  static void* ionic_dv_dlopen();
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_BACKEND_HPP_
