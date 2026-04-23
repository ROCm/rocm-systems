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

#include <hip/hip_runtime.h>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <cassert>
#include <algorithm>

#include "backend_gda.hpp"
#include "debug_gda.hpp"
#include "ibv_wrapper.hpp"
#include "envvar.hpp"
#include "gda_team.hpp"
#include "log.hpp"
#include "mpi_instance.hpp"
#include "util.hpp"
#include "topology.hpp"

namespace rocshmem {

#define NET_CHECK(cmd) {                                     \
    if (cmd != MPI_SUCCESS) {                                \
      LOG_ERROR_EXIT("Unrecoverable error: MPI Failure");    \
    }                                                        \
  }

extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

rocshmem_team_t get_external_team(GDATeam *team) {
  return reinterpret_cast<rocshmem_team_t>(team);
}

static int get_ls_non_zero_bit(char *bitmask, int mask_length) {
  int position{-1};
  for (int bit_i = 0; bit_i < mask_length; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    if (bitmask[byte_i] & (1 << (bit_i % CHAR_BIT))) {
      position = bit_i;
      break;
    }
  }

  return position;
}

GDABackend::GDABackend(MPI_Comm comm):  Backend(comm) {
  init();
}

GDABackend::GDABackend(TcpBootstrap *bootstrap):  Backend(bootstrap) {
  init();
}

void GDABackend::init() {

  type = BackendType::GDA_BACKEND;

  // Initialize QP allocator to finegrained allocator
  qp_allocator_ = new HIPAllocatorFinegrained();

  select_nics();

  qps_per_pe_default_ctx_ = envvar::gda::num_qps_per_pe_default_ctx.get_value();
  qps_per_pe_usr_ctx_     = envvar::gda::num_qps_per_pe_usr_ctx.get_value();

  // Determine number of QPs to create per PE
  num_qps_per_pe = qps_per_pe_default_ctx_ +
                   qps_per_pe_usr_ctx_ * envvar::max_num_contexts;

  // Total number of QPs created
  num_qps = num_qps_per_pe * num_pes;

  configure_nic_policy();

  LOG_TRACE("PE %d QP config: num_nics=%d, qps_per_pe_default_ctx=%zu, "
            "qps_per_pe_usr_ctx=%zu, num_qps_per_pe=%zu, num_qps=%u, "
            "nic_policy=%s",
            my_pe, num_nics_, qps_per_pe_default_ctx_, qps_per_pe_usr_ctx_,
            num_qps_per_pe, num_qps,
            nic_policy_ == NicPolicy::PER_CONTEXT ? "PER_CONTEXT"
                                                  : "ROUND_ROBIN");

  //TODO setup_host_interface();
  /* Initialize the host interface */
  if (MPI_COMM_NULL != backend_comm)
    host_interface = std::make_shared<HostInterface>(hdp_proxy_.get(), //TODO: need an hdp proxy?
                                                     backend_comm,
                                                     &heap);
  else
    host_interface = std::make_shared<HostInterface>(hdp_proxy_.get(), //TODO: need an hdp proxy?
                                                     backend_bootstr,
                                                     &heap);

  setup_wrk_sync_buffer();
  setup_fence_buffer();
  setup_collectives();

  setup_teams();
  setup_team_world();
  rte_barrier();

  setup_ipc();

  /*
   * setup_team_shared() must follow setup_ipc() because it uses
   * ipcImpl.pes_with_ipc_avail to determine shared-memory membership.
   */
  setup_team_shared();

  setup_ibv();
  setup_heap_memory_rkey();
  setup_gpu_qps();

  setup_ctxs();
  rte_barrier();
}

GDABackend::~GDABackend() {
  cleanup_ctxs();

  cleanup_teams();

  auto *team_shared{static_cast<GDATeam*>(team_tracker.get_team_shared())};
  if (team_shared) {
    team_shared->~GDATeam();
    CHECK_HIP(hipFree(team_shared));
  }

  auto *team_world{static_cast<GDATeam*>(team_tracker.get_team_world())};
  team_world->~GDATeam();
  CHECK_HIP(hipFree(team_world));

  cleanup_wrk_sync_buffer();

  cleanup_ipc();

  cleanup_gpu_qps();
  cleanup_heap_memory_rkey();
  cleanup_ibv();

  close_dv_libs();

  // Cleanup QP allocator
  if (qp_allocator_ != nullptr) {
    delete qp_allocator_;
    qp_allocator_ = nullptr;
  }
}

void GDABackend::configure_nic_policy() {
  const std::string &policy_str = envvar::gda::nic_policy.get_value();

  std::string policy_upper = policy_str;
  std::transform(policy_upper.begin(), policy_upper.end(), policy_upper.begin(),
                 ::toupper);

  if (policy_upper == "PER_CONTEXT") {
    nic_policy_ = NicPolicy::PER_CONTEXT;
  } else if (policy_upper == "ROUND_ROBIN") {
    nic_policy_ = NicPolicy::ROUND_ROBIN;
  } else {
    LOG_WARN("Unknown NIC_POLICY '%s', using ROUND_ROBIN", policy_str.c_str());
    nic_policy_ = NicPolicy::ROUND_ROBIN;
  }

  if (nic_policy_ == NicPolicy::ROUND_ROBIN && num_nics_ > 1) {
    int limit = static_cast<int>(
        std::max(qps_per_pe_default_ctx_, qps_per_pe_usr_ctx_));
    if (limit < 1) limit = 1;
    if (limit < num_nics_) {
      LOG_TRACE("ROUND_ROBIN limiting num_nics from %d to %d "
                "(max qps_per_pe=%d)",
                num_nics_, limit, limit);
      nic_devices_.resize(limit);
      num_nics_ = limit;
    }
  }
}

void GDABackend::select_nics() {
  const std::string &force_merge = envvar::gda::net_force_merge.get_value();
  const std::string &merge_level_str = envvar::gda::net_merge_level.get_value();
  bool use_force_merge = !force_merge.empty();
  bool use_auto_merge  = envvar::gda::merge_nics;

  int gpu_dev = 0;
  CHECK_HIP(hipGetDevice(&gpu_dev));

  const char *hca_list = envvar::hca_list.get_value().c_str();
  std::vector<std::string> nic_names;

  if (!envvar::requested_nic.is_default()) {
    nic_names.push_back(envvar::requested_nic.get_value());
  } else if (use_force_merge) {
    std::string my_group = SelectRankGroup(force_merge, my_pe);
    nic_names = ParseNicList(my_group);
    if (nic_names.empty()) {
      LOG_ERROR_EXIT("ROCSHMEM_GDA_NET_FORCE_MERGE is set but contains no valid"
                     " NIC names for PE %d: '%s'",
                     my_pe, force_merge.c_str());
    }
  } else if (use_auto_merge) {
    auto merge_level = rocshmem::ParseNicMergeLevel(merge_level_str);

    int found = rocshmem::GetClosestNicsToGpu(gpu_dev, hca_list, merge_level,
                                              nic_names);

    if (found <= 0) {
      LOG_ERROR_EXIT("NIC fusion enabled but no NICs found (merge_level=%s)",
                     merge_level_str.c_str());
    }
  } else {
    std::string name;
    if (rocshmem::GetClosestNicToGpu(gpu_dev, hca_list, &name) >= 0) {
      nic_names.push_back(name);
    }
  }

  if (nic_names.empty()) {
    LOG_ERROR_EXIT("No NIC found for PE %d (GPU %d)", my_pe, gpu_dev);
  }

  nic_devices_.resize(nic_names.size());
  for (size_t i = 0; i < nic_names.size(); i++) {
    nic_devices_[i].nic_name = nic_names[i];
  }
  num_nics_ = static_cast<int>(nic_devices_.size());

  {
    std::string nic_list;
    for (int i = 0; i < num_nics_; i++) {
      nic_list += " " + nic_devices_[i].nic_name;
    }
    LOG_INFO("PE %d GPU %d selected %d NIC(s):%s", my_pe, gpu_dev, num_nics_,
             nic_list.c_str());
  }
}

void GDABackend::setup_ipc() {
  const auto &heap_bases{heap.get_heap_bases()};

  if (MPI_COMM_NULL != backend_comm)
    ipcImpl.ipcHostInit(my_pe, heap_bases, backend_comm);
  else
    ipcImpl.ipcHostInit(my_pe, heap_bases, backend_bootstr);
}

void GDABackend::cleanup_ipc() {
  ipcImpl.ipcHostStop();
}

void GDABackend::setup_host_ctx() {
  default_host_ctx = std::make_unique<GDAHostContext>(this, 0);
  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();
}

void GDABackend::setup_default_ctx() {
  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;
  default_context_proxy_ = GDADefaultContextProxyT(this, tinfo, gda_provider);
}

void GDABackend::log_ctx_nics([[maybe_unused]] unsigned int ctx_id,
                               size_t qps_per_pe, int qp_offset) {
  std::string nic_list;
  for (size_t r = 0; r < qps_per_pe; r++) {
    int nidx = nic_idx_for_qp(qp_offset + static_cast<int>(r) * num_pes);
    if (r) nic_list += " ";
    nic_list += nic_devices_[nidx].nic_name;
  }
  LOG_TRACE("PE %d ctx %u qps_per_pe=%zu NICs=[%s]", my_pe, ctx_id, qps_per_pe,
            nic_list.c_str());
}

void GDABackend::setup_ctxs() {
  setup_host_ctx();
  setup_default_ctx();

  bool verbose = envvar::debug_level.get_value() >= envvar::types::debug_level::TRACE;
  if (verbose) log_ctx_nics(0, qps_per_pe_default_ctx_, 0);

  CHECK_HIP(hipMalloc(&ctx_array, sizeof(GDAContext) * envvar::max_num_contexts));
  // 0th context is default context
  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    unsigned int cid = static_cast<unsigned int>(i + 1);
    new (&ctx_array[i]) GDAContext(this, cid, gda_provider);
    ctx_free_list.get()->push_back(ctx_array + i);

    if (verbose) {
      int offset = (qps_per_pe_default_ctx_ +
                    qps_per_pe_usr_ctx_ * (cid - 1)) * num_pes;
      log_ctx_nics(cid, qps_per_pe_usr_ctx_, offset);
    }
  }

  rocshmem_ctx_t *rocshmem_ctx_array_device = nullptr;
  rocshmem_ctx_t *rocshmem_ctx_array_ptr = nullptr;
  size_t ctx_array_size = sizeof(rocshmem_ctx_t) * envvar::max_num_contexts;

  CHECK_HIP(hipMalloc((void**)&rocshmem_ctx_array_device, ctx_array_size));

  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    rocshmem_ctx_array_device[i].ctx_opaque  = &ctx_array[i];
    rocshmem_ctx_array_device[i].team_opaque = team_tracker.get_team_world()->tinfo_wrt_world;
  }

  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&rocshmem_ctx_array_ptr),
                                HIP_SYMBOL(rocshmem_ctx_array)));

  CHECK_HIP(hipMemcpy(rocshmem_ctx_array_ptr,
                      &rocshmem_ctx_array_device,
                      sizeof(rocshmem_ctx*),
                      hipMemcpyDefault));
}

void GDABackend::cleanup_ctxs() {
  /* Free ctx array */
  rocshmem_ctx_t *rocshmem_ctx_array_ptr = nullptr;
  rocshmem_ctx_t *rocshmem_ctx_array_device = nullptr;

  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&rocshmem_ctx_array_ptr),
                                HIP_SYMBOL(rocshmem_ctx_array)));

  CHECK_HIP(hipMemcpy(&rocshmem_ctx_array_device,
                      rocshmem_ctx_array_ptr,
                      sizeof(rocshmem_ctx*),
                      hipMemcpyDefault));

  CHECK_HIP(hipFree(rocshmem_ctx_array_device));

  ctx_free_list.~FreeListProxy();
  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    ctx_array[i].~GDAContext();
  }

  CHECK_HIP(hipFree(ctx_array));
}

__device__ bool GDABackend::create_ctx([[maybe_unused]] int64_t options, rocshmem_ctx_t *ctx) {
  GDAContext *ctx_{nullptr};

  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;

  ctx_->tinfo = reinterpret_cast<TeamInfo *>(ctx->team_opaque);
  return true;
}

__device__ void GDABackend::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<GDAContext *>(ctx->ctx_opaque));
}

void GDABackend::setup_team_world() {
  TeamInfo team_info_wrt_parent(nullptr, 0, 1, num_pes);
  TeamInfo team_info_wrt_world(nullptr, 0, 1, num_pes);

  GDATeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(GDATeam)));
  new (team_world) GDATeam(this, team_info_wrt_parent, team_info_wrt_world,
                           num_pes, my_pe, backend_comm, 0);
  team_tracker.set_team_world(team_world);

  /**
   * Copy the address to ROCSHMEM_TEAM_WORLD.
   */
  host::ROCSHMEM_TEAM_WORLD = reinterpret_cast<rocshmem_team_t>(team_world);
  set_team_world_device(host::ROCSHMEM_TEAM_WORLD);
}

void GDABackend::setup_team_shared() {
#if defined(USE_IPC)
  if (ipcImpl.pes_with_ipc_avail == nullptr) {
    host::ROCSHMEM_TEAM_SHARED = ROCSHMEM_TEAM_INVALID;
    set_team_shared_device(ROCSHMEM_TEAM_INVALID);
    return;
  }

  int shm_size = ipcImpl.shm_size;
  int shm_rank = ipcImpl.shm_rank;

  /*
   * Determine pe_start/stride from the IPC PE list. The list is on
   * device memory, so copy it to host for inspection.
   */
  std::vector<int> team_shared_pes(shm_size);
  CHECK_HIP(hipMemcpy(team_shared_pes.data(), ipcImpl.pes_with_ipc_avail,
                       shm_size * sizeof(int), hipMemcpyDeviceToHost));

  int pe_start = team_shared_pes[0];
  int stride = (shm_size > 1) ? (team_shared_pes[1] - team_shared_pes[0]) : 1;
  bool uniform = (stride > 0);
  for (int i = 2; i < shm_size && uniform; i++) {
    if (team_shared_pes[i] - team_shared_pes[i - 1] != stride) {
      uniform = false;
    }
  }

  if (!uniform) {
    /*
     * Node-local ranks are not uniformly strided, so TEAM_SHARED
     * cannot be represented with pe_start/stride. Mark it invalid
     * since context-based operations rely on the strided formula.
     */
    host::ROCSHMEM_TEAM_SHARED = ROCSHMEM_TEAM_INVALID;
    set_team_shared_device(ROCSHMEM_TEAM_INVALID);
    return;
  }

  TeamInfo team_info_wrt_parent(nullptr, 0, 1, shm_size);
  TeamInfo team_info_wrt_world(nullptr, pe_start, stride, shm_size);

  GDATeam *team_shared{nullptr};
  CHECK_HIP(hipMalloc(&team_shared, sizeof(GDATeam)));
  new (team_shared) GDATeam(this, team_info_wrt_parent, team_info_wrt_world,
                             shm_size, shm_rank, MPI_COMM_NULL, 1);

  team_tracker.set_team_shared(team_shared);

  host::ROCSHMEM_TEAM_SHARED = reinterpret_cast<rocshmem_team_t>(team_shared);
  set_team_shared_device(host::ROCSHMEM_TEAM_SHARED);
#else
  host::ROCSHMEM_TEAM_SHARED = ROCSHMEM_TEAM_INVALID;
  set_team_shared_device(ROCSHMEM_TEAM_INVALID);
#endif
}

void GDABackend::team_destroy(rocshmem_team_t team) {
  GDATeam *team_obj = get_internal_gda_team(team);

  /* Mark the pool as available */
  int bit = team_obj->pool_index_;
  int byte_i = bit / CHAR_BIT;
  team_pool_bitmask_[byte_i] |= 1 << (bit % CHAR_BIT);

  team_obj->~GDATeam();
  CHECK_HIP(hipFree(team_obj));
}

//TODO: factorize somewhere else maybe backend_bc
void GDABackend::Alltoall_char_inplace (char *inoutbuf, size_t num_bytes, rocshmem_team_t team) {
  // Implement an Alltoall outside of MPI assuming in_place communication
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  std::vector<int> pes_in_world;

  for (int i = 0; i < num_pes; i++) {
    pes_in_world.push_back(team_obj->get_pe_in_world(i));
  }
  backend_bootstr->groupAlltoall(inoutbuf, num_bytes, pes_in_world);
}

//TODO: factorize somewhere else, maybe backend_bc?
void GDABackend::Allreduce_char_BAND (char* inbuf, char *outbuf, size_t num_bytes,
                                      Team *team) {

  // Implement an Allreduce outside of MPI. This is specialized for the scenario
  // required for the team creation, i.e. assuming bytes and using BAND operation.
  // Implementation uses an Allgather operation followed a local reduction.

  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int num_pes = team_obj->num_pes;
  std::vector<int> pes_in_world;

  char *tmp_buffer = new char[num_pes * num_bytes];
  std::memset(tmp_buffer, 0, num_pes * num_bytes);
  std::memcpy(&tmp_buffer[my_pe * num_bytes], inbuf, num_bytes);

  for (int i = 0; i < num_pes; i++) {
    pes_in_world.push_back(team_obj->get_pe_in_world(i));
  }
  backend_bootstr->groupAllGather(tmp_buffer, num_bytes, pes_in_world);

  for (size_t i = 0; i < num_bytes; i++) {
    outbuf[i] = tmp_buffer[i];
    for (int j = 1; j < num_pes; j++) {
      outbuf[i] &= tmp_buffer[j * num_bytes + i];
    }
  }

  delete[] tmp_buffer;
}

void GDABackend::create_new_team([[maybe_unused]] Team *parent_team,
                                const TeamInfo& team_info_wrt_parent,
                                const TeamInfo& team_info_wrt_world,
                                int num_pes, int my_pe_in_new_team,
                                MPI_Comm team_comm,
                                rocshmem_team_t *new_team) {
  /**
   * Read the bit mask and find out a common index into
   * the pool of available work arrays.
   */
  if (team_comm != MPI_COMM_NULL) {
    NET_CHECK(mpilib_ftable_.Allreduce(team_pool_bitmask_, team_reduced_bitmask_, team_bitmask_size_,
                            MPI_CHAR, MPI_BAND, team_comm));
  } else {
    Allreduce_char_BAND (team_pool_bitmask_, team_reduced_bitmask_, team_bitmask_size_, parent_team);
  }

  /* Pick the least significant non-zero bit (logical layout) in the reduced
   * bitmask */
  auto max_num_teams{team_tracker.get_max_num_teams()};
  int common_index = get_ls_non_zero_bit(team_reduced_bitmask_, max_num_teams);
  if (common_index < 0) {
    /* No team available */
    LOG_ERROR_EXIT("Could not create team, out of resources: all bits in use.\n"
                   "  Please adjust ROCSHMEM_MAX_NUM_TEAMS\n");
  }

  /* Mark the team as taken (by unsetting the bit in the pool bitmask) */
  int byte = common_index / CHAR_BIT;
  team_pool_bitmask_[byte] &= ~(1 << (common_index % CHAR_BIT));

  /**
   * Allocate device-side memory for team_world and
   * construct a GDA team in it
   */
  GDATeam *new_team_obj;
  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(GDATeam)));
  new (new_team_obj)
      GDATeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes,
                my_pe_in_new_team, team_comm, common_index);

  *new_team = get_external_team(new_team_obj);
}

void GDABackend::ctx_create(int64_t options, void **ctx) {
  GDAHostContext *new_ctx{nullptr};
  new_ctx = new GDAHostContext(this, options);
  *ctx = new_ctx;
}

GDAHostContext *get_internal_gda_net_ctx(Context *ctx) {
  return reinterpret_cast<GDAHostContext *>(ctx);
}

void GDABackend::ctx_destroy(Context *ctx) {
  GDAHostContext *gda_host_ctx{get_internal_gda_net_ctx(ctx)};
  delete gda_host_ctx;
}

void GDABackend::reset_backend_stats() {
  assert(false);
}

void GDABackend::dump_backend_stats() {
  assert(false);
}

__host__ void GDABackend::global_exit(int status) {
  if (backend_comm != MPI_COMM_NULL)
    mpilib_ftable_.Abort(backend_comm, status);
  else
    abort();
}

void GDABackend::cleanup_teams() {
  free(team_pool_bitmask_);
  free(team_reduced_bitmask_);
}

void GDABackend::setup_wrk_sync_buffer() {
  /**
   * compute work/sync buffer size
   */
  auto max_num_teams{team_tracker.get_max_num_teams()};

  /**
   * size of barrier sync
   */
  wrk_sync_pool_size_ += sizeof(*barrier_sync) * ROCSHMEM_BARRIER_SYNC_SIZE;

  /**
   * Size of sync arrays for the teams
  */
  wrk_sync_pool_size_ += sizeof(long) * max_num_teams *
                           (ROCSHMEM_BARRIER_SYNC_SIZE +
                            ROCSHMEM_REDUCE_SYNC_SIZE +
                            ROCSHMEM_BCAST_SYNC_SIZE +
                            ROCSHMEM_ALLTOALL_SYNC_SIZE);

  /**
   * Size of work arrays for the teams
   * Accommodate largest possible data type for pWrk
  */
  wrk_sync_pool_size_ += sizeof(double) * max_num_teams *
                           ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;

  /**
   * Size of fence array
   */
  wrk_sync_pool_size_ += sizeof(int) * num_pes; //TODO: do we need a fence array?

  /**
   * Allocate a buffer of size wrk_sync_pool_size_, using heap memory
   * (should be uncached fine-grained ideally)
  */
  heap.malloc((void**)&wrk_sync_pool_, wrk_sync_pool_size_);
  assert(wrk_sync_pool_);
  wrk_sync_pool_top_ = wrk_sync_pool_;
}

void GDABackend::cleanup_wrk_sync_buffer() {
  heap.free(wrk_sync_pool_);
}

void GDABackend::setup_fence_buffer() { //TODO is this used?
  /*
   * Reserve memory for fence
   */
  fence_pool = reinterpret_cast<int *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(int) * num_pes;
}

void GDABackend::setup_collectives() {
  /*
   * Allocate heap space for barrier_sync
   */
  size_t one_sync_size_bytes {sizeof(*barrier_sync)};
  size_t sync_size_bytes {one_sync_size_bytes * ROCSHMEM_BARRIER_SYNC_SIZE};

  barrier_sync = reinterpret_cast<int64_t*>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sync_size_bytes;

  /*
   * Initialize the barrier synchronization array with default values.
   */
  for (size_t i = 0; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
    barrier_sync[i] = ROCSHMEM_SYNC_VALUE;
  }

  /*
   * Make sure that all processing elements have done this before
   * continuing.
   */
  rte_barrier();
}

void GDABackend::setup_teams() {
  /**
   * Allocate pools for the teams sync and work array from the SHEAP.
   */
  auto max_num_teams{team_tracker.get_max_num_teams()};

  barrier_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_BARRIER_SYNC_SIZE
                            * max_num_teams;

  reduce_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_REDUCE_SYNC_SIZE
                            * max_num_teams;

  bcast_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_BCAST_SYNC_SIZE
                            * max_num_teams;

  alltoall_pSync_pool = reinterpret_cast<long *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(long) * ROCSHMEM_ALLTOALL_SYNC_SIZE *
                        max_num_teams;

  /* Accommodating for largest possible data type for pWrk */
  pWrk_pool = reinterpret_cast<void *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(double) * ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE
                            * max_num_teams;

  /**
   * Initialize the sync arrays in the pool with default values.
   */
  long *barrier_pSync, *reduce_pSync, *bcast_pSync, *alltoall_pSync;
  for (int team_i = 0; team_i < max_num_teams; team_i++) {
    barrier_pSync = reinterpret_cast<long *>(
        &barrier_pSync_pool[team_i * ROCSHMEM_BARRIER_SYNC_SIZE]);
    reduce_pSync = reinterpret_cast<long *>(
        &reduce_pSync_pool[team_i * ROCSHMEM_REDUCE_SYNC_SIZE]);
    bcast_pSync = reinterpret_cast<long *>(
        &bcast_pSync_pool[team_i * ROCSHMEM_BCAST_SYNC_SIZE]);
    alltoall_pSync = reinterpret_cast<long *>(
        &alltoall_pSync_pool[team_i * ROCSHMEM_ALLTOALL_SYNC_SIZE]);

    for (size_t i = 0; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
      barrier_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_REDUCE_SYNC_SIZE; i++) {
      reduce_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_BCAST_SYNC_SIZE; i++) {
      bcast_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    for (size_t i = 0; i < ROCSHMEM_ALLTOALL_SYNC_SIZE; i++) {
      alltoall_pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
  }

  /**
   * Initialize bit mask
   *
   * Logical:
   * MSB..........................................................................LSB
   * Physical: MSB...1st least significant 8 bits...LSB  MSB...2nd least
   * significant 8 bits...LSB
   *
   * Description shows only a 2-byte long mask but idea extends to any
   * arbitrary size.
   */
  team_bitmask_size_ = (max_num_teams % CHAR_BIT) ? (max_num_teams / CHAR_BIT + 1)
                                             : (max_num_teams / CHAR_BIT);
  team_pool_bitmask_ = reinterpret_cast<char *>(malloc(team_bitmask_size_));
  team_reduced_bitmask_ = reinterpret_cast<char *>(malloc(team_bitmask_size_));

  memset(team_pool_bitmask_, 0, team_bitmask_size_);
  memset(team_reduced_bitmask_, 0, team_bitmask_size_);
  /* Set all to available except reserved teams (TEAM_WORLD and TEAM_SHARED) */
  for (int bit_i = TeamTracker::NUM_RESERVED_TEAMS; bit_i < max_num_teams; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    team_pool_bitmask_[byte_i] |= 1 << (bit_i % CHAR_BIT);
  }

  /**
   * Make sure that all processing elements have done this before
   * continuing.
   */
  rte_barrier();
}

void GDABackend::rte_barrier() {
  if (backend_comm != MPI_COMM_NULL) {
    NET_CHECK(mpilib_ftable_.Barrier(backend_comm));
  } else {
    backend_bootstr->barrier();
  }
}

GDAProvider GDABackend::requested_provider() {
  /* Check whether the user explicitly requests a particular provider type */
  std::string envstr = envvar::gda::provider;
  std::transform(envstr.begin(), envstr.end(), envstr.begin(), ::tolower);
  if (!envstr.empty()) {
    LOG_INFO("Found environment variable ROCSHMEM_GDA_PROVIDER, value is %s", envstr.c_str());
    if (envstr.find("bnxt") != std::string::npos) {
      return GDAProvider::BNXT;
    }
    if (envstr.find("ionic") != std::string::npos || envstr.find("pensando") != std::string::npos) {
      return GDAProvider::IONIC;
    }
    if (envstr.find("mlx5") != std::string::npos) {
      return GDAProvider::MLX5;
    }
  }
  return GDAProvider::UNSET;
}

/* Check if a device's vendor ID matches the expected vendor for a given provider.
 * Returns true if the device matches, false otherwise.
 */
bool GDABackend::device_matches_provider_vendor(GDAProvider provider,
                                                 const struct ibv_device_attr &device_attr,
                                                 [[maybe_unused]] const char *device_name) {
  uint32_t expected_vendor_id = 0;
  [[maybe_unused]] const char *vendor_name = nullptr;

  switch (provider) {
    case GDAProvider::BNXT:
      expected_vendor_id = GDA_BNXT_VENDOR_ID;
      vendor_name = "BNXT/Broadcom";
      break;
    case GDAProvider::IONIC:
      expected_vendor_id = GDA_IONIC_VENDOR_ID;
      vendor_name = "IONIC/Pensando";
      break;
    case GDAProvider::MLX5:
      expected_vendor_id = GDA_MLX5_VENDOR_ID;
      vendor_name = "MLX5/Mellanox";
      break;
    case GDAProvider::UNSET:
      // UNSET accepts any vendor
      return true;
    default:
      return true;
  }

  if (device_attr.vendor_id != expected_vendor_id) {
    LOG_TRACE("Skipping device %s with vendor_id=0x%04x (not %s)",
            device_name, device_attr.vendor_id, vendor_name);
    return false;
  }

  return true;
}

/* Check whether there are active InfiniBand/RDMA interfaces available.
 * Verifies the device vendor matches the requested provider to avoid selecting
 * the wrong NIC when multiple vendors are present.
 * Returns true if at least one active port is found on a matching device.
 */
bool GDABackend::has_active_ib_interface(GDAProvider provider) {
  struct ibv_device **device_list = nullptr;
  int num_devices = 0;
  bool has_active = false;

  device_list = ibv.get_device_list(&num_devices);
  if (!device_list || num_devices == 0) {
    LOG_WARN("No RDMA NIC devices found");
    return false;
  }

  for (int i = 0; i < num_devices && !has_active; i++) {
    LOG_TRACE("ibv.open device[%d] of %d", i, num_devices);
    struct ibv_context *context = ibv.open_device(device_list[i]);
    if (!context) {
      continue;
    }

    struct ibv_device_attr device_attr;
    if (ibv.query_device(context, &device_attr) == 0) {
      // Check if device vendor matches the provider
      if (!device_matches_provider_vendor(provider, device_attr,
                                          ibv.get_device_name(device_list[i]))) {
        ibv.close_device(context);
        continue;
      }

      for (int port = 1; port <= device_attr.phys_port_cnt; ++port) {
        struct ibv_port_attr port_attr;
        if (ibv.query_port(context, port, &port_attr) == 0) {
          if (port_attr.state == IBV_PORT_ACTIVE) {
            LOG_TRACE("Found at least one device with an active RDMA NIC (port=%d device=%s vendor_id=0x%04x, state=%d, phys_state=%d)",
                    port, ibv.get_device_name(device_list[i]),
                    device_attr.vendor_id, port_attr.state, port_attr.phys_state);
            has_active = true;
            break;
          }
        }
      }
    }

    ibv.close_device(context);
  }

  ibv.free_device_list(device_list);

  if (!has_active) {
    LOG_WARN("No active InfiniBand ports found on any device");
  }

  return has_active;
}

/* Check whether we can dlopen a Direct Verbs library and verify that
 * there are active InfiniBand/RDMA interfaces available to use.
 */
int GDABackend::backend_can_run() {
  void *handle{nullptr};
  GDAProvider requested = requested_provider();

  /* Basic verbs? */
  if (!ibv.is_initialized) return ROCSHMEM_ERROR;

  /* Try opening bnxt DV libraries */
#if defined(GDA_BNXT)
  if (requested == GDAProvider::UNSET || requested == GDAProvider::BNXT) {
    handle = bnxt_dv_dlopen();
    if (handle) {
      auto ret = has_active_ib_interface(GDAProvider::BNXT);
//      dlclose(handle); //TODO: unloading the lib crashes the next call to ibv_open_device
      if (ret) return ROCSHMEM_SUCCESS;
      LOG_TRACE("BNXT DV library found but no active InfiniBand interface available");
    }
  }
#endif //defined(GDA_BNXT)

  /* Try opening ionic DV libraries */
#if defined(GDA_IONIC)
  if (requested == GDAProvider::UNSET || requested == GDAProvider::IONIC) {
    handle = ionic_dv_dlopen();
    if (handle) {
      auto ret = has_active_ib_interface(GDAProvider::IONIC);
//      dlclose(handle); //TODO: unloading the lib crashes the next call to ibv_open_device
      if (ret) return ROCSHMEM_SUCCESS;
      LOG_TRACE("IONIC DV library found but no active InfiniBand interface available");
    }
  }
#endif //defined(GDA_IONIC)

  /* Try opening mlx5 DV libraries */
#if defined(GDA_MLX5)
  if (requested == GDAProvider::UNSET || requested == GDAProvider::MLX5) {
    handle = mlx5_dv_dlopen();
    if (handle) {
      auto ret = has_active_ib_interface(GDAProvider::MLX5);
//      dlclose(handle); //TODO: unloading the lib crashes the next call to ibv_open_device
      if (ret) return ROCSHMEM_SUCCESS;
      LOG_TRACE("MLX5 DV library found but no active InfiniBand interface available");
    }
  }
#endif //defined(GDA_MLX5)

  return ROCSHMEM_ERROR;
}

void GDABackend::setup_ibv() {
  open_dv_libs();

  open_ib_device();

  create_queues();

  exchange_qp_dest_info();

  modify_qps_reset_to_init();
  modify_qps_init_to_rtr();
  modify_qps_rtr_to_rts();

  rte_barrier();
}

void GDABackend::cleanup_ibv() {
  int err;

  if (gda_provider == GDAProvider::BNXT) {
    for (size_t i = 0; i < qps.size(); i++) {
      NicDevice &nic = nic_for_qp(i);

      err = bnxt_re_dv.destroy_qp(qps[i]);
      CHECK_ZERO(err, "bnxt_re_dv_destroy_qp");

      CHECK_HIP(hipHostUnregister(bnxt_qps[i].db_region_attr->dbr));

      err = bnxt_re_dv.free_db_region(nic.context, bnxt_qps[i].db_region_attr);
      CHECK_ZERO(err, "bnxt_re_dv_free_db_region");

      err = bnxt_re_dv.umem_dereg(bnxt_qps[i].attr.rq_umem_handle);
      CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (RQ)");

      err = bnxt_re_dv.umem_dereg(bnxt_qps[i].attr.sq_umem_handle);
      CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (SQ)");

      qp_allocator_->deallocate(bnxt_qps[i].sq_buf);
      qp_allocator_->deallocate(bnxt_qps[i].rq_buf);

      close(bnxt_qps[i].sq_dmabuf_fd);
      close(bnxt_qps[i].rq_dmabuf_fd);

      err = bnxt_re_dv.destroy_cq(bnxt_scqs[i].cq);
      CHECK_ZERO(err, "bnxt_re_dv_destroy_cq (SCQ)");

      err = bnxt_re_dv.destroy_cq(bnxt_rcqs[i].cq);
      CHECK_ZERO(err, "bnxt_re_dv_destroy_cq (RCQ)");

      err = bnxt_re_dv.umem_dereg(bnxt_scqs[i].umem_handle);
      CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (SCQ)");

      err = bnxt_re_dv.umem_dereg(bnxt_rcqs[i].umem_handle);
      CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (RCQ)");

      close(bnxt_scqs[i].dmabuf_fd);
      close(bnxt_rcqs[i].dmabuf_fd);

      qp_allocator_->deallocate(bnxt_scqs[i].buf);
      qp_allocator_->deallocate(bnxt_rcqs[i].buf);
    }
  } else if (gda_provider == GDAProvider::MLX5) {
    for (size_t i = 0; i < mlx5_qps.size(); i++) {
      // mlx5dv::destroy_qp also destroys the associated CQ
      err = mlx5dv.destroy_qp(mlx5_qps[i]);
      CHECK_ZERO(err, "mlx5dv::destroy_qp");
    }
  } else {
    for (size_t i = 0; i < qps.size(); i++) {
      err = ibv.destroy_qp(qps[i]);
      CHECK_ZERO(err, "ibv_destroy_qp");

      err = ibv.destroy_cq(cqs[i]);
      CHECK_ZERO(err, "ibv_destroy_cq");
    }
  }

  for (auto &nic : nic_devices_) {
    if (gda_provider == GDAProvider::IONIC) {
      if (nic.pd_uxdma[0]) {
        err = ibv.dealloc_pd(nic.pd_uxdma[0]);
        CHECK_ZERO(err, "ibv_dealloc_pd (uxdma[0])");
      }
      if (nic.pd_uxdma[1]) {
        err = ibv.dealloc_pd(nic.pd_uxdma[1]);
        CHECK_ZERO(err, "ibv_dealloc_pd (uxdma[1])");
      }
    }
    if (nic.pd_parent) {
      err = ibv.dealloc_pd(nic.pd_parent);
      CHECK_ZERO(err, "ibv_dealloc_pd (pd_parent)");
    }
    if (nic.pd_orig) {
      err = ibv.dealloc_pd(nic.pd_orig);
      CHECK_ZERO(err, "ibv_dealloc_pd (pd_orig)");
    }
    if (nic.context) {
      err = ibv.close_device(nic.context);
      CHECK_ZERO(err, "ibv_close_device");
    }
  }
}


void GDABackend::open_dv_libs() {
  int ret;
  GDAProvider requested = requested_provider();

  //this hardcoded init order will always prefer BNXT>IONIC>MLX5
  //if all three drivers are installed and enabled

#if defined(GDA_BNXT)
  if (gda_provider == GDAProvider::UNSET
  && (requested == GDAProvider::UNSET || requested == GDAProvider::BNXT)) {
    ret = bnxt_dv_dl_init();

    if (ret == ROCSHMEM_SUCCESS) {
      gda_provider = GDAProvider::BNXT;
    } else {
      LOG_TRACE("Initializing rocSHMEM BNXT GDA support failed");
    }
  }
#endif // defined(GDA_BNXT)

#if defined(GDA_IONIC)
  if (gda_provider == GDAProvider::UNSET
  && (requested == GDAProvider::UNSET || requested == GDAProvider::IONIC)) {
    ret = ionic_dv_dl_init();

    if (ret == ROCSHMEM_SUCCESS) {
      gda_provider = GDAProvider::IONIC;
    } else {
      LOG_TRACE("Initializing rocSHMEM IONIC GDA support failed");
    }
  }
#endif // defined(GDA_IONIC)

#if defined(GDA_MLX5)
  if (gda_provider == GDAProvider::UNSET
  && (requested == GDAProvider::UNSET || requested == GDAProvider::MLX5)) {
    ret = mlx5_dv_dl_init();

    if (ret == ROCSHMEM_SUCCESS) {
      gda_provider = GDAProvider::MLX5;
    } else {
      LOG_TRACE("Initializing rocSHMEM MLX5 GDA support failed");
    }
  }
#endif // defined(GDA_MLX5)

  if (gda_provider == GDAProvider::UNSET) {
    LOG_ERROR_EXIT("gda:open_dv_libs: no DV library could dlopen for IONIC, BNXT, or MLX5 GDA support");
  }
}

void GDABackend::close_dv_libs() {
  if (bnxtdv_handle_ != nullptr)
    dlclose(bnxtdv_handle_);

  if (ionicdv_handle_ != nullptr)
    dlclose(ionicdv_handle_);

  if (mlx5dv_handle_ != nullptr)
    dlclose(mlx5dv_handle_);

  gda_provider = GDAProvider::UNSET;
}

void GDABackend::exchange_qp_dest_info() {
  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    dest_info[i].lid = nic.portinfo.lid;
    if (gda_provider == GDAProvider::MLX5) {
      dest_info[i].qpn = mlx5_qps[i].qpn;
    } else {
      dest_info[i].qpn = qps[i]->qp_num;
    }
    dest_info[i].psn = 0;
    dest_info[i].gid = nic.gid;
  }

  for (size_t i = 0; i < num_qps_per_pe; i++) {
    if (backend_comm != MPI_COMM_NULL) {
      mpilib_ftable_.Alltoall(MPI_IN_PLACE, sizeof(dest_info_t), MPI_CHAR, dest_info.data() + i * num_pes, sizeof(dest_info_t), MPI_CHAR, backend_comm);
    } else {
      Alltoall_char_inplace(reinterpret_cast<char*>(dest_info.data() + i * num_pes), sizeof(dest_info_t), ROCSHMEM_TEAM_WORLD);
    }
  }
}

void GDABackend::setup_heap_memory_rkey() {
  auto *base_heap = heap.get_local_heap_base();
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

  if (envvar::gda::pcie_relaxed_ordering) {
    access |= IBV_ACCESS_RELAXED_ORDERING;
  }
  for (int n = 0; n < num_nics_; n++) {
    nic_devices_[n].heap_mr = ibv.reg_mr(nic_devices_[n].pd_orig, base_heap,
                                         heap.get_size(), access);
    CHECK_NNULL(nic_devices_[n].heap_mr, "ibv_reg_mr");
  }

  const size_t rkeys_per_pe = sizeof(uint32_t) * num_nics_;
  const size_t rkeys_size = rkeys_per_pe * num_pes;
  uint32_t *host_rkey_cpy = reinterpret_cast<uint32_t*>(malloc(rkeys_size));
  if (!host_rkey_cpy) { abort(); }

  CHECK_HIP(hipHostMalloc(&heap_rkey, rkeys_size));
  for (int n = 0; n < num_nics_; n++) {
    heap_rkey[my_pe * num_nics_ + n] = nic_devices_[n].heap_mr->rkey;
  }

  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemcpyAsync(host_rkey_cpy, heap_rkey, rkeys_size,
                           hipMemcpyDefault, stream));
  CHECK_HIP(hipStreamSynchronize(stream));

  if (backend_comm != MPI_COMM_NULL)
    mpilib_ftable_.Allgather(MPI_IN_PLACE, rkeys_per_pe, MPI_CHAR,
                             host_rkey_cpy, rkeys_per_pe, MPI_CHAR,
                             backend_comm);
  else
    backend_bootstr->allGather(host_rkey_cpy, rkeys_per_pe);

  CHECK_HIP(hipMemcpyAsync(heap_rkey, host_rkey_cpy, rkeys_size,
                           hipMemcpyDefault, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipStreamDestroy(stream));

  free(host_rkey_cpy);
}

void GDABackend::cleanup_heap_memory_rkey() {
  for (auto &nic : nic_devices_) {
    if (nic.heap_mr) {
      int ret = ibv.dereg_mr(nic.heap_mr);
      CHECK_ZERO(ret, "ibv_dereg_mr");
      nic.heap_mr = nullptr;
    }
  }

  CHECK_HIP(hipHostFree(heap_rkey));
}

void GDABackend::setup_gpu_qps() {
  size_t qp_objs_count;
  size_t qp_objs_mem_size;

  qp_objs_count    = num_qps;
  qp_objs_mem_size = sizeof(QueuePair) * qp_objs_count;

  CHECK_HIP(hipMalloc(&gpu_qps, qp_objs_mem_size));

  host_qps = (QueuePair*) malloc(qp_objs_mem_size);
  CHECK_NNULL(host_qps, "malloc (host_qps)");

  for (size_t i = 0; i < qp_objs_count; i++) {
    new (&host_qps[i]) QueuePair(nic_for_qp(i).pd_orig, gda_provider);
    CHECK_HIP(hipMemcpy(&gpu_qps[i], &host_qps[i], sizeof(QueuePair), hipMemcpyDefault));

    initialize_gpu_qp(&gpu_qps[i], i);
  }
}

void GDABackend::cleanup_gpu_qps() {
  size_t qp_objs_count;

  qp_objs_count = num_qps;

  for (size_t i = 0; i < qp_objs_count; i++) {
    host_qps[i].~QueuePair();
  }

  free(host_qps);

  CHECK_HIP(hipFree(gpu_qps));
  gpu_qps = nullptr;
}

void GDABackend::open_ib_device() {
  struct ibv_device **device_list = nullptr;
  int num_devices = 0;
  int err;

  device_list = ibv.get_device_list(&num_devices);
  CHECK_NNULL(device_list, "ibv_get_device_list");

  for (auto &nic : nic_devices_) {
    nic.device = nullptr;
    for (int i = 0; i < num_devices; i++) {
      const char *select_device = ibv.get_device_name(device_list[i]);
      CHECK_NNULL(select_device, "ibv_get_device_name");

      if (0 == strcmp(select_device, nic.nic_name.c_str())) {
        nic.device = device_list[i];
        break;
      }
    }

    if (nullptr == nic.device) {
      LOG_ERROR_EXIT(
        "Failed to select NIC '%s' when initializing GDA backend.\n"
        "  ROCSHMEM_HCA_LIST or ROCSHMEM_USE_IB_HCA may have excluded all available NICs.\n"
        "  Please adjust HCA_LIST or NIC configuration.",
        nic.nic_name.c_str());
    }

    if (gda_provider == GDAProvider::MLX5) {
      /* Explicitly request DevX context */
      struct mlx5dv_context_attr context_attr = {};
      context_attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
      nic.context = mlx5dv.open_device(nic.device, &context_attr);
    } else {
      nic.context = ibv.open_device(nic.device);
    }
    CHECK_NNULL(nic.context, "ib open device");
    dump_ibv_context(nic.context);
    dump_ibv_device(nic.context->device);

    validate_ib_device(nic);

    nic.pd_orig = ibv.alloc_pd(nic.context);
    CHECK_NNULL(nic.pd_orig, "ib allocate pd");
    dump_ibv_pd(nic.pd_orig);

    if (gda_provider == GDAProvider::IONIC) {
      create_parent_domain(nic);
    }

    err = ibv.query_port(nic.context, nic.port, &nic.portinfo);
    CHECK_ZERO(err, "ibv_query_port");
    dump_ibv_port_attr(&nic.portinfo);

    /* Must init after querying port */
    select_gid_index(nic);

    /* Zero out the device pointer to avoid usage after free_device_list */
    nic.device = nullptr;
  }

  ibv.free_device_list(device_list);
}

void GDABackend::validate_ib_device(NicDevice &nic) {
  char hostname[HOST_NAME_MAX + 1];
  const char *nicname;
  int err;

  err = gethostname(hostname, sizeof(hostname));
  CHECK_ZERO(err, "gethostname");

  nicname = ibv.get_device_name(nic.device);
  CHECK_NNULL(nicname, "ibv_get_device_name");

  std::string debug_str = "[" + std::string(hostname) + ", " + std::string(nicname) + "]";

  err = ibv.query_device(nic.context, &nic.device_attr);
  CHECK_ZERO(err, "ibv_query_device");

  if (gda_provider == GDAProvider::BNXT) {
    const std::set<uint32_t> supported_bnxt_part_ids = { 0x1760 /* BCM57608 */};
    const char min_supported_bnxt_fw_ver[12] = "233.2.104.0";

    if (nic.device_attr.vendor_id != GDA_BNXT_VENDOR_ID) {
      LOG_ERROR_EXIT("%s GDAProvider::BNXT requested but an invalid device is selected", debug_str.c_str());
    }

    if (supported_bnxt_part_ids.find(nic.device_attr.vendor_part_id) == supported_bnxt_part_ids.end()) {
      LOG_ERROR_EXIT("%s Unsupported Broadcom Part: %x", debug_str.c_str(), nic.device_attr.vendor_part_id);
    }

    if (strverscmp(min_supported_bnxt_fw_ver, nic.device_attr.fw_ver) > 0) {
      LOG_ERROR("%s Unsupported firmware version: %s", debug_str.c_str(), nic.device_attr.fw_ver);
      if (envvar::gda::override_nic_firmware_check == false) {
        exit(EXIT_FAILURE);
      }

      LOG_WARN("BNXT NIC Firmware check is disabled");
    }
  }

  for (int port = 1; port <= nic.device_attr.phys_port_cnt; ++port) {
    struct ibv_port_attr port_attr;
    if (ibv.query_port(nic.context, port, &port_attr) == 0) {
      if (port_attr.state == IBV_PORT_ACTIVE) {
        LOG_INFO("Using NIC %s: it has an active RDMA NIC port %d (vendor_id=0x%04x, state=%d, phys_state=%d)",
                  nicname, port, nic.device_attr.vendor_id, port_attr.state, port_attr.phys_state);
        return;
      }
    }
  }
  LOG_ERROR_EXIT("Could not validate that selected RDMA NIC %s has an active port", debug_str.c_str());
}

void GDABackend::modify_qps_reset_to_init() {
  int err;
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));

  attr.qp_state        = IBV_QPS_INIT;
  attr.pkey_index      = 0;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_LOCAL_WRITE
                       | IBV_ACCESS_REMOTE_READ
                       | IBV_ACCESS_REMOTE_ATOMIC;

  attr_mask = IBV_QP_STATE
            | IBV_QP_PKEY_INDEX
            | IBV_QP_PORT
            | IBV_QP_ACCESS_FLAGS;

  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    attr.port_num = nic.port;

    if (gda_provider == GDAProvider::BNXT) {
      err = bnxt_re_dv.modify_qp(qps[i], &attr, attr_mask, 0, 0);
    } else if (gda_provider == GDAProvider::MLX5) {
      err = mlx5dv.modify_qp(mlx5_qps[i], &attr, attr_mask, nic.gid_type);
    } else {
      err = ibv.modify_qp(qps[i], &attr, attr_mask);
    }
    CHECK_ZERO(err, "modify_qp (INIT)");
  }
}

void GDABackend::modify_qps_init_to_rtr() {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state               = IBV_QPS_RTR;
  attr.min_rnr_timer          = 12;

  if (gda_provider == GDAProvider::IONIC) {
    attr.max_dest_rd_atomic = 15;
  } else {
    attr.max_dest_rd_atomic = 1;
  }

  attr_mask = IBV_QP_STATE
            | IBV_QP_PATH_MTU
            | IBV_QP_RQ_PSN
            | IBV_QP_DEST_QPN
            | IBV_QP_AV
            | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER;

  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);

    attr.path_mtu         = nic.portinfo.active_mtu;
    attr.ah_attr.port_num = nic.port;

    if (nic.portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
      attr.ah_attr.grh.sgid_index = nic.gid_index;
      attr.ah_attr.is_global      = 1;
      attr.ah_attr.grh.hop_limit  = 255; // Max possible value
      attr.ah_attr.sl             = 1;
      attr.ah_attr.grh.traffic_class = envvar::gda::traffic_class;
    } else {
      attr.ah_attr.is_global         = 0;
      attr.ah_attr.grh.sgid_index    = 0;
      attr.ah_attr.grh.hop_limit     = 0;
      attr.ah_attr.sl                = 0;
      attr.ah_attr.grh.traffic_class = 0;
    }

    attr.rq_psn      = dest_info[i].psn;
    attr.dest_qp_num = dest_info[i].qpn;

    if (nic.portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
      memcpy(&attr.ah_attr.grh.dgid, &dest_info[i].gid, 16);
    } else {
      attr.ah_attr.dlid = dest_info[i].lid;
    }

    if (gda_provider == GDAProvider::BNXT) {
      err = bnxt_re_dv.modify_qp(qps[i], &attr, attr_mask, 0, 0);
    } else if (gda_provider == GDAProvider::MLX5) {
      err = mlx5dv.modify_qp(mlx5_qps[i], &attr, attr_mask, nic.gid_type);
    } else {
      err = ibv.modify_qp(qps[i], &attr, attr_mask);
    }
    CHECK_ZERO(err, "modify_qp (RTR)");
  }
}

void GDABackend::modify_qps_rtr_to_rts() {
  struct ibv_qp_attr attr;
  int attr_mask;
  int err;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state      = IBV_QPS_RTS;
  attr.timeout       = 14;
  attr.retry_cnt     = 7;
  attr.rnr_retry     = 7;

  if (gda_provider == GDAProvider::IONIC) {
    attr.max_rd_atomic = 15;
  } else {
    attr.max_rd_atomic = 1;
  }

  attr_mask = IBV_QP_STATE
            | IBV_QP_SQ_PSN
            | IBV_QP_MAX_QP_RD_ATOMIC
            | IBV_QP_TIMEOUT
            | IBV_QP_RETRY_CNT
            | IBV_QP_RNR_RETRY;

  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    attr.sq_psn = dest_info[i].psn;

    if (gda_provider == GDAProvider::BNXT) {
      err = bnxt_re_dv.modify_qp(qps[i], &attr, attr_mask, 0, 0);
    } else if (gda_provider == GDAProvider::MLX5) {
      err = mlx5dv.modify_qp(mlx5_qps[i], &attr, attr_mask, nic.gid_type);
    } else {
      err = ibv.modify_qp(qps[i], &attr, attr_mask);
    }
    CHECK_ZERO(err, "modify_qp (RTS)");
  }
}

void GDABackend::create_queues() {
  int ncqes;
  uint32_t sq_size = envvar::gda::sq_size;

  if (gda_provider == GDAProvider::IONIC) {
    ncqes = sq_size << 1;
  } else {
    ncqes = sq_size;
  }

  dest_info.resize(num_qps);
  cqs.resize(num_qps);
  qps.resize(num_qps);

  bnxt_scqs.resize(num_qps);
  bnxt_rcqs.resize(num_qps);
  bnxt_qps.resize(num_qps);

  mlx5_qps.resize(num_qps);

  if (gda_provider == GDAProvider::BNXT) {
    bnxt_create_cqs(ncqes);
    bnxt_create_qps(sq_size);
  } else if (gda_provider == GDAProvider::IONIC) {
    ionic_create_cqs(ncqes);
    create_qps(sq_size);
  } else if (gda_provider == GDAProvider::MLX5) {
    // mlx5_create_qps also creates the associated CQs
    mlx5_create_qps(sq_size);
  }

  alternate_qp_ports();
}

void GDABackend::alternate_qp_ports() {
  size_t cur_qp_idx;
  size_t new_qp_idx;

  /* We can't remap anything */
  if (envvar::max_num_contexts == 1) {
    return;
  }

  if (envvar::gda::alternate_qp_ports) {
    /* If we assume two PEs and a default context and two user context,
     * initially QPs are in the following port order:
     *
     * Labels :| DCTX PE0 | DCTX PE1 | CTX0 PE0 | CTX0 PE1 | CTX1 PE0 | CTX1 PE1 |
     * QPs    :| QP0      | QP1      | QP2      | QP3      | QP4      | QP5      |
     * Port   :| 0        | 1        | 0        | 1        | 0        | 1        |
     *
     * This creates the pattern where PE1 is always mapped to port 0 but we want it
     * to use both ports to maximize throughput/bandwidth.
     *
     * So we reorder our QPs
     *
     * Labels :| DCTX PE0 | DCTX PE1 | CTX0 PE0 | CTX0 PE1 | CTX1 PE0 | CTX1 PE1 |
     * QPs    :| QP0      | QP1      | QP2      | QP4      | QP3      | QP5      |
     * Port   :| 0        | 1        | 1        | 0        | 0        | 1        |
     *
     * We alternate the ports [0,1] and [1,0] for each context.
     * Therefore, when we use two contexts we use both ports
     *
     */

    /* Re-Map each context */
    for (size_t i = 1; i < num_qps_per_pe; i += 2) {
      for (size_t p = 0; p < static_cast<size_t>(num_pes); p += 2) {
        cur_qp_idx = (i * num_pes) + p;
        new_qp_idx = cur_qp_idx + 1;

        if (static_cast<size_t>(new_qp_idx) < qps.size()) {
          // Swap QPs
          std::swap(cqs[cur_qp_idx],       cqs[new_qp_idx]);
          std::swap(qps[cur_qp_idx],       qps[new_qp_idx]);
          std::swap(bnxt_scqs[cur_qp_idx], bnxt_scqs[new_qp_idx]);
          std::swap(bnxt_rcqs[cur_qp_idx], bnxt_rcqs[new_qp_idx]);
          std::swap(bnxt_qps[cur_qp_idx],  bnxt_qps[new_qp_idx]);
          std::swap(mlx5_qps[cur_qp_idx],  mlx5_qps[new_qp_idx]);
        }
      }
    }
  }
}

void* GDABackend::pd_alloc_device_uncached([[maybe_unused]] struct ibv_pd* pd, [[maybe_unused]] void* pd_context, size_t size, [[maybe_unused]] size_t alignment, [[maybe_unused]] uint64_t resource_type) {
  void* dev_ptr{nullptr};
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocUncached));
  CHECK_HIP(hipMemset(dev_ptr, 0, size));
  CHECK_HIP(hipStreamSynchronize(0));
  return dev_ptr;
}

void* GDABackend::pd_alloc_host([[maybe_unused]] struct ibv_pd* pd, [[maybe_unused]] void* pd_context, size_t size, [[maybe_unused]] size_t alignment, [[maybe_unused]] uint64_t resource_type) {
  void* dev_ptr{nullptr};
  CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&dev_ptr), size, hipHostMallocDefault));
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

void GDABackend::pd_release([[maybe_unused]] struct ibv_pd* pd, [[maybe_unused]] void* pd_context, void* ptr, [[maybe_unused]] uint64_t resource_type) {
  CHECK_HIP(hipFree(ptr));
}

void GDABackend::create_parent_domain(NicDevice &nic) {
  struct ibv_parent_domain_init_attr pattr;

  memset(&pattr, 0, sizeof(struct ibv_parent_domain_init_attr));
  pattr.pd         = nic.pd_orig;
  pattr.td         = nullptr,
  pattr.comp_mask  = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  pattr.free       = GDABackend::pd_release;
  pattr.pd_context = nullptr;

  if (gda_provider == GDAProvider::IONIC) {
    pattr.alloc      = GDABackend::pd_alloc_device_uncached;
  } else {
    pattr.alloc      = GDABackend::pd_alloc_host;
  }

  nic.pd_parent = ibv.alloc_parent_domain(nic.context, &pattr);
  CHECK_NNULL(nic.pd_parent, "ibv_alloc_parent_domain");
  dump_ibv_pd(nic.pd_parent);

  if (gda_provider == GDAProvider::IONIC) {
    ionic_setup_parent_domain(nic, &pattr);
  }
}

void GDABackend::create_cqs(int cqe) {
  struct ibv_cq_init_attr_ex cq_attr;
  struct ibv_cq_ex *cq_ex;

  assert(gda_provider != GDAProvider::BNXT);
  assert(gda_provider != GDAProvider::IONIC);

  memset(&cq_attr, 0, sizeof(struct ibv_cq_init_attr_ex));
  cq_attr.cqe           = cqe;
  cq_attr.cq_context    = nullptr;
  cq_attr.channel       = nullptr;
  cq_attr.comp_vector   = 0;
  cq_attr.flags         = 0;
  cq_attr.comp_mask     = IBV_CQ_INIT_ATTR_MASK_PD;
  /* enable mlx5 CQ collapsing by setting CQ length to 1 and enabling CQ overrun ignore:
   *  - mlx5 driver sets mlx5_ifc_cqc_bits::oi bit when IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN is set
   *    this has the hardware ignore CQ overruns; CQ consumer counter doorbells should not be rung
   *  - see Mellanox Adapters Programmer’s Reference Manual Rev 0.40, §7.12.8, Tables 75-76
   *    and linux/include/linux/mlx5/mlx5_ifc.h for Completion Queue Context definition
   *  - see also rdma-core/libibverbs/cmd_cq.c and linux/drivers/infiniband/hw/mlx5/cq.c
   *    for how this flag sets the bit */
  if (gda_provider == GDAProvider::MLX5) {
    cq_attr.cqe         = 1;
    cq_attr.comp_mask  |= IBV_CQ_INIT_ATTR_MASK_FLAGS;
    cq_attr.flags      |= IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
  }

  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    cq_attr.parent_domain = nic.pd_parent;
    cq_ex = ibv.create_cq_ex(nic.context, &cq_attr);
    CHECK_NNULL(cq_ex, "ibv_create_cq_ex");

    cqs[i] = ibv.cq_ex_to_cq(cq_ex);
    CHECK_NNULL(cqs[i], "ibv_cq_ex_to_cq");
  }
}

void GDABackend::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  switch (gda_provider) {
  case GDAProvider::IONIC:
    ionic_initialize_gpu_qp(gpu_qp, conn_num);
    dump_ibv_qp(qps[conn_num], conn_num);
    break;
  case GDAProvider::BNXT:
    bnxt_initialize_gpu_qp(gpu_qp, conn_num);
    dump_ibv_qp(qps[conn_num], conn_num);
    break;
  case GDAProvider::MLX5:
    mlx5_initialize_gpu_qp(gpu_qp, conn_num);
    break;
  default:
    assert(false /* GDAProvider initialize_gpu_qp */);
  }
}

void GDABackend::create_qps(int sq_length) {
  struct ibv_qp_init_attr_ex attr;

  memset(&attr, 0, sizeof(struct ibv_qp_init_attr_ex));
  attr.cap.max_send_wr     = sq_length;
  attr.cap.max_send_sge    = 1;
  attr.cap.max_inline_data = inline_threshold;
  attr.sq_sig_all          = 0;
  attr.qp_type             = IBV_QPT_RC;
  attr.comp_mask           = IBV_QP_INIT_ATTR_PD;

  if (gda_provider == GDAProvider::IONIC) {
    attr.cap.max_recv_sge    = 1; // TODO allow zero sges in the driver
  }

  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    if (gda_provider == GDAProvider::IONIC) {
      attr.pd = nic.pd_uxdma[i & 1];
    } else {
      attr.pd = nic.pd_parent;
    }
    attr.send_cq = cqs[i];
    attr.recv_cq = cqs[i];

    qps[i] = ibv.create_qp_ex(nic.context, &attr);
    CHECK_NNULL(qps[i], "ibv_create_qp_ex");
  }
}

void GDABackend::select_gid_index(NicDevice &nic) {
  struct ibv_gid_entry *gid_entries;
  union ibv_gid current_gid;
  union ibv_gid selected_gid;
  uint32_t current_gid_type;
  int err;

  const uint8_t local_gid_prefix[2] = {0xFE, 0x80};
  uint32_t selected_gid_type        = IBV_GID_TYPE_ROCE_V1;
  int selected_gid_index            = -1;
  ssize_t gid_tbl_entries           = 0;

  int gid_tbl_len         = nic.portinfo.gid_tbl_len;

  gid_entries = (struct ibv_gid_entry*) calloc(gid_tbl_len, sizeof(struct ibv_gid_entry));

  gid_tbl_entries = ibv.query_gid_table(nic.context, gid_entries, gid_tbl_len, 0);
  if (gid_tbl_entries < 0) {
    LOG_WARN("ibv_query_gid_table failed: GIDs not available");
    free(gid_entries);
    return;
  }

  for (int i = 0; i < gid_tbl_entries; i++) {
    current_gid_type = gid_entries[i].gid_type;
    current_gid      = gid_entries[i].gid;

    /* rocSHMEM does not use GIDs for IB mode */
    if (current_gid_type == IBV_GID_TYPE_IB) {
      selected_gid_index = i;
      selected_gid_type  = current_gid_type;
      selected_gid       = current_gid;
      break;
    }

    err = ibv.query_gid(nic.context, nic.port, i, &current_gid);
    CHECK_ZERO(err, "ibv_query_gid");

    /* We don't want local GIDs */
    if (memcmp(current_gid.raw, &local_gid_prefix, 2) == 0) {
      continue;
    }

    /* Initialize using first available GID */
    if (selected_gid_index == -1) {
      selected_gid_index = i;
      selected_gid_type  = current_gid_type;
      selected_gid       = current_gid;
    }
    /* Choose RoCEv2 over RoCEv1 */
    else if (current_gid_type > selected_gid_type) {
      selected_gid_index = i;
      selected_gid_type  = current_gid_type;
      selected_gid       = current_gid;
    }
  }

  nic.gid_index = selected_gid_index;
  nic.gid_type  = selected_gid_type;
  nic.gid       = selected_gid;

  free(gid_entries);
}

int GDABackend::ibv_mtu_to_int(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:  return 256;
    case IBV_MTU_512:  return 512;
    case IBV_MTU_1024: return 1024;
    case IBV_MTU_2048: return 2048;
    case IBV_MTU_4096: return 4096;
    default: {
      LOG_WARN("Invalid ibv_mtu %d", mtu);
      return 0;
    }
  }
}

}  // namespace rocshmem
