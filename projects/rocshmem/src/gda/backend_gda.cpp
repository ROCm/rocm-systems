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

#include <cstring>

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cassert>

#include "backend_gda.hpp"
#include "ibv_wrapper.hpp"
#include "envvar.hpp"
#include "gda_team.hpp"
#include "mpi_instance.hpp"
#include "util.hpp"
#include "topology.hpp"

namespace rocshmem {

#define NET_CHECK(cmd) {                                     \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort();                                               \
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

  select_nic();

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

  setup_ibv();
  setup_heap_memory_rkey();
  setup_gpu_qps();

  setup_ctxs();
  rte_barrier();
}

GDABackend::~GDABackend() {
  cleanup_ctxs();

  cleanup_teams();
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  CHECK_HIP(hipFree(team_world));

  cleanup_wrk_sync_buffer();

  cleanup_ipc();

  cleanup_gpu_qps();
  cleanup_heap_memory_rkey();
  cleanup_ibv();

  close_dv_libs();
}

void GDABackend::select_nic() {
  if (!envvar::requested_nic.is_default()) {
    requested_nic = envvar::requested_nic.get_value().c_str();
  } else {
    int gpu_dev = 0;
    CHECK_HIP(hipGetDevice(&gpu_dev));
    rocshmem::GetClosestNicToGpu(gpu_dev, envvar::hca_list.get_value().c_str(), &requested_nic);
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

void GDABackend::setup_ctxs() {
  setup_host_ctx();
  setup_default_ctx();

  CHECK_HIP(hipMalloc(&ctx_array, sizeof(GDAContext) * envvar::max_num_contexts));
  // 0th context is default context
  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    new (&ctx_array[i]) GDAContext(this, i + 1, gda_provider);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

void GDABackend::cleanup_ctxs() {
  ctx_free_list.~FreeListProxy();
  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    ctx_array[i].~GDAContext();
  }

  CHECK_HIP(hipFree(ctx_array));
}

__device__ bool GDABackend::create_ctx(int64_t options, rocshmem_ctx_t *ctx) {
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
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

  /**
   * Allocate device-side memory for team_world and construct a
   * GDA team in it.
   */
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));

  new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);
  new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);

  GDATeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(GDATeam)));
  new (team_world) GDATeam(this, team_info_wrt_parent, team_info_wrt_world,
                           num_pes, my_pe, backend_comm, 0);
  team_tracker.set_team_world(team_world);

  /**
   * Copy the address to ROCSHMEM_TEAM_WORLD.
   */
  ROCSHMEM_TEAM_WORLD = reinterpret_cast<rocshmem_team_t>(team_world);
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

  for (int i = 0; i < num_bytes; i++) {
    outbuf[i] = tmp_buffer[i];
    for (int j = 1; j < num_pes; j++) {
      outbuf[i] &= tmp_buffer[j * num_bytes + i];
    }
  }

  delete[] tmp_buffer;
}

void GDABackend::create_new_team([[maybe_unused]] Team *parent_team,
                                TeamInfo *team_info_wrt_parent,
                                TeamInfo *team_info_wrt_world, int num_pes,
                                int my_pe_in_new_team, MPI_Comm team_comm,
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
    printf("Could not create team, all bits in use. Aborting.\n");
    abort();
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
  for (int i = 0; i < ROCSHMEM_BARRIER_SYNC_SIZE; i++) {
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
   * Allocate pools for the teams sync and work arrary from the SHEAP.
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
   * signifant 8 bits...LSB
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
  /* Set all to available except the 0th one (reserved for TEAM_WORLD) */
  for (int bit_i = 1; bit_i < max_num_teams; bit_i++) {
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
  /* Check whether the user explicitely requests a particular provider type */
  std::string envstr = envvar::gda::provider;
  std::transform(envstr.begin(), envstr.end(), envstr.begin(), ::tolower);
  if (!envstr.empty()) {
    DPRINTF("Found environment variable ROCSHMEM_GDA_PROVIDER, value is %s\n", envstr.c_str());
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
                                                 const char *device_name) {
  uint32_t expected_vendor_id = 0;
  const char *vendor_name = nullptr;

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
    DPRINTF("Skipping device %s with vendor_id=0x%04x (not %s)\n",
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
    DPRINTF("No RDMA NIC devices found\n");
    return false;
  }

  for (int i = 0; i < num_devices && !has_active; i++) {
    DPRINTF("ibv.open device[%d] of %d\n", i, num_devices);
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
            DPRINTF("Found active RDMA NIC port %d on device %s (vendor_id=0x%04x, state=%d, phys_state=%d)\n",
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
    DPRINTF("No active InfiniBand ports found on any device\n");
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
      DPRINTF("BNXT DV library found but no active InfiniBand interface available\n");
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
      DPRINTF("IONIC DV library found but no active InfiniBand interface available\n");
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
      DPRINTF("MLX5 DV library found but no active InfiniBand interface available\n");
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
    for (int i = 0; i < qps.size(); i++) {
      err = bnxt_re_dv.destroy_qp(qps[i]);
      CHECK_ZERO(err, "bnxt_re_dv_destroy_qp");

      CHECK_HIP(hipHostUnregister(bnxt_qps[i].db_region_attr->dbr));

      err = bnxt_re_dv.free_db_region(context, bnxt_qps[i].db_region_attr);
      CHECK_ZERO(err, "bnxt_re_dv_free_db_region");

      err = bnxt_re_dv.umem_dereg(bnxt_qps[i].attr.rq_umem_handle);
      CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (RQ)");

      err = bnxt_re_dv.umem_dereg(bnxt_qps[i].attr.sq_umem_handle);
      CHECK_ZERO(err, "bnxt_re_dv_umem_dereg (SQ)");

      CHECK_HIP(hipFree(bnxt_qps[i].sq_buf));
      CHECK_HIP(hipFree(bnxt_qps[i].rq_buf));

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

      CHECK_HIP(hipFree(bnxt_scqs[i].buf));
      CHECK_HIP(hipFree(bnxt_rcqs[i].buf));
    }
  } else {
    for (int i = 0; i < qps.size(); i++) {
      err = ibv.destroy_qp(qps[i]);
      CHECK_ZERO(err, "ibv_destroy_qp");

      err = ibv.destroy_cq(cqs[i]);
      CHECK_ZERO(err, "ibv_destroy_cqs");
    }

    if (gda_provider == GDAProvider::IONIC) {
      err = ibv.dealloc_pd(pd_uxdma[0]);
      CHECK_ZERO(err, "ibv_dealloc_pd (uxdma[0])");

      err = ibv.dealloc_pd(pd_uxdma[1]);
      CHECK_ZERO(err, "ibv_dealloc_pd (uxdma[1])");
    }

    err = ibv.dealloc_pd(pd_parent);
    CHECK_ZERO(err, "ibv_dealloc_pd (pd_parent)");
  }

  err = ibv.dealloc_pd(pd_orig);
  CHECK_ZERO(err, "ibv_dealloc_pd (pd_orig)");

  err = ibv.close_device(context);
  CHECK_ZERO(err, "ibv_close_device");
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
      DPRINTF("Initializing rocSHMEM BNXT GDA support failed\n");
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
      DPRINTF("Initializing rocSHMEM IONIC GDA support failed\n");
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
      DPRINTF("Initializing rocSHMEM MLX5 GDA support failed\n");
    }
  }
#endif // defined(GDA_MLX5)

  if (gda_provider == GDAProvider::UNSET) {
    printf("rocshmem::gda:open_dv_libs: no DV library could dlopen for IONIC, BNXT, or MLX5 GDA support\n");
    exit(1);
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
  for (int i = 0; i < qps.size(); i++) {
    dest_info[i].lid = portinfo.lid;
    dest_info[i].qpn = qps[i]->qp_num;
    dest_info[i].psn = 0;
    dest_info[i].gid = gid;
  }

  for (size_t i = 0; i < envvar::max_num_contexts + 1; i++) {
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
  heap_mr = ibv.reg_mr(pd_orig, base_heap, heap.get_size(), access);
  CHECK_NNULL(heap_mr, "ibv_reg_mr");

  const size_t rkeys_size = sizeof(uint32_t) * num_pes;
  uint32_t *host_rkey_cpy = reinterpret_cast<uint32_t*>(malloc(rkeys_size));
  if (!host_rkey_cpy) { abort(); }

  CHECK_HIP(hipHostMalloc(&heap_rkey, sizeof(uint32_t) * num_pes));
  heap_rkey[my_pe] = heap_mr->rkey;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemcpyAsync(host_rkey_cpy, heap_rkey, rkeys_size, hipMemcpyDeviceToHost, stream));
  CHECK_HIP(hipStreamSynchronize(stream));

  if (backend_comm != MPI_COMM_NULL)
    mpilib_ftable_.Allgather(MPI_IN_PLACE, sizeof(uint32_t), MPI_CHAR, host_rkey_cpy, sizeof(uint32_t), MPI_CHAR, backend_comm);
  else
    backend_bootstr->allGather(host_rkey_cpy, sizeof(uint32_t));

  CHECK_HIP(hipMemcpyAsync(heap_rkey, host_rkey_cpy, rkeys_size, hipMemcpyHostToDevice, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipStreamDestroy(stream));

  free(host_rkey_cpy);
}

void GDABackend::cleanup_heap_memory_rkey() {
  int ret = ibv.dereg_mr(heap_mr);
  CHECK_ZERO(ret, "ibv_dereg_mr");

  CHECK_HIP(hipHostFree(heap_rkey));
}

void GDABackend::setup_gpu_qps() {
  size_t qp_objs_count;
  size_t qp_objs_mem_size;

  qp_objs_count    = (envvar::max_num_contexts + 1) * num_pes;
  qp_objs_mem_size = sizeof(QueuePair) * qp_objs_count;

  CHECK_HIP(hipMalloc(&gpu_qps, qp_objs_mem_size));

  host_qps = (QueuePair*) malloc(qp_objs_mem_size);
  CHECK_NNULL(host_qps, "malloc (host_qps)");

  for (size_t i = 0; i < qp_objs_count; i++) {
    new (&host_qps[i]) QueuePair(pd_orig, gda_provider);
    CHECK_HIP(hipMemcpy(&gpu_qps[i], &host_qps[i], sizeof(QueuePair), hipMemcpyDefault));

    initialize_gpu_qp(&gpu_qps[i], i);
  }
}

void GDABackend::cleanup_gpu_qps() {
  size_t qp_objs_count;

  qp_objs_count = (envvar::max_num_contexts + 1) * num_pes;

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

  if (requested_nic) {
    for (int i = 0; i < num_devices; i++) {
      const char *select_device = ibv.get_device_name(device_list[i]);
      CHECK_NNULL(select_device, "ibv_get_device_name");

      if (0 == strcmp(select_device, requested_nic)) {
        device = device_list[i];
        break;
      }
    }
  }

  if (nullptr == device) {
    fprintf(stderr,
      "rocshmem error: failed to select a NIC when initializing GDA backend.\n"
      "  ROCSHMEM_HCA_LIST or ROCSHMEM_USE_IB_HCA may have excluded all available NICs.\n"
      "  Please adjust HCA_LIST or NIC configuration.\n");
    exit(1);
  }

  context = ibv.open_device(device);
  CHECK_NNULL(context, "ib open device");
  dump_ibv_context(context);
  dump_ibv_device(context->device);

  validate_ib_device();

  pd_orig = ibv.alloc_pd(context);
  CHECK_NNULL(pd_orig, "ib allocate pd");
  dump_ibv_pd(pd_orig);

  if (gda_provider == GDAProvider::IONIC || gda_provider == GDAProvider::MLX5) {
    create_parent_domain();
  }

  err = ibv.query_port(context, port, &portinfo);
  CHECK_ZERO(err, "ibv_query_port");
  dump_ibv_port_attr(&portinfo);

  /* Must init after querying port */
  select_gid_index();

  ibv.free_device_list(device_list);
}

void GDABackend::validate_ib_device() {
  char hostname[HOST_NAME_MAX + 1];
  const char *nicname;
  int err;

  err = gethostname(hostname, sizeof(hostname));
  CHECK_ZERO(err, "gethostname");

  nicname = ibv.get_device_name(device);
  CHECK_NNULL(nicname, "ibv_get_device_name");

  debug_str = "[" + std::string(hostname) + ", " + std::string(nicname) + "]";

  err = ibv.query_device(context, &device_attr);
  CHECK_ZERO(err, "ibv_query_device");

  if (gda_provider == GDAProvider::BNXT) {
    const std::set<uint32_t> supported_bnxt_part_ids = { 0x1760 /* BCM57608 */};
    const char min_supported_bnxt_fw_ver[12] = "233.2.104.0";

    if (device_attr.vendor_id != GDA_BNXT_VENDOR_ID) {
      printf("%s GDAProvider::BNXT requested but an invalid device is selected\n", debug_str.c_str());
      exit(1);
    }

    if (supported_bnxt_part_ids.find(device_attr.vendor_part_id) == supported_bnxt_part_ids.end()) {
      printf("%s Unsupported Broadcom Part: %x\n", debug_str.c_str(), device_attr.vendor_part_id);
      exit(1);
    }

    if (strverscmp(min_supported_bnxt_fw_ver, device_attr.fw_ver) > 0) {
      printf("%s Unsupported firmware version: %s\n", debug_str.c_str(), device_attr.fw_ver);
      exit(1);
    }
  }
}

void GDABackend::modify_qps_reset_to_init() {
  int err;
  struct ibv_qp_attr attr;
  int attr_mask;

  memset(&attr, 0, sizeof(struct ibv_qp_attr));

  attr.qp_state        = IBV_QPS_INIT;
  attr.pkey_index      = 0;
  attr.port_num        = port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE
                       | IBV_ACCESS_LOCAL_WRITE
                       | IBV_ACCESS_REMOTE_READ
                       | IBV_ACCESS_REMOTE_ATOMIC;

  attr_mask = IBV_QP_STATE
            | IBV_QP_PKEY_INDEX
            | IBV_QP_PORT
            | IBV_QP_ACCESS_FLAGS;

  for (int i =0; i < qps.size() ; i++) {
    if (gda_provider == GDAProvider::BNXT) {
      err = bnxt_re_dv.modify_qp(qps[i], &attr, attr_mask, 0, 0);
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
  attr.path_mtu               = portinfo.active_mtu;
  attr.min_rnr_timer          = 12;
  attr.ah_attr.port_num       = port;

  if (gda_provider == GDAProvider::IONIC) {
    attr.max_dest_rd_atomic = 15;
  } else {
    attr.max_dest_rd_atomic = 1;
  }

  if (portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.is_global      = 1;
    attr.ah_attr.grh.hop_limit  = 1;
    attr.ah_attr.sl             = 1;
    attr.ah_attr.grh.traffic_class = envvar::gda::traffic_class;
  }

  attr_mask = IBV_QP_STATE
            | IBV_QP_PATH_MTU
            | IBV_QP_RQ_PSN
            | IBV_QP_DEST_QPN
            | IBV_QP_AV
            | IBV_QP_MAX_DEST_RD_ATOMIC
            | IBV_QP_MIN_RNR_TIMER;

  for (int i = 0; i < qps.size(); i++) {
    attr.rq_psn      = dest_info[i].psn;
    attr.dest_qp_num = dest_info[i].qpn;

    if (portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
      memcpy(&attr.ah_attr.grh.dgid, &dest_info[i].gid, 16);
    } else {
      attr.ah_attr.dlid = dest_info[i].lid;
    }

    if (gda_provider == GDAProvider::BNXT) {
      err = bnxt_re_dv.modify_qp(qps[i], &attr, attr_mask, 0, 0);
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

  for (int i = 0; i < qps.size(); i++) {
    attr.sq_psn = dest_info[i].psn;

    if (gda_provider == GDAProvider::BNXT) {
      err = bnxt_re_dv.modify_qp(qps[i], &attr, attr_mask, 0, 0);
    } else {
      err = ibv.modify_qp(qps[i], &attr, attr_mask);
    }
    CHECK_ZERO(err, "modify_qp (RTS)");
  }
}

void GDABackend::create_queues() {
  int ncqes;
  size_t resize_length;

  if (gda_provider == GDAProvider::IONIC) {
    ncqes = envvar::sq_size << 1;
  } else {
    ncqes = envvar::sq_size;
  }

  resize_length = (envvar::max_num_contexts + 1) * num_pes;

  dest_info.resize(resize_length);
  cqs.resize(resize_length);
  qps.resize(resize_length);

  bnxt_scqs.resize(resize_length);
  bnxt_rcqs.resize(resize_length);
  bnxt_qps.resize(resize_length);

  if (gda_provider == GDAProvider::BNXT) {
    bnxt_create_cqs(ncqes);
    bnxt_create_qps(envvar::sq_size);
  } else if (gda_provider == GDAProvider::IONIC) {
    ionic_create_cqs(ncqes);
    create_qps(envvar::sq_size);
  } else {
    create_cqs(ncqes);
    create_qps(envvar::sq_size);
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
    for (size_t i = 1; i < (envvar::max_num_contexts + 1); i += 2) {
      for (size_t p = 0; p < num_pes; p += 2) {
        cur_qp_idx = (i * num_pes) + p;
        new_qp_idx = cur_qp_idx + 1;

        if (new_qp_idx < qps.size()) {
          // Swap QPs
          std::swap(cqs[cur_qp_idx],       cqs[new_qp_idx]);
          std::swap(qps[cur_qp_idx],       qps[new_qp_idx]);
          std::swap(bnxt_scqs[cur_qp_idx], bnxt_scqs[new_qp_idx]);
          std::swap(bnxt_rcqs[cur_qp_idx], bnxt_rcqs[new_qp_idx]);
          std::swap(bnxt_qps[cur_qp_idx],  bnxt_qps[new_qp_idx]);
        }
      }
    }
  }
}

void* GDABackend::pd_alloc_device_uncached(struct ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type) {
  void* dev_ptr{nullptr};
  CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size, hipDeviceMallocUncached));
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

void* GDABackend::pd_alloc_host(struct ibv_pd* pd, void* pd_context, size_t size, size_t alignment, uint64_t resource_type) {
  void* dev_ptr{nullptr};
  CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&dev_ptr), size, hipHostMallocDefault));
  memset(dev_ptr, 0, size);
  return dev_ptr;
}

void GDABackend::pd_release(struct ibv_pd* pd, void* pd_context, void* ptr, uint64_t resource_type) {
  CHECK_HIP(hipFree(ptr));
}

void GDABackend::create_parent_domain() {
  struct ibv_parent_domain_init_attr pattr;

  memset(&pattr, 0, sizeof(struct ibv_parent_domain_init_attr));
  pattr.pd         = pd_orig;
  pattr.td         = nullptr,
  pattr.comp_mask  = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  pattr.free       = GDABackend::pd_release;
  pattr.pd_context = nullptr;

  if (gda_provider == GDAProvider::IONIC) {
    pattr.alloc      = GDABackend::pd_alloc_device_uncached;
  } else {
    pattr.alloc      = GDABackend::pd_alloc_host;
  }

  pd_parent = ibv.alloc_parent_domain(context, &pattr);
  CHECK_NNULL(pd_parent, "ibv_alloc_parent_domain");
  dump_ibv_pd(pd_parent);

  if (gda_provider == GDAProvider::IONIC) {
    ionic_setup_parent_domain(&pattr);
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
  cq_attr.parent_domain = pd_parent;

  for (int i = 0; i < qps.size(); i++) {
    cq_ex = ibv.create_cq_ex(context, &cq_attr);
    CHECK_NNULL(cq_ex, "ibv_create_cq_ex");

    cqs[i] = ibv.cq_ex_to_cq(cq_ex);
    CHECK_NNULL(cqs[i], "ibv_cq_ex_to_cq");
  }
}

void GDABackend::initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  switch (gda_provider) {
  case GDAProvider::IONIC:
    ionic_initialize_gpu_qp(gpu_qp, conn_num);
    break;
  case GDAProvider::BNXT:
    bnxt_initialize_gpu_qp(gpu_qp, conn_num);
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
  attr.pd                  = pd_parent;

  if (gda_provider == GDAProvider::IONIC) {
    attr.cap.max_recv_sge    = 1; // TODO allow zero sges in the driver
  }

  for (int i = 0; i < qps.size(); i++) {
    if (gda_provider == GDAProvider::IONIC) {
      attr.pd      = pd_uxdma[i & 1];
    }
    attr.send_cq = cqs[i];
    attr.recv_cq = cqs[i];

    qps[i] = ibv.create_qp_ex(context, &attr);
    CHECK_NNULL(qps[i], "ibv_create_qp_ex");
  }
}

void GDABackend::select_gid_index() {
  struct ibv_gid_entry *gid_entries;
  struct ibv_gid_entry *gid_entry;
  union ibv_gid current_gid;
  union ibv_gid selected_gid;
  uint32_t gid_type;
  int err;

  const uint8_t local_gid_prefix[2] = {0xFE, 0x80};
  uint32_t selected_gid_type        = IBV_GID_TYPE_ROCE_V1;
  int selected_gid_index            = -1;
  ssize_t gid_tbl_entries           = 0;

  int gid_tbl_len         = portinfo.gid_tbl_len;

  gid_entries = (struct ibv_gid_entry*) calloc(gid_tbl_len, sizeof(struct ibv_gid_entry));

  gid_tbl_entries = ibv.query_gid_table(context, gid_entries, gid_tbl_len, 0);
  if (gid_tbl_entries < 0) {
    fprintf(stderr, "[Warning] ibv_query_gid_table failed. No available GIDs\n");
    free(gid_entries);
    return;
  }

  for (int i = 0; i < gid_tbl_entries; i++) {
    gid_type = gid_entries[i].gid_type;

    /* rocSHMEM does not use GIDs for IB mode */
    if (gid_type == IBV_GID_TYPE_IB) {
      break;
    }

    current_gid = gid_entries[i].gid;

    err = ibv.query_gid(context, port, i, &current_gid);
    CHECK_ZERO(err, "ibv_query_gid");

    /* We don't want local GIDs */
    if (memcmp(current_gid.raw, &local_gid_prefix, 2) == 0) {
      continue;
    }

    /* Initialize using first available GID */
    if (selected_gid_index == -1) {
      selected_gid_index = i;
      selected_gid_type  = gid_type;
      selected_gid       = current_gid;
    }
    /* Choose RoCEv2 over RoCEv1 */
    else  if (gid_type > selected_gid_type) {
      selected_gid_index = i;
      selected_gid_type  = gid_type;
      selected_gid       = current_gid;
    }
  }

  gid_index = selected_gid_index;
  gid       = selected_gid;

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
      fprintf(stderr, "[ERROR] Invalid ibv_mtu\n");
      return 0;
    }
  }
}

}  // namespace rocshmem
