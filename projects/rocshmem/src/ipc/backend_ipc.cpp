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

#include "backend_ipc.hpp"
#include "envvar.hpp"
#include "ipc_team.hpp"
#include "mpi_instance.hpp"

namespace rocshmem {

#define NET_CHECK(cmd)                                       \
  {                                                          \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort() ;                                              \
    }                                                        \
  }

extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

rocshmem_team_t get_external_team(IPCTeam *team) {
  return reinterpret_cast<rocshmem_team_t>(team);
}

static int get_ls_non_zero_bit(char *bitmask, int mask_length) {
  int position = -1;

  for (int bit_i = 0; bit_i < mask_length; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    if (bitmask[byte_i] & (1 << (bit_i % CHAR_BIT))) {
      position = bit_i;
      break;
    }
  }

  return position;
}

IPCBackend::IPCBackend(MPI_Comm comm):  Backend(comm) {
  type = BackendType::IPC_BACKEND;

  initIPC();

  /**
   * Check if num_pes == ipcImpl.shm_size)
   * All the PEs must be with in a node for IPC conduit
   */
  if(num_pes != ipcImpl.shm_size) {
    fprintf(stderr, "rocSHMEM: IPC Backend selected but some PEs are non-local. This is not a supported configuration.\n"
                    "  The GDA and RO backends mix off-node -and- IPC on-node communication as needed.\n");
    exit(1);
  }

  /* Initialize the host interface */
  host_interface = std::make_shared<HostInterface>(hdp_proxy_.get(),
                                                   backend_comm,
                                                   &heap);

  default_host_ctx = std::make_unique<IPCHostContext>(this, 0);

  init();
}

IPCBackend::IPCBackend(TcpBootstrap *bootstrap):  Backend(bootstrap) {
  type = BackendType::IPC_BACKEND;

  initIPC(bootstrap); // no MPI involved

  /**
   * Check if num_pes == ipcImpl.shm_size)
   * All the PEs must be with in a node for IPC conduit
   */
  assert(num_pes == ipcImpl.shm_size);

  /* Initialize the host interface */
  host_interface = std::make_shared<HostInterface>(hdp_proxy_.get(),
                                                   bootstrap,
                                                   &heap);

  default_host_ctx = std::make_unique<IPCHostContext>(this, 0);

  init();
}

void IPCBackend::init() {
  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();

  const char *arch_name = get_arch_name(hip_dev_id);
  if (strncmp(arch_name, "gfx1201", strlen("gfx1201")) == 0) {
    fine_grained_allocator_ = new HIPAllocatorFinegrained();
  } else {
    fine_grained_allocator_ = new HIPDefaultFinegrainedAllocator();
  }

  setup_team_world();

  setup_wrk_sync_buffers();

  rocshmem_collective_init();

  setup_fence_buffer();

  teams_init();

  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;

  default_context_proxy_ = IPCDefaultContextProxyT(this, tinfo);

  setup_ctxs();
}

IPCBackend::~IPCBackend() {
  /**
   * Destroy teams infrastructure
   * and team world
   */
  teams_destroy();
  cleanup_wrk_sync_buffer();
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  CHECK_HIP(hipFree(team_world));

  CHECK_HIP(hipFree(ctx_array));
  if (fine_grained_allocator_) {
    const char *arch_name = get_arch_name(hip_dev_id);
    if (strncmp(arch_name, "gfx1201", strlen("gfx1201")) == 0) {
      delete static_cast<HIPAllocatorFinegrained *>(fine_grained_allocator_);
    } else {
      delete static_cast<HIPDefaultFinegrainedAllocator *>(fine_grained_allocator_);
    }
  }
}

int IPCBackend::backend_can_run(MPI_Comm comm, TcpBootstrap* bootstrap) {
  int ret = ROCSHMEM_ERROR;

  if (comm != MPI_COMM_NULL) {
    int comm_size;
    mpilib_ftable_.Comm_size(comm, &comm_size);
    MPI_Comm shmcomm;
    mpilib_ftable_.Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                  &shmcomm);
    int shm_comm_size;
    mpilib_ftable_.Comm_size(shmcomm, &shm_comm_size);
    mpilib_ftable_.Comm_free(&shmcomm);
    if (shm_comm_size == comm_size) {
      ret = ROCSHMEM_SUCCESS;
    }
  } else if (bootstrap != nullptr) {
      int world_size = bootstrap->getNranks();
      int shm_size = bootstrap->getNranksPerNode();
      if (shm_size == world_size) {
        ret = ROCSHMEM_SUCCESS;
      }
  }

  return ret;
}
void IPCBackend::setup_ctxs() {
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(IPCContext) * envvar::max_num_contexts));
  // 0th context is default context
  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    new (&ctx_array[i]) IPCContext(this, i + 1);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

__device__ bool IPCBackend::create_ctx(int64_t options, rocshmem_ctx_t *ctx) {
  IPCContext *ctx_{nullptr};

  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;

  ctx_->tinfo = reinterpret_cast<TeamInfo *>(ctx->team_opaque);
  return true;
}

__device__ void IPCBackend::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<IPCContext *>(ctx->ctx_opaque));
}

void IPCBackend::setup_team_world() {
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

  /**
   * Allocate device-side memory for team_world and construct a
   * IPC team in it.
   */
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));

  new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);
  new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);

  IPCTeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(IPCTeam)));
  new (team_world) IPCTeam(this, team_info_wrt_parent, team_info_wrt_world,
                             num_pes, my_pe, backend_comm, 0);
  team_tracker.set_team_world(team_world);

  /**
   * Copy the address to ROCSHMEM_TEAM_WORLD.
   */
  ROCSHMEM_TEAM_WORLD = reinterpret_cast<rocshmem_team_t>(team_world);
}

void IPCBackend::team_destroy(rocshmem_team_t team) {
  IPCTeam *team_obj = get_internal_ipc_team(team);

  /* Mark the pool as available */
  int bit = team_obj->pool_index_;
  int byte_i = bit / CHAR_BIT;
  team_pool_bitmask_[byte_i] |= 1 << (bit % CHAR_BIT);

  team_obj->~IPCTeam();
  CHECK_HIP(hipFree(team_obj));
}

void IPCBackend::Allreduce_char_BAND (char* inbuf, char *outbuf, size_t num_bytes,
                                      Team *team) {

  // Implement an Allreduce outside of MPI. This is specialized for the scenario
  // required for the team creation, i.e. assuming bytes and using BAND operation.
  // Implementation uses an Allgather operation followed a local reduction.

  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  int num_pes = team_obj->num_pes;
  int my_pe = team_obj->my_pe;

  char *tmp_buffer = new char[num_pes * num_bytes];
  std::memset(tmp_buffer, 0, num_pes * num_bytes);
  std::memcpy (&tmp_buffer[my_pe * num_bytes], inbuf, num_bytes);

  if (num_pes == backend_bootstr->getNranks() ) {
    backend_bootstr->allGather(tmp_buffer, num_bytes);
  } else {
    printf("IPCBackend::create_new_team: non-mpi version only supports parent_teams that contain all processes. Aborting.\n");
    abort();
  }

  for (int i = 0; i < num_bytes; i++) {
    outbuf[i] = tmp_buffer[i];
    for (int j = 1; j < num_pes; j++) {
      outbuf[i] &= tmp_buffer[j * num_bytes + i];
    }
  }

  delete[] tmp_buffer;
}

void IPCBackend::create_new_team([[maybe_unused]] Team *parent_team,
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
   * construct a IPC team in it
   */
  IPCTeam *new_team_obj;
  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(IPCTeam)));
  new (new_team_obj)
      IPCTeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes,
                my_pe_in_new_team, team_comm, common_index);

  *new_team = get_external_team(new_team_obj);
}

void IPCBackend::ctx_create(int64_t options, void **ctx) {
  IPCHostContext *new_ctx{nullptr};
  new_ctx = new IPCHostContext(this, options);
  *ctx = new_ctx;
}

IPCHostContext *get_internal_ipc_net_ctx(Context *ctx) {
  return reinterpret_cast<IPCHostContext *>(ctx);
}

void IPCBackend::ctx_destroy(Context *ctx) {
  IPCHostContext *ro_net_host_ctx{get_internal_ipc_net_ctx(ctx)};
  delete ro_net_host_ctx;
}

void IPCBackend::reset_backend_stats() {
  assert(false);
}

void IPCBackend::dump_backend_stats() {
  assert(false);
}

void IPCBackend::initIPC() {
  const auto &heap_bases{heap.get_heap_bases()};

  ipcImpl.ipcHostInit(my_pe, heap_bases,
                      backend_comm);
}

void IPCBackend::initIPC(TcpBootstrap *bootstr) {
  const auto &heap_bases{heap.get_heap_bases()};

  ipcImpl.ipcHostInit(my_pe, heap_bases,
                      bootstr);
}

void IPCBackend::global_exit(int status) {
  if (backend_comm != MPI_COMM_NULL)
    mpilib_ftable_.Abort(backend_comm, status);
  else
    abort();
}

void IPCBackend::teams_destroy() {
  free(team_pool_bitmask_);
  free(team_reduced_bitmask_);
}

void IPCBackend::setup_wrk_sync_buffers() {
  /**
   * calcualte work/sync buffer size
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
  wrk_sync_pool_size_ += sizeof(int) * num_pes;

  /**
   * Allocate a buffer of size wrk_sync_pool_size_, using fine-grained
   * memory allocator
  */
  fine_grained_allocator_->allocate((void**)&wrk_sync_pool_,
                                    wrk_sync_pool_size_);
  assert(wrk_sync_pool_);
  wrk_sync_pool_top_ = wrk_sync_pool_;

  /*
   * Allocate a c-array to hold the IPC handles
   */
  hipIpcMemHandle_t *ipc_handle = reinterpret_cast<hipIpcMemHandle_t*>(
            malloc(num_pes * sizeof(hipIpcMemHandle_t)));

  /*
   * Call into the hip runtime to get an IPC handle for the allocated
   * wrk_sync_pool_ buffer and store that IPC handle
   */
  CHECK_HIP(hipIpcGetMemHandle(&ipc_handle[my_pe], wrk_sync_pool_));

  /*
   * all-to-all exchange with each PE to share the IPC handles.
   */
  if (backend_comm != MPI_COMM_NULL) {
    mpilib_ftable_.Allgather(MPI_IN_PLACE, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                             ipc_handle, sizeof(hipIpcMemHandle_t), MPI_CHAR, backend_comm);
  } else {
    assert (backend_bootstr != nullptr);
    backend_bootstr->allGather(ipc_handle, sizeof(hipIpcMemHandle_t));
  }

  /*
   * Allocate device-side fine grained memory to hold IPC addresses of
   * work/sync buffers
   */
  fine_grained_allocator_->allocate(
    reinterpret_cast<void**>(&wrk_sync_pool_bases_),
    num_pes * sizeof(char*));
  assert(wrk_sync_pool_bases_);

  /*
   * For all local processing elements, initialize the device-side array
   * with the IPC work/sync buffer addresses.
   */
  for (int i = 0; i < num_pes; i++) {
    if (i != my_pe) {
      CHECK_HIP(hipIpcOpenMemHandle(
          reinterpret_cast<void**>(&wrk_sync_pool_bases_[i]),
          ipc_handle[i],
          hipIpcMemLazyEnablePeerAccess));
    } else {
      wrk_sync_pool_bases_[i] = wrk_sync_pool_;
    }
  }
}

void IPCBackend::cleanup_wrk_sync_buffer() {
  for (int i = 0; i < num_pes; i++) {
    if (i != my_pe) {
      CHECK_HIP(hipIpcCloseMemHandle(wrk_sync_pool_bases_[i]));
    }
  }
  fine_grained_allocator_->deallocate(wrk_sync_pool_bases_);
  fine_grained_allocator_->deallocate(wrk_sync_pool_);
}

void IPCBackend::setup_fence_buffer() {
  /*
  * Allocate memory for fence
  */
  fence_pool = reinterpret_cast<int *>(wrk_sync_pool_top_);
  wrk_sync_pool_top_ += sizeof(int) * num_pes;
}

void IPCBackend::rocshmem_collective_init() {
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
  if (backend_comm != MPI_COMM_NULL) {
    NET_CHECK(mpilib_ftable_.Barrier(backend_comm));
  } else {
    backend_bootstr->barrier();
  }
}

void IPCBackend::teams_init() {
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
  if (backend_comm != MPI_COMM_NULL) {
    NET_CHECK(mpilib_ftable_.Barrier(backend_comm));
  } else {
    backend_bootstr->barrier();
  }
}

}  // namespace rocshmem
