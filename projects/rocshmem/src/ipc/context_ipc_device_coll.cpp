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

#include "rocshmem/rocshmem.hpp"
#include "constmem.hpp"
#include "context_incl.hpp"
#include "context_ipc_tmpl_device.hpp"
#include "util.hpp"
#include "ipc_team.hpp"

namespace rocshmem {

__device__ void IPCContext::internal_direct_barrier(int pe, int PE_start,
                                                    int stride, int n_pes,
                                                    int64_t *pSync) {
  int64_t flag_val = 1;
  if (pe == PE_start) {
    // Go through all PE offsets (except current offset = 0)
    // and wait until they all reach
#if defined(__gfx90a__)
    __threadfence_system();
#endif /* __gfx90a__ */
    for (int i = 1; i < n_pes; i++) {
      wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
      pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    threadfence_system();

    // Announce to other PEs that all have reached
    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem(&pSync[0], &flag_val, sizeof(*pSync), j);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
    }
  } else {
    // Mark current PE offset as reached
    size_t pe_offset = (pe - PE_start) / stride;
    internal_putmem(&pSync[pe_offset], &flag_val, sizeof(*pSync), PE_start);
#if defined(__gfx90a__)
    __threadfence_system();
#endif /* __gfx90a__ */
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    threadfence_system();
  }
}

__device__ void IPCContext::internal_atomic_barrier(int pe, int PE_start,
                                                    int stride, int n_pes,
                                                    int64_t *pSync) {
  int64_t flag_val = 1;
  if (pe == PE_start) {
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, (int64_t)(n_pes - 1));
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    threadfence_system();

    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem(&pSync[0], &flag_val, sizeof(*pSync), j);
    }
  } else {
    amo_add<int64_t>(&pSync[0], flag_val, PE_start);
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    threadfence_system();
  }
}

__device__ void IPCContext::internal_sync(int pe, int PE_start, int stride,
                                          int PE_size, int64_t *pSync) {
  if (PE_size < 64) {
    internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
  } else {
    internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
  }
}

__device__ void IPCContext::internal_sync_wave(int pe, int PE_start, int stride,
                                               int PE_size, int64_t *pSync) {
  if (is_thread_zero_in_wave()) {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
    }
  }
}

// Uses PE values that are relative to world
__device__ void IPCContext::internal_sync_wg(int pe, int PE_start, int stride,
                                             int PE_size, int64_t *pSync) {
  __syncthreads();
  if (is_thread_zero_in_block()) {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
    }
  }
  __syncthreads();
}

__device__ void IPCContext::sync(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::sync_wave(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::sync_wg(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::sync_all() {
  internal_sync(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync);
}

__device__ void IPCContext::sync_all_wave() {
  internal_sync_wave(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync);
}

__device__ void IPCContext::sync_all_wg() {
  internal_sync_wg(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync);
}

__device__ void IPCContext::barrier_all() {
  quiet();
  sync_all();
}

__device__ void IPCContext::barrier_all_wave() {
  if (is_thread_zero_in_wave()) {
    quiet();
  }
  sync_all_wave();
}

__device__ void IPCContext::barrier_all_wg() {
  if (is_thread_zero_in_block()) {
    quiet();
  }
  sync_all_wg();
  __syncthreads();
}

__device__ void IPCContext::barrier(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  quiet();
  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::barrier_wave(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  if (is_thread_zero_in_wave()) {
    quiet();
  }
  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void IPCContext::barrier_wg(rocshmem_team_t team) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  if (is_thread_zero_in_block()) {
    quiet();
  }
  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync);
  __syncthreads();
}

__device__ void IPCContext::alltoallmem_wg(rocshmem_team_t team, void *dst,
                                     const void *src, int nelems) {
  internal_alltoallmem_wg(team, dst, src, nelems);
}

__device__ void IPCContext::internal_alltoallmem_wg(rocshmem_team_t team, void *dst,
                                                    const void *src, int nelems) {
#if defined(USE_SDMA)
  if (nelems < 512 || ipcImpl_.sdmaImpl_.sdmaEnabled)
#else
  if (nelems < 512)
#endif
    alltoallmem_wg_linear_thread_puts(team, dst, src, nelems);
  else
    alltoallmem_wg_linear(team, dst, src, nelems);
}

__device__ void IPCContext::alltoallmem_wg_linear(rocshmem_team_t team, void *dst,
                                            const void *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    putmem_nbi_wg(reinterpret_cast<char *>(dst) + my_pe_in_team * nelems, 
                  reinterpret_cast<const char *>(src) + j * nelems, nelems, dest_pe);
  }
  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync_wg(my_pe, pe_start, stride, pe_size, pSync);
}

__device__ void IPCContext::alltoallmem_wg_linear_thread_puts(rocshmem_team_t team,
    void *dst, const void *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  size_t alltoall_pSync_offset = (team_obj->alltoall_sequence_number % 2) * pe_size;

  int tid = get_flat_block_id();
  // min(get_flat_block_size(), WF_SIZE)
  int step_size = get_flat_block_size() < WF_SIZE ? get_flat_block_size() : WF_SIZE;

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    putmem_nbi(reinterpret_cast<char *>(dst) + my_pe_in_team * nelems,
               reinterpret_cast<const char *>(src) + j * nelems, nelems, dest_pe);
  }
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    fence(dest_pe);
    ptrdiff_t L_offset = reinterpret_cast<char*>(&pSync[alltoall_pSync_offset + my_pe_in_team]) - wrk_sync_pool_bases_[my_pe];
    ipcImpl_.ipcAMOAdd(reinterpret_cast<long*>(wrk_sync_pool_bases_[dest_pe] + L_offset), 1L);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    long *sync_flag = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flag) != 1) { }

    //quiet(dest_pe);// needed to quiet add when it is nbi in gda, it is not nbi in ipc

    pSync[alltoall_pSync_offset + dest_pe] = ROCSHMEM_SYNC_VALUE;
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

__device__ void IPCContext::internal_put_broadcastmem_wave(
    void *dst, const void *src, int nelems, int pe_root, int pe_start,
    int stride, int pe_size) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
        putmem_nbi_wave(dst, src, nelems, i);
    }
  }
}

__device__ void IPCContext::internal_get_broadcastmem_wave(
  void *dst, const void *src, int nelems, int pe_root) {
    getmem_wave(dst, src, nelems, pe_root);
}

__device__ void IPCContext::internal_broadcastmem_wave(void *dst, const void *src, int nelems,
                                      int pe_root, int pe_start,
                                      int stride, int pe_size,
                                      long *p_sync) {  // NOLINT(runtime/int)
  if (num_pes < 4) {
    internal_put_broadcastmem_wave(dst, src, nelems, pe_root, pe_start, stride, pe_size);
  } else {
    internal_get_broadcastmem_wave(dst, src, nelems, pe_root);
  }

  // Synchronize on completion of broadcast
  internal_sync_wave(my_pe, pe_start, stride, pe_size, p_sync);
}

__device__ int IPCContext::broadcastmem_wave(rocshmem_team_t team,
                              void *dest, const void *source, int nelement, int PE_root) {

  if (dest == nullptr || source == nullptr || team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(PE_root);

  internal_broadcastmem_wave(dest, source, nelement, pe_root_world, 
                              pe_start, stride, pe_size, p_sync);
  return ROCSHMEM_SUCCESS;
}

__device__ void IPCContext::internal_put_broadcastmem_wg(
    void *dst, const void *src, int nelems, int pe_root, int pe_start,
    int stride, int pe_size) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
        putmem_nbi_wg(dst, src, nelems, i);
    }
  }
}

__device__ void IPCContext::internal_get_broadcastmem_wg(
  void *dst, const void *src, int nelems, int pe_root) {
    getmem_wg(dst, src, nelems, pe_root);
}

__device__ void IPCContext::internal_broadcastmem_wg(void *dst, const void *src, int nelems,
                                      int pe_root, int pe_start,
                                      int stride, int pe_size,
                                      long *p_sync) {  // NOLINT(runtime/int)
  if (num_pes < 4) {
    internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride, pe_size);
  } else {
    internal_get_broadcastmem_wg(dst, src, nelems, pe_root);
  }

  // Synchronize on completion of broadcast
  internal_sync_wg(my_pe, pe_start, stride, pe_size, p_sync);
}

__device__ void IPCContext::broadcastmem_wg(rocshmem_team_t team,
                              void *dest, const void *source, int nelement, int PE_root) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(PE_root);

  internal_broadcastmem_wg(dest, source, nelement, pe_root_world, 
                              pe_start, stride, pe_size, p_sync);
}
__device__ int IPCContext::alltoallmem_wave(rocshmem_team_t team, void* dest, 
                                  const void* source, int nelems){
  if (dest == nullptr || source == nullptr || team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  internal_alltoallmem_wave(team, dest, source, nelems);

  return ROCSHMEM_SUCCESS;
}
__device__ void IPCContext::internal_alltoallmem_wave(rocshmem_team_t team, void* dest, 
                                  const void* source, int nelems) {
#if defined(USE_SDMA)
  if (nelems < 512 || ipcImpl_.sdmaImpl_.sdmaEnabled)
#else
  if (nelems < 512)
  #endif
    alltoallmem_linear_thread_puts_wave(team, dest, source, nelems);
  else
    alltoallmem_linear_wave(team, dest, source, nelems);
}


__device__ void IPCContext::alltoallmem_linear_wave(rocshmem_team_t team, void *dst,
                                            const void *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    putmem_nbi_wave((static_cast<char *>(dst) + my_pe_in_team * nelems), 
                    (static_cast<const char *>(src) + j * nelems), nelems, dest_pe);
  }
  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync_wave(my_pe, pe_start, stride, pe_size, pSync);
}

__device__ void IPCContext::alltoallmem_linear_thread_puts_wave(rocshmem_team_t team,
    void *dst, const void *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  size_t alltoall_pSync_offset = (team_obj->alltoall_sequence_number % 2) * pe_size;

  int tid = get_flat_block_id();
  // min(get_flat_block_size(), WF_SIZE)
  int step_size = get_flat_block_size() < WF_SIZE ? get_flat_block_size() : WF_SIZE;

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    putmem_nbi((static_cast<char *>(dst) + my_pe_in_team * nelems), 
                    (static_cast<const char *>(src) + j * nelems), nelems, dest_pe);
  }
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    fence(dest_pe);
    ptrdiff_t L_offset = reinterpret_cast<char*>(&pSync[alltoall_pSync_offset + my_pe_in_team]) - wrk_sync_pool_bases_[my_pe];
    ipcImpl_.ipcAMOAdd(reinterpret_cast<long*>(wrk_sync_pool_bases_[dest_pe] + L_offset), 1L);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    long *sync_flag = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flag) != 1) { }

    //quiet(dest_pe);// needed to quiet add when it is nbi in gda, it is not nbi in ipc

    pSync[alltoall_pSync_offset + dest_pe] = ROCSHMEM_SYNC_VALUE;
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  sync_wave(team);
}


__device__ int IPCContext::fcollectmem_wave(rocshmem_team_t team, void *dst,
                                            const void *src, int nelems) {
  if (dst == nullptr || src == nullptr || team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;
  
  fcollectmem_linear_wave(team, dst, src, nelems);

  return ROCSHMEM_SUCCESS;
}

__device__ void IPCContext::fcollectmem_linear_wave(rocshmem_team_t team, void *dst,
                                            const void *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    putmem_nbi_wave(reinterpret_cast<char *>(dst) + my_pe_in_team * nelems, src, nelems, dest_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync_wave(my_pe, pe_start, stride, pe_size, pSync);
}

__device__ void IPCContext::fcollectmem_wg(rocshmem_team_t team, void *dst,
                                     const void *src, int nelems) {
  fcollectmem_linear_wg(team, dst, src, nelems);
}

__device__ void IPCContext::fcollectmem_linear_wg(rocshmem_team_t team, void *dst,
                                            const void *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    putmem_nbi_wg(reinterpret_cast<char *>(dst) + my_pe_in_team * nelems, src, nelems, dest_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync_wg(my_pe, pe_start, stride, pe_size, pSync);
}

}  // namespace rocshmem
