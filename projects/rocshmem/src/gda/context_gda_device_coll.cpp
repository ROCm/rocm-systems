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
#include "context_incl.hpp"
#include "context_gda_tmpl_device.hpp"
#include "util.hpp"
#include "gda_team.hpp"

namespace rocshmem {

__device__ void GDAContext::internal_direct_barrier(int pe, int PE_start,
    int stride, int n_pes, int64_t *pSync, ActiveWFInfo &wf_info) {
  int64_t flag_val{1};
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
    __threadfence_system();

    // Announce to other PEs that all have reached
    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      pSync[0] = flag_val;
      internal_putmem(&pSync[0], &pSync[0], sizeof(*pSync), j, j, wf_info);
#if defined(__gfx90a__)
      __threadfence_system();
#endif /* __gfx90a__ */
    }
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    // Mark current PE offset as reached
    size_t pe_offset = (pe - PE_start) / stride;
    pSync[pe_offset] = flag_val;
    internal_putmem(&pSync[pe_offset], &pSync[pe_offset], sizeof(*pSync),
      PE_start, PE_start, wf_info);
#if defined(__gfx90a__)
    __threadfence_system();
#endif /* __gfx90a__ */
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    pSync[pe_offset] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();
  }
}

__device__ void GDAContext::internal_direct_barrier_wg(int pe, int PE_start,
    int stride, int n_pes, int64_t *pSync, ActiveWFInfo &wf_info) {
  int64_t flag_val{1};

  if (pe == PE_start) {
    int wf_id = get_flat_block_id() / WF_SIZE;
    int wf_count = (int) ceil((double)get_flat_block_size() / (double)WF_SIZE);
    bool wf_leader = is_first_active_lane();

    // Go through all PE offsets (except current offset = 0)
    // and wait until they all reach
    if (wf_leader) {
      for (int j = wf_id + 1; j < n_pes; j+= wf_count) {
        wait_until(&pSync[j], ROCSHMEM_CMP_EQ, flag_val);
        pSync[j] = ROCSHMEM_SYNC_VALUE;
      }
    }

    __syncthreads();

    // Announce to other PEs that all have reached
    for (int i = wf_id + 1, j = PE_start + stride + wf_id;
             i < n_pes;
             i+= wf_count, j += (wf_count * stride)) {
      internal_putmem_nbi_wave(&pSync[0], &flag_val, sizeof(*pSync), j,
        j, wf_info);
    }

    for (int i = wf_id + 1, j = PE_start + stride + wf_id;
             i < n_pes;
             i+= wf_count, j += (wf_count * stride)) {
      qps[j].quiet(wf_info);
    }

    __syncthreads();

    if (is_thread_zero_in_block()) {
      pSync[0] = ROCSHMEM_SYNC_VALUE;
    }
  } else {
    if (is_thread_zero_in_block()) {
      // Mark current PE offset as reached
      size_t pe_offset = (pe - PE_start) / stride;
      internal_putmem(&pSync[pe_offset], &flag_val, sizeof(*pSync), PE_start,
        PE_start, wf_info);
      wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
      pSync[0] = ROCSHMEM_SYNC_VALUE;
      __threadfence_system();
    }
  }
}

__device__ void GDAContext::internal_atomic_barrier(int pe, int PE_start,
    int stride, int n_pes, int64_t *pSync, ActiveWFInfo &wf_info) {
  int64_t flag_val{1};
  if (pe == PE_start) {
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, (int64_t)(n_pes - 1));
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();

    pSync[0] = flag_val;
    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      internal_putmem_nbi(&pSync[0], &pSync[0], sizeof(*pSync), j, j, wf_info);
    }
    quiet();
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    internal_amo_add<int64_t>(&pSync[0], flag_val, PE_start,
      PE_start, wf_info);
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();
  }
}

__device__ void GDAContext::internal_sync(int pe, int PE_start, int stride,
    int PE_size, int64_t *pSync, ActiveWFInfo &wf_info) {
  if (PE_size < 64) {
    internal_direct_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
  } else {
    internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
  }
}

__device__ void GDAContext::internal_sync_wave(int pe, int PE_start,
    int stride, int PE_size, int64_t *pSync, ActiveWFInfo &wf_info) {
  if (is_thread_zero_in_wave()) {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
    }
  }
}

__device__ void GDAContext::internal_sync_wg(int pe, int PE_start, int stride,
    int PE_size, int64_t *pSync, ActiveWFInfo &wf_info) {
  __syncthreads();
  if (PE_size < 64) {
    internal_direct_barrier_wg(pe, PE_start, stride, PE_size, pSync, wf_info);
  } else {
    if (is_thread_zero_in_block()) {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync, wf_info);
    }
  }
  __threadfence_system();
  __syncthreads();
}

__device__ void GDAContext::sync(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_);
  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::sync_wave(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::sync_wg(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::sync_all() {
  ActiveWFInfo wf_info(ctx_id_);
  internal_sync(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync, wf_info);
}

__device__ void GDAContext::sync_all_wave() {
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
  internal_sync_wave(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync, wf_info);
}

__device__ void GDAContext::sync_all_wg() {
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  internal_sync_wg(constmem.my_pe, 0, 1, constmem.num_pes, barrier_sync, wf_info);
}

__device__ void GDAContext::barrier_all() {
  quiet();
  sync_all();
}

__device__ void GDAContext::barrier_all_wave() {
  quiet();
  sync_all_wave();
}

__device__ void GDAContext::barrier_all_wg() {
  if (is_wave_zero_in_block()) {
    quiet();
  }
  sync_all_wg();
  __syncthreads();
}

__device__ void GDAContext::barrier(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_);
  internal_quiet(wf_info);
  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::barrier_wave(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);
  internal_quiet(wf_info);
  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::barrier_wg(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (is_wave_zero_in_block()) {
    internal_quiet(wf_info);
  }
  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync, wf_info);
  __syncthreads();
}

__device__ void GDAContext::internal_put_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (i != constmem.my_pe) {
        internal_putmem_nbi_wave(dst, src, nelems, i, i, wf_info);
      }
    }
    memcpy_wave<MemcpyKind::Put>(dst, const_cast<void *>(src), nelems);
  }
}

__device__ void GDAContext::internal_get_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe != pe_root) {
    internal_getmem_wave(dst, src, nelems, pe_root, pe_root, wf_info);
  } else {
    memcpy_wave<MemcpyKind::Get>(dst, const_cast<void *>(src), nelems);
  }
}

__device__ int GDAContext::broadcastmem_wave(rocshmem_team_t team, void *dest, 
    const void* source, int nelement, int PE_root) {
  if (dest == nullptr || 
    source == nullptr || 
    team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(PE_root);
  internal_broadcastmem_wave(dest, source, nelement, pe_root_world, pe_start, stride,
               pe_size, p_sync);
  return ROCSHMEM_SUCCESS;
}

__device__ void GDAContext::internal_broadcastmem_wave(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) { //TODO: optimized for IPC
    internal_put_broadcastmem_wave(dst, src, nelems, pe_root, pe_start, stride,
      pe_size, wf_info);
  } else {
    internal_get_broadcastmem_wave(dst, src, nelems, pe_root, wf_info);
  }

  // Synchronize on completion of broadcast
  internal_sync_wave(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

__device__ void GDAContext::internal_put_broadcastmem_wg(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (constmem.my_pe != i)
        internal_putmem_nbi_wg(dst, src, nelems, i, i, wf_info);
    }
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nelems);
  }
}

__device__ void GDAContext::internal_get_broadcastmem_wg(void *dst, const void *src, int nelems,
    int pe_root, ActiveWFInfo &wf_info){
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<void *>(src), nelems);
  } else {
    internal_getmem_wg(dst, src, nelems, pe_root, pe_root, wf_info);
  }
}

__device__ void GDAContext::broadcastmem_wg(rocshmem_team_t team, void *dst,
    const void *src, int nelems, int pe_root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);
  internal_broadcastmem_wg(dst, src, nelems, pe_root_world, pe_start, stride,
               pe_size, p_sync);
}

__device__ void GDAContext::internal_broadcastmem_wg(void *dst, const void *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) { //TODO: optimized for IPC
    internal_put_broadcastmem_wg(dst, src, nelems, pe_root, pe_start, stride,
      pe_size, wf_info);
  } else {
    internal_get_broadcastmem_wg(dst, src, nelems, pe_root, wf_info);
  }

  // Synchronize on completion of broadcast
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

}  // namespace rocshmem
