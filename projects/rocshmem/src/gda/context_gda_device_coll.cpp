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
                                                    int stride, int n_pes,
                                                    int64_t *pSync) {
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
      put(&pSync[0], &pSync[0], 1, j);
#if defined(__gfx90a__)
      __threadfence_system();
#endif /* __gfx90a__ */
    }
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    // Mark current PE offset as reached
    size_t pe_offset = (pe - PE_start) / stride;
    pSync[pe_offset] = flag_val;
    put(&pSync[pe_offset], &pSync[pe_offset], 1, PE_start);
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
                                                       int stride, int n_pes,
                                                       int64_t *pSync) {
  int64_t flag_val{1};

  if (pe == PE_start) {
    int wf_id = get_flat_block_id() / WF_SIZE;
    int wf_count = (int) ceil((double)get_flat_block_size() / (double)WF_SIZE);
    bool wf_leader = 0 == get_active_lane_num();

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
      put_nbi_wave(&pSync[0], &flag_val, 1, j);
    }

    for (int i = wf_id + 1, j = PE_start + stride + wf_id;
             i < n_pes;
             i+= wf_count, j += (wf_count * stride)) {
      pe_quiet(j);
    }

    __syncthreads();

    if (is_thread_zero_in_block()) {
      pSync[0] = ROCSHMEM_SYNC_VALUE;
    }
  } else {
    if (is_thread_zero_in_block()) {
      // Mark current PE offset as reached
      size_t pe_offset = (pe - PE_start) / stride;
      put(&pSync[pe_offset], &flag_val, 1, PE_start);
      wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
      pSync[0] = ROCSHMEM_SYNC_VALUE;
      __threadfence_system();
    }
  }
}

__device__ void GDAContext::internal_atomic_barrier(int pe, int PE_start,
                                                    int stride, int n_pes,
                                                    int64_t *pSync) {
  int64_t flag_val{1};
  if (pe == PE_start) {
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, (int64_t)(n_pes - 1));
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();

    pSync[0] = flag_val;
    for (int i = 1, j = PE_start + stride; i < n_pes; ++i, j += stride) {
      put_nbi(&pSync[0], &pSync[0], 1, j);
    }
    quiet();
    pSync[0] = ROCSHMEM_SYNC_VALUE;
  } else {
    amo_add<int64_t>(&pSync[0], flag_val, PE_start);
    wait_until(&pSync[0], ROCSHMEM_CMP_EQ, flag_val);
    pSync[0] = ROCSHMEM_SYNC_VALUE;
    __threadfence_system();
  }
}

__device__ void GDAContext::internal_sync(int pe, int PE_start, int stride,
                                          int PE_size, int64_t *pSync) {
  if (PE_size < 64) {
    internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
  } else {
    internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
  }
}

__device__ void GDAContext::internal_sync_wave(int pe, int PE_start, int stride,
                                               int PE_size, int64_t *pSync) {
  if (is_thread_zero_in_wave()) {
    if (PE_size < 64) {
      internal_direct_barrier(pe, PE_start, stride, PE_size, pSync);
    } else {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
    }
  }
}

__device__ void GDAContext::internal_sync_wg(int pe, int PE_start, int stride,
                                             int PE_size, int64_t *pSync) {
  __syncthreads();
  if (PE_size < 64) {
    internal_direct_barrier_wg(pe, PE_start, stride, PE_size, pSync);
  } else {
    if (is_thread_zero_in_block()) {
      internal_atomic_barrier(pe, PE_start, stride, PE_size, pSync);
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

  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void GDAContext::sync_wave(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void GDAContext::sync_wg(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void GDAContext::sync_all() {
  internal_sync(my_pe, 0, 1, num_pes, barrier_sync);
}

__device__ void GDAContext::sync_all_wave() {
  internal_sync_wave(my_pe, 0, 1, num_pes, barrier_sync);
}

__device__ void GDAContext::sync_all_wg() {
  internal_sync_wg(my_pe, 0, 1, num_pes, barrier_sync);
}

__device__ void GDAContext::barrier_all() {
  quiet();
  sync_all();
}

__device__ void GDAContext::barrier_all_wave() {
  quiet_wave();
  sync_all_wave();
}

__device__ void GDAContext::barrier_all_wg() {
  if (is_wave_zero_in_block()) {
    quiet_wave();
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

  quiet();
  internal_sync(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void GDAContext::barrier_wave(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  quiet_wave();
  internal_sync_wave(pe, pe_start, pe_stride, pe_size, p_sync);
}

__device__ void GDAContext::barrier_wg(rocshmem_team_t team) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe = team_obj->my_pe_in_world;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_stride = team_obj->tinfo_wrt_world->stride;
  int pe_size = team_obj->num_pes;
  long *p_sync = team_obj->barrier_pSync;

  if (is_wave_zero_in_block()) {
    quiet_wave();
  }
  internal_sync_wg(pe, pe_start, pe_stride, pe_size, p_sync);
  __syncthreads();
}

}  // namespace rocshmem
