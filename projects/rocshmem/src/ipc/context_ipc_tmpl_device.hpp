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

#ifndef LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "constmem.hpp"
#include "context_ipc_device.hpp"
#include "log.hpp"
#include "util.hpp"
#include "tile_memcpy_device.hpp"
#include "ipc_team.hpp"
#include "rocshmem_calc.hpp"

#include <hip/hip_runtime.h>

namespace rocshmem {

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <typename T>
__device__ void IPCContext::p(T *dest, T value, int pe) {
  putmem_scalar(dest, &value, sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put(T *dest, const T *source, size_t nelems, int pe) {
  putmem(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ T IPCContext::g(const T *source, int pe) {
  T ret;
  getmem_scalar(&ret, source, sizeof(T), pe);
  return ret;
}

template <typename T>
__device__ void IPCContext::get(T *dest, const T *source, size_t nelems, int pe) {
  getmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void IPCContext::get_nbi(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

// Atomics
template <typename T>
__device__ void IPCContext::amo_add(void *dest, T value, int pe) {
  ipcImpl_.ipcAMOAdd(reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ void IPCContext::amo_set(void *dest, T value, int pe) {
  ipcImpl_.ipcAMOSet(reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ T IPCContext::amo_swap(void *dest, T value, int pe) {
  return ipcImpl_.ipcAMOSwap(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_and(void *dest, T value, int pe) {
  return ipcImpl_.ipcAMOFetchAnd(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ void IPCContext::amo_and(void *dest, T value, int pe) {
  ipcImpl_.ipcAMOAnd(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_or(void *dest, T value, int pe) {
  return ipcImpl_.ipcAMOFetchOr(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ void IPCContext::amo_or(void *dest, T value, int pe) {
  ipcImpl_.ipcAMOOr(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_xor(void *dest, T value, int pe) {
  return ipcImpl_.ipcAMOFetchXor(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ void IPCContext::amo_xor(void *dest, T value, int pe) {
  ipcImpl_.ipcAMOXor(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ void IPCContext::amo_cas(void *dest, T value, T cond, int pe) {
  ipcImpl_.ipcAMOCas(reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), cond, value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_add(void *dest, T value, int pe) {
  return ipcImpl_.ipcAMOFetchAdd(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_cas(void *dest, T value, T cond, int pe) {
  return ipcImpl_.ipcAMOFetchCas(
      reinterpret_cast<T *>(ipcImpl_.ipcPeerPtr(dest, pe)), cond, value);
}

// Collectives
template <typename T, ROCSHMEM_OP Op>
__device__ void ipc_compute_reduce(T *src, T *dst, int size, int wg_id, int wg_size) {
  for (int i = wg_id; i < size; i += wg_size) {
    OpWrap<Op>::Calc(src, dst, i);
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ void IPCContext::internal_direct_allreduce_wg(
    T *dst, const T *src, int nelems, IPCTeam *team_obj) {  // NOLINT(runtime/int)

  int stride = team_obj->tinfo_wrt_world->stride;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);

  int finish = PE_start + stride * PE_size;
  int pe = constmem.my_pe;

  int wg_id = get_flat_block_id();
  int wg_size = get_flat_block_size();
  int64_t flag_val = 1;

  for (int i = wg_id; i < nelems; i += wg_size) {
    dst[i] = src[i];
  }
  __syncthreads();

  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      internal_putmem_wg(&pWrk[pe * nelems], reinterpret_cast<const void *>(src),
                    nelems * sizeof(T), i);

      if (is_thread_zero_in_block()) {
        fence(i);
        internal_putmem(&pSync[pe], &flag_val, sizeof(*pSync), i);
      }
    }
  }
  threadfence_system();
  __syncthreads();

  // Do the compute and pSync reset in parallel.
  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      // Wait for leader thread to see that the buffer is ready.
      if (is_thread_zero_in_block()) {
        wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
      }
      __syncthreads();

      T *ptr = &pWrk[i * nelems];
      ipc_compute_reduce<T, Op>(ptr, dst, nelems, wg_id, wg_size);
      threadfence_system();
    }
  }
  __syncthreads();

  for (int i = wg_id; i < constmem.num_pes; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ void IPCContext::internal_direct_allreduce_wave(
    T *dst, const T *src, int nelems, IPCTeam *team_obj) {  // NOLINT(runtime/int)

  int stride = team_obj->tinfo_wrt_world->stride;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);

  int finish = PE_start + stride * PE_size;
  int pe = my_pe;
  int wf_tid = get_flat_block_id() % WF_SIZE;
  int64_t flag_val = 1;

  // Initialize local dst from src
  for (int i = wf_tid; i < nelems; i += WF_SIZE) {
    dst[i] = src[i];
  }

  // Put src to pWrk on all other PEs and signal via pSync
  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      internal_putmem_wave(&pWrk[pe * nelems],
        reinterpret_cast<const void *>(src), nelems * sizeof(T), i);
      if (is_thread_zero_in_wave()) {
        fence(i);
        internal_putmem(&pSync[pe], &flag_val, sizeof(*pSync), i);
      }
    }
  }

  // Reduce contributions from all other PEs into dst
  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      if (is_thread_zero_in_wave()) {
        wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
      }
      T *ptr = &pWrk[i * nelems];
      for (int j = wf_tid; j < nelems; j += WF_SIZE) {
        OpWrap<Op>::Calc(ptr, dst, j);
      }
    }
  }

  // Reset pSync entries
  if (is_thread_zero_in_wave()) {
    for (int i = 0; i < num_pes; i++) {
      pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    __threadfence_system();
  }
}

template <typename T, ROCSHMEM_OP Op>
__device__ int IPCContext::reduce_wave(rocshmem_team_t team, T *dest,
                                       const T *source, int nreduce) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int PE_size = team_obj->tinfo_wrt_world->size;

  size_t direct_pWrk = PE_size * nreduce;
  size_t direct_pSync = PE_size;
  size_t provided_pWrk = max(nreduce / 2 + 1, ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  size_t ring_pSync = 2 * PE_size;

  if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
    internal_direct_allreduce_wave<T, Op>(dest, source, nreduce, team_obj);
  } else {
    if (ring_pSync <= ROCSHMEM_REDUCE_SYNC_SIZE) {
      size_t ring_pWrk = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;
      int chunk_size = ring_pWrk / PE_size;
      int seg_size = chunk_size * PE_size;

      int n_seg = nreduce / seg_size;
      int n_seg_up = (nreduce - 1) / seg_size + 1;

      if (n_seg > 0) {
        internal_ring_allreduce_wave<T, Op>(dest, source, nreduce, team_obj,
                                            n_seg, seg_size, chunk_size);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;
        if (p_chunk > 0) {
          barrier_wave(team);
          internal_ring_allreduce_wave<T, Op>(p_dst, p_src,
                                              (p_chunk * PE_size), team_obj, 1,
                                              (p_chunk * PE_size), p_chunk);
        }

        if ((p_chunk * PE_size) < p_count) {
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);
          barrier_wave(team);
          internal_direct_allreduce_wave<T, Op>(p_dst, p_src2, p_count, team_obj);
        }
      }
    } else {
      LOGD_WARN("Unsupported reduction size for IPC wave conduit.");
      return ROCSHMEM_ERROR;
    }
  }
  barrier_wave(team);
  return ROCSHMEM_SUCCESS;
}

/*
 * Visual representation of the ring_allreduce algorithm below
 * assuming 4 PEs and a single segment.
 *
 *         Initial state
 *  PE#     0              1             2              3
 *        [00]           [10]          [20]           [30]
 *        [01]           [11]          [21]           [31]
 *        [02]           [12]          [22]           [32]
 *        [03]           [13]          [23]           [33]
 *
 * Loop 1:
 *        iter 0
 *  PE#     0              1             2              3
 *        [00+30]        [10]          [20]           [30]
 *        [01]           [01+11]       [21]           [31]
 *        [02]           [12]          [12+22]        [32]
 *        [03]           [13]          [23]           [23+33]
 *
 *        iter 1
 *  PE#     0              1             2              3
 *        [00+30]        [00+10+30]    [20]           [30]
 *        [01]           [01+11]       [01+11+21]     [31]
 *        [02]           [12]          [12+22]        [12+22+32]
 *        [03+23+33]     [13]          [23]           [23+33]
 *
 *        iter 2
 *  PE#     0              1             2              3
 *        [00+30]        [00+10+30]    [00+10+20+30]  [30]
 *        [01]           [01+11]       [01+11+21]     [01+11+21+31]
 *        [02+12+22+32]  [12]          [12+22]        [12+22+32]
 *        [03+23+33]     [03+13+23+33] [23]           [23+33]
 *
 * Loop 2:
 *
 *       iter 3
 *  PE#     0              1             2              3
 *        [00+30]        [00+10+30]    [00+10+20+30]  [00+10+20+30]
 *        [01+11+21+31]  [01+11]       [01+11+21]     [01+11+21+31]
 *        [02+12+22+32]  [02+12+22+32] [12+22]        [12+22+32]
 *        [03+23+33]     [03+13+23+33] [03+13+23+33]  [23+33]
 *
 *       iter 4
 *  PE#     0              1             2              3
 *        [00+10+20+30]  [00+10+30]    [00+10+20+30]  [00+10+20+30]
 *        [01+11+21+31]  [01+11+21+31] [01+11+21]     [01+11+21+31]
 *        [02+12+22+32]  [02+12+22+32] [02+12+22+32]  [12+22+32]
 *        [03+23+33]     [03+13+23+33] [03+13+23+33]  [03+13+23+33]
 *
 *        iter 5
 *  PE#     0              1             2              3
 *        [00+10+20+30]  [00+10+20+30] [00+10+20+30]  [00+10+20+30]
 *        [01+11+21+31]  [01+11+21+31] [01+11+21+31]  [01+11+21+31]
 *        [02+12+22+32]  [02+12+22+32] [02+12+22+32]  [02+12+22+32]
 *        [03+13+23+33]  [03+13+23+33] [03+13+23+33]  [03+13+23+33]
 */
template <typename T, ROCSHMEM_OP Op>
__device__ void IPCContext::internal_ring_allreduce_wg(
    T *dst, const T *src, int nelems, IPCTeam *team_obj,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size) {

  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  int my_pe_in_team = team_obj->my_pe;

  int off_seg, off_send, off_recv;
  int send_pe = (my_pe_in_team + 1) % PE_size;
  // send_pe is relative to team, convert it relative to team world
  send_pe = team_obj->get_pe_in_world(send_pe);
  long wait_val;  // NOLINT(runtime/int)

  int wg_size = get_flat_block_size();
  int wg_id = get_flat_block_id();

  for (int i = wg_id; i < nelems; i += wg_size) {
    dst[i] = src[i];
  }
  __syncthreads();

  for (int seg = 0; seg < n_seg; seg++) {
    off_seg = seg * seg_size;
    // Loop 1 in the algorithm above
    for (int iter = 0; iter < PE_size - 1; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      off_recv = (((my_pe_in_team - iter + 2 * PE_size) % PE_size) * chunk_size);

      internal_putmem_wg(reinterpret_cast<void *>(&pWrk[off_send]),
                    reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    chunk_size * sizeof(T), send_pe);
      fence(send_pe);
      if (is_thread_zero_in_block()) {
        wait_val = seg + 100;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      detail::atomic::threadfence<detail::atomic::memory_scope_system,
                             detail::atomic::memory_order_acquire>();
      __syncthreads();

      ipc_compute_reduce<T, Op>(&pWrk[off_recv], &dst[off_seg + off_recv],
                            chunk_size, wg_id, wg_size);
    }

    // Loop 2 in the example above
    for (int iter = PE_size - 1; iter < 2 * PE_size - 2; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      putmem_nbi_wg(reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    chunk_size * sizeof(T), send_pe);
      __syncthreads();
      fence(send_pe);
      if (is_thread_zero_in_block()) {
        wait_val = seg + 10;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      detail::atomic::threadfence<detail::atomic::memory_scope_system,
                             detail::atomic::memory_order_acquire>();

      __syncthreads();
    }
  }

  for (int i = wg_id; i < 2 * constmem.num_pes - 2; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ void IPCContext::internal_ring_allreduce_wave(
    T *dst, const T *src, int nelems, IPCTeam *team_obj,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size) {

  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  int my_pe_in_team = team_obj->my_pe;

  int off_seg, off_send, off_recv;
  int send_pe = (my_pe_in_team + 1) % PE_size;
  send_pe = team_obj->get_pe_in_world(send_pe);
  long wait_val;  // NOLINT(runtime/int)
  int wave_size = wave_SZ();
  int wf_tid = get_flat_block_id() % wave_size;

  for (int i = wf_tid; i < nelems; i += wave_size) {
    dst[i] = src[i];
  }
  __builtin_amdgcn_wave_barrier();

  for (int seg = 0; seg < n_seg; seg++) {
    off_seg = seg * seg_size;
    // Loop 1: reduce-scatter
    for (int iter = 0; iter < PE_size - 1; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      off_recv = (((my_pe_in_team - iter + 2 * PE_size) % PE_size) * chunk_size);

      internal_putmem_wave(reinterpret_cast<void *>(&pWrk[off_send]),
                           reinterpret_cast<void *>(&dst[off_send + off_seg]),
                           chunk_size * sizeof(T), send_pe);
      __builtin_amdgcn_wave_barrier();
      fence(send_pe);

      wait_val = seg + 100;
      if (is_thread_zero_in_wave()) {
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
      }
      wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      detail::atomic::threadfence<detail::atomic::memory_scope_system,
                             detail::atomic::memory_order_acquire>();
      __builtin_amdgcn_wave_barrier();

      for (int j = wf_tid; j < chunk_size; j += wave_size) {
        OpWrap<Op>::Calc(&pWrk[off_recv], &dst[off_seg + off_recv], j);
      }
      __builtin_amdgcn_wave_barrier();
    }

    // Loop 2: all-gather
    for (int iter = PE_size - 1; iter < 2 * PE_size - 2; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      putmem_nbi_wave(reinterpret_cast<void *>(&dst[off_send + off_seg]),
                      reinterpret_cast<void *>(&dst[off_send + off_seg]),
                      chunk_size * sizeof(T), send_pe);
      __builtin_amdgcn_wave_barrier();
      fence(send_pe);
      
      wait_val = seg + 10;
      if (is_thread_zero_in_wave()) {
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
      }
      wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      detail::atomic::threadfence<detail::atomic::memory_scope_system,
                             detail::atomic::memory_order_acquire>();
      __builtin_amdgcn_wave_barrier();
    }
  }

  if (is_thread_zero_in_wave()) {
    for (int i = 0; i < 2 * constmem.num_pes - 2; i++) {
      pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
  }
}

template <typename T, ROCSHMEM_OP Op>
__device__ int IPCContext::reduce_wg(rocshmem_team_t team, T *dest,
                                  const T *source, int nreduce) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int PE_size = team_obj->tinfo_wrt_world->size;

  size_t direct_pWrk = PE_size * nreduce;
  size_t direct_pSync = PE_size;
  size_t ring_pSync = 2 * PE_size;
  size_t provided_pWrk = max(nreduce / 2 + 1, ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
    internal_direct_allreduce_wg<T, Op>(dest, source, nreduce, team_obj);
  } else {
    if (ring_pSync <= ROCSHMEM_REDUCE_SYNC_SIZE) {
      size_t ring_pWrk = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;
      // integer division truncating value
      int chunk_size = ring_pWrk / PE_size;
      int seg_size = chunk_size * PE_size;

      // integer division truncating value
      int n_seg = nreduce / seg_size;
      // integer division rounding up
      int n_seg_up = (nreduce - 1) / seg_size + 1;

      if (n_seg > 0) {
        internal_ring_allreduce_wg<T, Op>(dest, source, nreduce, team_obj, n_seg,
                                       seg_size, chunk_size);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;
        if (p_chunk > 0) {
          barrier_wg(team);
          internal_ring_allreduce_wg<T, Op>(p_dst, p_src,
                                         (p_chunk * PE_size), team_obj, 1,
                                         (p_chunk * PE_size), p_chunk);
        }

        if ((p_chunk * PE_size) < p_count) {
          // Final elements need to use direct_allreduce
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);
          barrier_wg(team);
          internal_direct_allreduce_wg<T, Op>(p_dst, p_src2, p_count, team_obj);
        }
      }
    } else {
      LOGD_WARN("Unsupported reduction size for IPC conduit.");
      return ROCSHMEM_ERROR;
    }
  }
  barrier_wg(team);
  return ROCSHMEM_SUCCESS;
}

/*
 * Reduce-scatter: PE r receives the element-wise reduction of
 * source[r*nreduce .. (r+1)*nreduce - 1] across all PEs into dest[0..nreduce-1].
 *
 * Only workgroup 0 (is_block_zero_in_grid) runs the algorithm; all other
 * workgroups wait at the final barrier_wg.  This prevents concurrent
 * accumulation races when multiple workgroups share the same team
 * pSync/pWrk/dest buffers.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ int IPCContext::reduce_scatter_wg(rocshmem_team_t team, T *dest,
                                             const T *source, int nreduce) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int PE_size   = team_obj->tinfo_wrt_world->size;
  int PE_start  = team_obj->tinfo_wrt_world->pe_start;
  int stride    = team_obj->tinfo_wrt_world->stride;
  int team_rank = (my_pe - PE_start) / stride;

  long  *pSync = team_obj->reduce_pSync;
  T     *pWrk  = reinterpret_cast<T *>(team_obj->pWrk);

  int wg_id   = get_flat_block_id();
  int wg_size = get_flat_block_size();

  int pWrk_elems = (int)(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  int chunk_size = max(1, pWrk_elems / PE_size);
  int n_chunks   = (nreduce + chunk_size - 1) / chunk_size;
  int64_t flag_val = 1;
  int finish = PE_start + stride * PE_size;

  // Only workgroup 0 runs the reduction algorithm; other workgroups participate
  // in the barriers only (same number of barrier_wg calls as WG 0).
  for (int c = 0; c < n_chunks; c++) {
    if (is_block_zero_in_grid()) {
      int offset = c * chunk_size;
      int count  = min(chunk_size, nreduce - offset);

      // Seed dest[offset..offset+count) from my own contribution.
      for (int j = wg_id; j < count; j += wg_size) {
        dest[offset + j] = source[team_rank * nreduce + offset + j];
      }
      __syncthreads();

      // Send my contribution for each remote PE's output block, then signal.
      for (int i = PE_start; i < finish; i += stride) {
        if (i != my_pe) {
          int remote_rank = (i - PE_start) / stride;
          internal_putmem_wg(&pWrk[team_rank * chunk_size],
                             reinterpret_cast<const void *>(
                                 source + remote_rank * nreduce + offset),
                             count * sizeof(T), i);
          if (is_thread_zero_in_block()) {
            fence(i);
            internal_putmem(&pSync[team_rank], &flag_val, sizeof(*pSync), i);
          }
        }
      }
      threadfence_system();
      __syncthreads();

      // Wait for each remote PE s, then accumulate into dest.
      for (int i = PE_start; i < finish; i += stride) {
        if (i != my_pe) {
          int remote_rank = (i - PE_start) / stride;
          if (is_thread_zero_in_block()) {
            wait_until(&pSync[remote_rank], ROCSHMEM_CMP_EQ, flag_val);
          }
          __syncthreads();
          ipc_compute_reduce<T, Op>(&pWrk[remote_rank * chunk_size],
                                    dest + offset, count, wg_id, wg_size);
          threadfence_system();
        }
      }
      __syncthreads();

      // Reset pSync before reuse.
      for (int j = wg_id; j < PE_size; j += wg_size) {
        pSync[j] = ROCSHMEM_SYNC_VALUE;
      }
      __syncthreads();
      // Sync with workgroup 0 of other PEs
      barrier_wg(team);
    }
  }

  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ int IPCContext::reduce_scatter_wave(rocshmem_team_t team, T *dest,
                                               const T *source, int nreduce) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int PE_size   = team_obj->tinfo_wrt_world->size;
  int PE_start  = team_obj->tinfo_wrt_world->pe_start;
  int stride    = team_obj->tinfo_wrt_world->stride;
  int team_rank = (my_pe - PE_start) / stride;

  long  *pSync = team_obj->reduce_pSync;
  T     *pWrk  = reinterpret_cast<T *>(team_obj->pWrk);

  int wave_tid  = get_flat_block_id() % WF_SIZE;

  int pWrk_elems = (int)(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  int chunk_size = max(1, pWrk_elems / PE_size);
  int n_chunks   = (nreduce + chunk_size - 1) / chunk_size;
  int finish = PE_start + stride * PE_size;
  
  for (int c = 0; c < n_chunks; c++) {
    int offset = c * chunk_size;
    int count  = min(chunk_size, nreduce - offset);
    int64_t flag_val = static_cast<int64_t>(c) + 1;
    // Seed dest[offset..offset+count) from my own contribution.
    for (int j = wave_tid; j < count; j += WF_SIZE) {
      dest[offset + j] = source[team_rank * nreduce + offset + j];
    }
	  __builtin_amdgcn_wave_barrier();

    // Send my contribution to each remote PE's pWrk slot, then signal.
    for (int i = PE_start; i < finish; i += stride) {
      if (i != my_pe) {
        int remote_rank = (i - PE_start) / stride;
        internal_putmem_wave(&pWrk[team_rank * chunk_size],
                              reinterpret_cast<const void *>(
                                  source + remote_rank * nreduce + offset),
                              count * sizeof(T), i);
        if (is_thread_zero_in_wave()) {
          fence(i);
          internal_putmem(&pSync[team_rank], &flag_val, sizeof(*pSync), i);
        }
      }
    }
    threadfence_system();
    __builtin_amdgcn_wave_barrier();

    // Wait for each remote PE's signal, then accumulate into dest.
    for (int i = PE_start; i < finish; i += stride) {
      if (i != my_pe) {
        int remote_rank = (i - PE_start) / stride;
        if (is_thread_zero_in_wave()) {
          wait_until(&pSync[remote_rank], ROCSHMEM_CMP_EQ, flag_val);
        }
	      __builtin_amdgcn_wave_barrier();
        T *src_chunk = &pWrk[remote_rank * chunk_size];
        T *dst_chunk = dest + offset;
        for (int j = wave_tid; j < count; j += WF_SIZE) {
          OpWrap<Op>::Calc(src_chunk, dst_chunk, j);
        }
        threadfence_system();
      }
    }
    
    sync_wave(team);
    __builtin_amdgcn_wave_barrier();

  }
  // Reset pSync before reuse.
  for (int j = wave_tid; j < PE_size; j += WF_SIZE) {
    pSync[j] = ROCSHMEM_SYNC_VALUE;
  }

  sync_wave(team);
  __builtin_amdgcn_wave_barrier();
  barrier_wave(team);

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void IPCContext::internal_put_broadcast_wave(T *dst, const T *src, int nelems, int pe_root, int pe_start,
    int stride, int pe_size) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
        put_nbi_wave(dst, src, nelems, i);
    }
  }
}

template <typename T>
__device__ void IPCContext::internal_get_broadcast_wave(
  T *dst, const T *src, int nelems, int pe_root) {
    get_wave(dst, src, nelems, pe_root);
}

template <typename T>
__device__ int IPCContext::broadcast_wave(rocshmem_team_t team,
                              T *dest, const T *source, int nelems, int PE_root) {
  if (dest == nullptr || 
    source == nullptr || 
    team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(PE_root);
  internal_broadcast_wave<T>(dest, source, nelems, pe_root_world,
                              pe_start, stride, pe_size, p_sync);

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void IPCContext::internal_broadcast_wave(T *dst, const T *src, int nelems,
                                      int pe_root, int pe_start,
                                      int stride, int pe_size,
                                      long *p_sync) {  // NOLINT(runtime/int)
  if (constmem.num_pes < 4) {
    internal_put_broadcast_wave<T>(dst, src, nelems, pe_root, pe_start, stride, pe_size);
  } else {
    internal_get_broadcast_wave<T>(dst, src, nelems, pe_root);
  }

  // Synchronize on completion of broadcast
  internal_sync_wave(my_pe, pe_start, stride, pe_size, p_sync);
}
  
template <typename T>
__device__ void IPCContext::broadcast_wg(rocshmem_team_t team, T *dst,
                                      const T *src, int nelems, int pe_root) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);
  internal_broadcastmem_wg(dst, src, nelems * sizeof(T), pe_root_world, pe_start, stride,
               pe_size, p_sync);
}

template <typename T>
__device__ void IPCContext::alltoall_wg(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
#if defined(USE_SDMA)
  if (sizeof(T) * nelems < 512 || ipcImpl_.sdmaImpl_.sdmaEnabled)
#else
  if (sizeof(T) * nelems < 512)
#endif
    alltoall_linear_thread_puts_wg(team, dst, src, nelems);
  else
    alltoall_linear_wg(team, dst, src, nelems);
}

template <typename T>
__device__ void IPCContext::alltoall_linear_wg(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi_wg(&dst[my_pe_in_team * nelems], &src[j * nelems], nelems, dest_pe);
  }
  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void IPCContext::alltoall_linear_thread_puts_wg(rocshmem_team_t team,
    T *dst, const T *src, int nelems) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  size_t alltoall_pSync_offset = (team_obj->alltoall_sequence_number % 2) * pe_size;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi(
      &dst[my_pe_in_team * nelems],
      &src[j * nelems], nelems, dest_pe);
  }
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    fence(dest_pe);
    ptrdiff_t L_offset = reinterpret_cast<char*>(&pSync[alltoall_pSync_offset + my_pe_in_team]) - wrk_sync_pool_bases_[constmem.my_pe];
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

template <typename T>
__device__ int IPCContext::alltoall_wave(rocshmem_team_t team, T *dest,
                                  const T *source, int nelems) {
  if (dest == nullptr || source == nullptr || team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  internal_alltoallmem_wave(team, dest, source, nelems * sizeof(T));

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void IPCContext::alltoallv([[maybe_unused]] rocshmem_team_t team,
                                      [[maybe_unused]] T *dest, [[maybe_unused]] const size_t dest_nelems[],
                                      [[maybe_unused]] const size_t dest_displs[],
                                      [[maybe_unused]] T *source, [[maybe_unused]] const size_t source_nelems[],
                                      [[maybe_unused]] const size_t source_displs[]) {
  LOGD_ERROR_ABORT("ipc:alltoallv not implemented");
}

template <typename T>
__device__ void IPCContext::fcollect_wg(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  fcollectmem_linear_wg(team, dst, src, nelems * sizeof(T));
}

template <typename T>
__device__ int IPCContext::fcollect_wave(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  if (dst == nullptr || src == nullptr || team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;
  
  fcollectmem_linear_wave(team, dst, src, nelems * sizeof(T));

  return ROCSHMEM_SUCCESS;
}

// Block/wave functions
template <typename T>
__device__ void IPCContext::put_wg(T *dest, const T *source, size_t nelems, int pe) {
  putmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

  template <typename T>
__device__ void IPCContext::put_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_wg(T *dest, const T *source, size_t nelems, int pe) {
  getmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_wave(T *dest, const T *source, size_t nelems, int pe) {
  getmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void IPCContext::get_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

#define IPC_CONTEXT_PUT_SIGNAL_DEF(SUFFIX)                                                            \
  template <typename T>                                                                               \
  __device__ void IPCContext::put_signal##SUFFIX(T *dest, const T *source, size_t nelems,             \
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,     \
                                                 int pe) {                                            \
    putmem_signal##SUFFIX(dest, source, nelems * sizeof(T), sig_addr, signal, sig_op, pe);            \
  }                                                                                                   \
                                                                                                      \
  template <typename T>                                                                               \
  __device__ void IPCContext::put_signal_nbi##SUFFIX(T *dest, const T *source, size_t nelems,         \
                                                     uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                     int pe) {                                        \
    putmem_signal##SUFFIX(dest, source, nelems * sizeof(T), sig_addr, signal, sig_op, pe);            \
  }

IPC_CONTEXT_PUT_SIGNAL_DEF()
IPC_CONTEXT_PUT_SIGNAL_DEF(_wg)
IPC_CONTEXT_PUT_SIGNAL_DEF(_wave)

/******************************************************************************
 ******************** TILE API STUB IMPLEMENTATIONS ***************************
 *****************************************************************************/

// RMA Operations - Type-erased implementations
__device__ inline int IPCContext::tile_put(void* dst_data, const void* src_data,
                                           const size_t* dst_strides, const size_t* src_strides,
                                           const size_t* start_coord, const size_t* boundary,
                                           int ndim, size_t element_size, int pe,
                                           [[maybe_unused]] uint64_t flags) {
  void* remote_base = shmem_ptr(dst_data, pe);
  if (!remote_base) {
    return ROCSHMEM_ERROR;
  }

  tile_memcpy_rma<MemcpyKind::Put, TileScope::Lane>(
      remote_base, src_data, dst_strides, src_strides, start_coord, boundary,
      ndim, element_size);
  ipcImpl_.ipcQuiet();
  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_put_wave(void* dst_data, const void* src_data,
                                                const size_t* dst_strides, const size_t* src_strides,
                                                const size_t* start_coord, const size_t* boundary,
                                                int ndim, size_t element_size, int pe,
                                                [[maybe_unused]] uint64_t flags) {
  void* remote_base = shmem_ptr(dst_data, pe);
  if (!remote_base) {
    return ROCSHMEM_ERROR;
  }

  tile_memcpy_rma<MemcpyKind::Put, TileScope::Wave>(
      remote_base, src_data, dst_strides, src_strides, start_coord, boundary,
      ndim, element_size);
  if (is_thread_zero_in_wave()) {
    ipcImpl_.ipcQuiet();
  }
  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_put_wg(void* dst_data, const void* src_data,
                                              const size_t* dst_strides, const size_t* src_strides,
                                              const size_t* start_coord, const size_t* boundary,
                                              int ndim, size_t element_size, int pe,
                                              [[maybe_unused]] uint64_t flags) {
  void* remote_base = shmem_ptr(dst_data, pe);
  if (!remote_base) {
    return ROCSHMEM_ERROR;
  }

  tile_memcpy_rma<MemcpyKind::Put, TileScope::Wg>(
      remote_base, src_data, dst_strides, src_strides, start_coord, boundary,
      ndim, element_size);
  if (get_flat_block_id() == 0) {
    ipcImpl_.ipcQuiet();
  }
  __builtin_amdgcn_s_barrier();

  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_get(void* dst_data, const void* src_data,
                                           const size_t* dst_strides, const size_t* src_strides,
                                           const size_t* start_coord, const size_t* boundary,
                                           int ndim, size_t element_size, int pe,
                                           [[maybe_unused]] uint64_t flags) {
  void* remote_base = shmem_ptr(const_cast<void*>(src_data), pe);
  if (!remote_base) {
    return ROCSHMEM_ERROR;
  }

  tile_memcpy_rma<MemcpyKind::Get, TileScope::Lane>(
      dst_data, remote_base, dst_strides, src_strides, start_coord, boundary,
      ndim, element_size);
  ipcImpl_.ipcQuiet();
  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_get_wave(void* dst_data, const void* src_data,
                                                const size_t* dst_strides, const size_t* src_strides,
                                                const size_t* start_coord, const size_t* boundary,
                                                int ndim, size_t element_size, int pe,
                                                [[maybe_unused]] uint64_t flags) {
  void* remote_base = shmem_ptr(const_cast<void*>(src_data), pe);
  if (!remote_base) {
    return ROCSHMEM_ERROR;
  }

  tile_memcpy_rma<MemcpyKind::Get, TileScope::Wave>(
      dst_data, remote_base, dst_strides, src_strides, start_coord, boundary,
      ndim, element_size);
  if (is_thread_zero_in_wave()) {
    ipcImpl_.ipcQuiet();
  }
  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_get_wg(void* dst_data, const void* src_data,
                                              const size_t* dst_strides, const size_t* src_strides,
                                              const size_t* start_coord, const size_t* boundary,
                                              int ndim, size_t element_size, int pe,
                                              [[maybe_unused]] uint64_t flags) {
  void* remote_base = shmem_ptr(const_cast<void*>(src_data), pe);
  if (!remote_base) {
    return ROCSHMEM_ERROR;
  }

  tile_memcpy_rma<MemcpyKind::Get, TileScope::Wg>(
      dst_data, remote_base, dst_strides, src_strides, start_coord, boundary,
      ndim, element_size);
  if (get_flat_block_id() == 0) {
    ipcImpl_.ipcQuiet();
  }
  __builtin_amdgcn_s_barrier();

  return ROCSHMEM_SUCCESS;
}

// Collective Allgather - Type-erased implementations
__device__ inline int IPCContext::tile_allgather(rocshmem_team_t team,
                                                 void* dst_data,
                                                 const void* src_data,
                                                 const size_t* dst_strides,
                                                 const size_t* src_strides,
                                                 const size_t* start_coord,
                                                 const size_t* boundary,
                                                 int ndim,
                                                 size_t element_size,
                                                 uint64_t flags) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  int team_size = team_obj->num_pes;

  // Calculate tile extent along dimension 0
  size_t tile_extent_dim0 = boundary[0] - start_coord[0];

  // Each PE gathers tiles from all PEs in the team
  for (int src_pe_in_team = 0; src_pe_in_team < team_size; src_pe_in_team++) {
    int src_pe_world = team_obj->get_pe_in_world(src_pe_in_team);

    // Compute destination offset for this PE's tile using dst_strides[0]
    // Stack tiles along dimension 0: each PE's tile is offset by tile_extent_dim0 * dst_strides[0]
    // Destination layout: [PE0's tile][PE1's tile]...[PEn's tile]
    char* dst_offset = static_cast<char*>(dst_data) +
                       src_pe_in_team * tile_extent_dim0 * dst_strides[0] * element_size;

    // Use tile_get to fetch this PE's tile into the appropriate destination slot
    int result = tile_get(dst_offset, src_data, dst_strides, src_strides, start_coord,
                          boundary, ndim, element_size, src_pe_world, flags);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }
  }

  // Synchronize to ensure all PEs complete before any can modify buffers
  sync(team);

  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_allgather_wave(rocshmem_team_t team,
                                                      void* dst_data,
                                                      const void* src_data,
                                                      const size_t* dst_strides,
                                                      const size_t* src_strides,
                                                      const size_t* start_coord,
                                                      const size_t* boundary,
                                                      int ndim,
                                                      size_t element_size,
                                                      uint64_t flags) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  int team_size = team_obj->num_pes;

  // Calculate tile extent along dimension 0
  size_t tile_extent_dim0 = boundary[0] - start_coord[0];

  // Each PE gathers tiles from all PEs in the team (wave-collective)
  for (int src_pe_in_team = 0; src_pe_in_team < team_size; src_pe_in_team++) {
    int src_pe_world = team_obj->get_pe_in_world(src_pe_in_team);

    // Compute destination offset for this PE's tile using dst_strides[0]
    // Stack tiles along dimension 0: each PE's tile is offset by tile_extent_dim0 * dst_strides[0]
    char* dst_offset = static_cast<char*>(dst_data) +
                       src_pe_in_team * tile_extent_dim0 * dst_strides[0] * element_size;

    // Use tile_get_wave to fetch this PE's tile
    int result = tile_get_wave(dst_offset, src_data, dst_strides, src_strides, start_coord,
                                boundary, ndim, element_size, src_pe_world, flags);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }
  }

  // Synchronize to ensure all PEs complete
  sync_wave(team);

  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_allgather_wg(rocshmem_team_t team,
                                                    void* dst_data,
                                                    const void* src_data,
                                                    const size_t* dst_strides,
                                                    const size_t* src_strides,
                                                    const size_t* start_coord,
                                                    const size_t* boundary,
                                                    int ndim,
                                                    size_t element_size,
                                                    uint64_t flags) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  int team_size = team_obj->num_pes;

  // Calculate tile extent along dimension 0
  size_t tile_extent_dim0 = boundary[0] - start_coord[0];

  // Each PE gathers tiles from all PEs in the team (workgroup-collective)
  for (int src_pe_in_team = 0; src_pe_in_team < team_size; src_pe_in_team++) {
    int src_pe_world = team_obj->get_pe_in_world(src_pe_in_team);

    // Compute destination offset for this PE's tile using dst_strides[0]
    // Stack tiles along dimension 0: each PE's tile is offset by tile_extent_dim0 * dst_strides[0]
    char* dst_offset = static_cast<char*>(dst_data) +
                       src_pe_in_team * tile_extent_dim0 * dst_strides[0] * element_size;

    // Use tile_get_wg to fetch this PE's tile
    int result = tile_get_wg(dst_offset, src_data, dst_strides, src_strides, start_coord,
                              boundary, ndim, element_size, src_pe_world, flags);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }
  }

  // Synchronize to ensure all PEs complete
  sync_wg(team);

  return ROCSHMEM_SUCCESS;
}

// Collective Broadcast - Type-erased implementations
__device__ inline int IPCContext::tile_broadcast(rocshmem_team_t team,
                                                 void* dst_data,
                                                 const void* src_data,
                                                 const size_t* dst_strides,
                                                 const size_t* src_strides,
                                                 const size_t* start_coord,
                                                 const size_t* boundary,
                                                 int ndim,
                                                 size_t element_size,
                                                 int pe_root,
                                                 uint64_t flags) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  int my_pe_in_team = team_obj->my_pe;
  int root_pe_world = team_obj->get_pe_in_world(pe_root);

  // Non-root PEs fetch tile from root using GET
  if (my_pe_in_team != pe_root) {
    int result = tile_get(dst_data, src_data, dst_strides, src_strides, start_coord,
                          boundary, ndim, element_size, root_pe_world, flags);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }
  }
  // Note: Root PE's data is already in src, no need to copy to dst unless src != dst

  // Synchronize to ensure all PEs complete before root can modify buffer
  sync(team);

  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_broadcast_wave(rocshmem_team_t team,
                                                      void* dst_data,
                                                      const void* src_data,
                                                      const size_t* dst_strides,
                                                      const size_t* src_strides,
                                                      const size_t* start_coord,
                                                      const size_t* boundary,
                                                      int ndim,
                                                      size_t element_size,
                                                      int pe_root,
                                                      uint64_t flags) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  int my_pe_in_team = team_obj->my_pe;
  int root_pe_world = team_obj->get_pe_in_world(pe_root);

  // Non-root PEs fetch tile from root using GET (wave-collective)
  if (my_pe_in_team != pe_root) {
    int result = tile_get_wave(dst_data, src_data, dst_strides, src_strides, start_coord,
                                boundary, ndim, element_size, root_pe_world, flags);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }
  }

  // Synchronize to ensure all PEs complete before root can modify buffer
  sync_wave(team);

  return ROCSHMEM_SUCCESS;
}

__device__ inline int IPCContext::tile_broadcast_wg(rocshmem_team_t team,
                                                    void* dst_data,
                                                    const void* src_data,
                                                    const size_t* dst_strides,
                                                    const size_t* src_strides,
                                                    const size_t* start_coord,
                                                    const size_t* boundary,
                                                    int ndim,
                                                    size_t element_size,
                                                    int pe_root,
                                                    uint64_t flags) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  int my_pe_in_team = team_obj->my_pe;
  int root_pe_world = team_obj->get_pe_in_world(pe_root);

  // Non-root PEs fetch tile from root using GET (workgroup-collective)
  if (my_pe_in_team != pe_root) {
    int result = tile_get_wg(dst_data, src_data, dst_strides, src_strides, start_coord,
                              boundary, ndim, element_size, root_pe_world, flags);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }
  }

  // Synchronize to ensure all PEs complete before root can modify buffer
  sync_wg(team);

  return ROCSHMEM_SUCCESS;
}

// Returns 0 if any dimension has boundary < start_coord (underflow guard) or
// if the product overflows size_t (overflow guard).
__device__ inline size_t ipc_tile_num_elements(const size_t* start_coord,
                                               const size_t* boundary,
                                               int ndim) {
  size_t total = 1;
  for (int dim = 0; dim < ndim; dim++) {
    if (boundary[dim] < start_coord[dim]) {
      return 0;
    }
    const size_t extent = boundary[dim] - start_coord[dim];
    if (extent != 0 && total > SIZE_MAX / extent) {
      return 0;
    }
    total *= extent;
  }
  return total;
}

__device__ inline size_t ipc_tile_dst_offset(size_t flat_idx,
                                             const size_t* dst_strides,
                                             const size_t* start_coord,
                                             const size_t* boundary,
                                             int ndim) {
  size_t offset = 0;
  for (int dim = ndim - 1; dim >= 0; dim--) {
    const size_t extent = boundary[dim] - start_coord[dim];
    const size_t coord = flat_idx % extent;
    flat_idx /= extent;
    offset += (start_coord[dim] + coord) * dst_strides[dim];
  }
  return offset;
}

__device__ inline size_t ipc_tile_src_offset(size_t flat_idx,
                                             const size_t* src_strides,
                                             const size_t* start_coord,
                                             const size_t* boundary,
                                             int ndim) {
  size_t offset = 0;
  for (int dim = ndim - 1; dim >= 0; dim--) {
    const size_t extent = boundary[dim] - start_coord[dim];
    const size_t coord = flat_idx % extent;
    flat_idx /= extent;
    offset += (start_coord[dim] + coord) * src_strides[dim];
  }
  return offset;
}

__device__ inline bool ipc_tile_is_contiguous(const size_t* strides,
                                              const size_t* start_coord,
                                              const size_t* boundary,
                                              int ndim) {
  size_t expected_stride = 1;
  for (int dim = ndim - 1; dim >= 0; dim--) {
    if (strides[dim] != expected_stride) {
      return false;
    }
    expected_stride *= boundary[dim] - start_coord[dim];
  }
  return true;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int IPCContext::tile_reduce_typed_impl(
    rocshmem_team_t team, const void* src_data, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root,
    size_t segment_start, size_t segment_elems, size_t segment_capacity,
    int worker_id, int worker_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  const int team_size = team_obj->num_pes;
  const int my_pe_in_team = team_obj->my_pe;
  const int root_pe_world = team_obj->get_pe_in_world(root);

  if (root < 0 || root >= team_size || ndim <= 0) {
    LOGD_WARN("Invalid tile reduce arguments for IPC backend");
    return ROCSHMEM_ERROR;
  }

  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  T *my_pWrk = &pWrk[my_pe_in_team * segment_capacity];

  const bool src_contiguous =
      ipc_tile_is_contiguous(src_strides, start_coord, boundary, ndim);
  const size_t first_src_offset = ipc_tile_src_offset(
      segment_start, src_strides, start_coord, boundary, ndim);
  const char *src_base = static_cast<const char *>(src_data) +
                         first_src_offset * sizeof(T);

  if (src_contiguous && my_pe_in_team != root) {
    if (worker_count == 1) {
      internal_putmem(my_pWrk, src_base, segment_elems * sizeof(T),
                      root_pe_world);
    } else if (worker_count == WF_SIZE) {
      internal_putmem_wave(my_pWrk, src_base, segment_elems * sizeof(T),
                           root_pe_world);
    } else {
      internal_putmem_wg(my_pWrk, src_base, segment_elems * sizeof(T),
                         root_pe_world);
    }
  } else if (src_contiguous && my_pe_in_team == root) {
    const T *src_typed = reinterpret_cast<const T *>(src_base);
    if (worker_count == 1) {
      __builtin_memcpy(my_pWrk, src_typed, segment_elems * sizeof(T));
    } else if (worker_count == WF_SIZE) {
      for (size_t elem = worker_id; elem < segment_elems; elem += worker_count)
        my_pWrk[elem] = src_typed[elem];
    } else {
      for (size_t elem = worker_id; elem < segment_elems; elem += worker_count)
        my_pWrk[elem] = src_typed[elem];
    }
  } else {
    for (size_t elem = worker_id; elem < segment_elems; elem += worker_count) {
      const size_t tile_elem = segment_start + elem;
      const size_t src_offset =
          ipc_tile_src_offset(tile_elem, src_strides, start_coord, boundary, ndim);
      const T src_value =
          *reinterpret_cast<const T *>(static_cast<const char *>(src_data) +
                                       src_offset * sizeof(T));

      if (my_pe_in_team == root) {
        my_pWrk[elem] = src_value;
      } else {
        internal_putmem(&my_pWrk[elem], &src_value, sizeof(T), root_pe_world);
      }
    }
  }

  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline void ipc_tile_reduce_root_compute(
    rocshmem_team_t team, void* dst_data, const size_t* dst_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root,
    size_t segment_start, size_t segment_elems, size_t segment_capacity,
    int worker_id, int worker_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  const int team_size = team_obj->num_pes;
  if (team_obj->my_pe != root) {
    return;
  }

  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);

  for (size_t elem = worker_id; elem < segment_elems; elem += worker_count) {
    T reduced_value = pWrk[elem];
    for (int src_pe_in_team = 1; src_pe_in_team < team_size; src_pe_in_team++) {
      T src_value = pWrk[src_pe_in_team * segment_capacity + elem];
      OpWrap<Op>::Calc(&src_value, &reduced_value, 0);
    }

    const size_t tile_elem = segment_start + elem;
    const size_t dst_offset =
        ipc_tile_dst_offset(tile_elem, dst_strides, start_coord, boundary, ndim);
    char *dst_base = static_cast<char *>(dst_data);
    *reinterpret_cast<T *>(dst_base + dst_offset * sizeof(T)) = reduced_value;
  }
}

__device__ inline void ipc_tile_reduce_reset_psync(rocshmem_team_t team,
                                                   int root,
                                                   int worker_id,
                                                   int worker_count) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  if (team_obj->my_pe != root) {
    return;
  }

  long *pSync = team_obj->reduce_pSync;
  for (int i = worker_id; i < team_obj->num_pes; i += worker_count) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int IPCContext::tile_reduce_typed(
    rocshmem_team_t team, void* dst_data, const void* src_data,
    const size_t* dst_strides, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  long *pSync = team_obj->reduce_pSync;
  const int root_pe_world = team_obj->get_pe_in_world(root);
  const size_t tile_elements =
      ipc_tile_num_elements(start_coord, boundary, ndim);
  const size_t pwrk_capacity =
      (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / sizeof(T);
  const size_t segment_capacity = pwrk_capacity / team_obj->num_pes;

  if (tile_elements == 0) {
    LOGD_WARN("Tile reduce: invalid coordinate range (underflow or overflow)");
    return ROCSHMEM_ERROR;
  }

  if (segment_capacity == 0) {
    LOGD_WARN("Tile reduce type exceeds IPC pWrk capacity");
    return ROCSHMEM_ERROR;
  }

  if (team_obj->num_pes > static_cast<int>(ROCSHMEM_REDUCE_SYNC_SIZE)) {
    LOGD_WARN("Tile reduce team size exceeds IPC reduce_pSync capacity");
    return ROCSHMEM_ERROR;
  }

  long flag_val = 1;
  for (size_t segment_start = 0; segment_start < tile_elements;
       segment_start += segment_capacity, flag_val++) {
    const size_t segment_elems =
        min(segment_capacity, tile_elements - segment_start);
    int result = tile_reduce_typed_impl<T, Op>(
        team, src_data, src_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, 0, 1);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }

    if (team_obj->my_pe == root) {
      pSync[team_obj->my_pe] = flag_val;
      for (int i = 0; i < team_obj->num_pes; i++) {
        wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
      }
      threadfence_system();
    } else {
      fence(root_pe_world);
      internal_putmem(&pSync[team_obj->my_pe], &flag_val, sizeof(flag_val),
                      root_pe_world);
    }

    ipc_tile_reduce_root_compute<T, Op>(
        team, dst_data, dst_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, 0, 1);
    sync(team);
  }

  ipc_tile_reduce_reset_psync(team, root, 0, 1);
  sync(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int IPCContext::tile_reduce_typed_wave(
    rocshmem_team_t team, void* dst_data, const void* src_data,
    const size_t* dst_strides, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root) {
  const int wave_id = get_flat_block_id() % WF_SIZE;
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  long *pSync = team_obj->reduce_pSync;
  const int root_pe_world = team_obj->get_pe_in_world(root);
  const size_t tile_elements =
      ipc_tile_num_elements(start_coord, boundary, ndim);
  const size_t pwrk_capacity =
      (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / sizeof(T);
  const size_t segment_capacity = pwrk_capacity / team_obj->num_pes;

  if (tile_elements == 0) {
    LOGD_WARN("Tile reduce: invalid coordinate range (underflow or overflow)");
    return ROCSHMEM_ERROR;
  }

  if (segment_capacity == 0) {
    LOGD_WARN("Tile reduce type exceeds IPC pWrk capacity");
    return ROCSHMEM_ERROR;
  }

  if (team_obj->num_pes > static_cast<int>(ROCSHMEM_REDUCE_SYNC_SIZE)) {
    LOGD_WARN("Tile reduce team size exceeds IPC reduce_pSync capacity");
    return ROCSHMEM_ERROR;
  }

  long flag_val = 1;
  for (size_t segment_start = 0; segment_start < tile_elements;
       segment_start += segment_capacity, flag_val++) {
    const size_t segment_elems =
        min(segment_capacity, tile_elements - segment_start);
    int result = tile_reduce_typed_impl<T, Op>(
        team, src_data, src_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, wave_id, WF_SIZE);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }

    __builtin_amdgcn_wave_barrier();
    if (is_thread_zero_in_wave()) {
      if (team_obj->my_pe == root) {
        pSync[team_obj->my_pe] = flag_val;
        for (int i = 0; i < team_obj->num_pes; i++) {
          wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
        }
        threadfence_system();
      } else {
        fence(root_pe_world);
        internal_putmem(&pSync[team_obj->my_pe], &flag_val, sizeof(flag_val),
                        root_pe_world);
      }
    }

    __builtin_amdgcn_wave_barrier();
    ipc_tile_reduce_root_compute<T, Op>(
        team, dst_data, dst_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, wave_id, WF_SIZE);
    sync_wave(team);
  }

  ipc_tile_reduce_reset_psync(team, root, wave_id, WF_SIZE);
  sync_wave(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int IPCContext::tile_reduce_typed_wg(
    rocshmem_team_t team, void* dst_data, const void* src_data,
    const size_t* dst_strides, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root) {
  const int thread_id = get_flat_block_id();
  const int block_size = get_flat_block_size();
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);
  long *pSync = team_obj->reduce_pSync;
  const int root_pe_world = team_obj->get_pe_in_world(root);
  const size_t tile_elements =
      ipc_tile_num_elements(start_coord, boundary, ndim);
  const size_t pwrk_capacity =
      (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / sizeof(T);
  const size_t segment_capacity = pwrk_capacity / team_obj->num_pes;

  if (tile_elements == 0) {
    LOGD_WARN("Tile reduce: invalid coordinate range (underflow or overflow)");
    return ROCSHMEM_ERROR;
  }

  if (segment_capacity == 0) {
    LOGD_WARN("Tile reduce type exceeds IPC pWrk capacity");
    return ROCSHMEM_ERROR;
  }

  if (team_obj->num_pes > static_cast<int>(ROCSHMEM_REDUCE_SYNC_SIZE)) {
    LOGD_WARN("Tile reduce team size exceeds IPC reduce_pSync capacity");
    return ROCSHMEM_ERROR;
  }

  long flag_val = 1;
  for (size_t segment_start = 0; segment_start < tile_elements;
       segment_start += segment_capacity, flag_val++) {
    const size_t segment_elems =
        min(segment_capacity, tile_elements - segment_start);
    int result = tile_reduce_typed_impl<T, Op>(
        team, src_data, src_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, thread_id, block_size);
    if (result != ROCSHMEM_SUCCESS) {
      return result;
    }

    __syncthreads();
    if (is_thread_zero_in_block()) {
      if (team_obj->my_pe == root) {
        pSync[team_obj->my_pe] = flag_val;
        for (int i = 0; i < team_obj->num_pes; i++) {
          wait_until(&pSync[i], ROCSHMEM_CMP_EQ, flag_val);
        }
        threadfence_system();
      } else {
        fence(root_pe_world);
        internal_putmem(&pSync[team_obj->my_pe], &flag_val, sizeof(flag_val),
                        root_pe_world);
      }
    }

    __syncthreads();
    ipc_tile_reduce_root_compute<T, Op>(
        team, dst_data, dst_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, thread_id, block_size);
    sync_wg(team);
  }

  ipc_tile_reduce_reset_psync(team, root, thread_id, block_size);
  sync_wg(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int ipc_tile_reduce_typed(IPCContext* ctx,
                                            rocshmem_team_t team,
                                            void* dst_data,
                                            const void* src_data,
                                            const size_t* dst_strides,
                                            const size_t* src_strides,
                                            const size_t* start_coord,
                                            const size_t* boundary,
                                            int ndim,
                                            int root) {
  return ctx->tile_reduce_typed<T, Op>(
      team, dst_data, src_data, dst_strides, src_strides, start_coord,
      boundary, ndim, root);
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int ipc_tile_reduce_typed_wave(IPCContext* ctx,
                                                 rocshmem_team_t team,
                                                 void* dst_data,
                                                 const void* src_data,
                                                 const size_t* dst_strides,
                                                 const size_t* src_strides,
                                                 const size_t* start_coord,
                                                 const size_t* boundary,
                                                 int ndim,
                                                 int root) {
  return ctx->tile_reduce_typed_wave<T, Op>(
      team, dst_data, src_data, dst_strides, src_strides, start_coord,
      boundary, ndim, root);
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int ipc_tile_reduce_typed_wg(IPCContext* ctx,
                                               rocshmem_team_t team,
                                               void* dst_data,
                                               const void* src_data,
                                               const size_t* dst_strides,
                                               const size_t* src_strides,
                                               const size_t* start_coord,
                                               const size_t* boundary,
                                               int ndim,
                                               int root) {
  return ctx->tile_reduce_typed_wg<T, Op>(
      team, dst_data, src_data, dst_strides, src_strides, start_coord,
      boundary, ndim, root);
}

#define IPC_TILE_REDUCE_DISPATCH_CASES(TYPED_FN)                              \
  case ROCSHMEM_TILE_ELEMENT_INT8:                                            \
    return TYPED_FN<signed char, Op>(ctx, team, dst_data, src_data,            \
                                     dst_strides, src_strides, start_coord,    \
                                     boundary, ndim, root);                   \
  case ROCSHMEM_TILE_ELEMENT_UINT8:                                           \
    return TYPED_FN<unsigned char, Op>(ctx, team, dst_data, src_data,          \
                                       dst_strides, src_strides, start_coord,  \
                                       boundary, ndim, root);                 \
  case ROCSHMEM_TILE_ELEMENT_INT16:                                           \
  case ROCSHMEM_TILE_ELEMENT_SHORT:                                           \
    return TYPED_FN<short, Op>(ctx, team, dst_data, src_data, dst_strides,     \
                               src_strides, start_coord, boundary, ndim,       \
                               root);                                         \
  case ROCSHMEM_TILE_ELEMENT_UINT16:                                          \
  case ROCSHMEM_TILE_ELEMENT_USHORT:                                          \
    return TYPED_FN<unsigned short, Op>(                                      \
        ctx, team, dst_data, src_data, dst_strides, src_strides, start_coord,  \
        boundary, ndim, root);                                                \
  case ROCSHMEM_TILE_ELEMENT_INT32:                                           \
  case ROCSHMEM_TILE_ELEMENT_INT:                                             \
    return TYPED_FN<int, Op>(ctx, team, dst_data, src_data, dst_strides,       \
                             src_strides, start_coord, boundary, ndim, root);  \
  case ROCSHMEM_TILE_ELEMENT_UINT32:                                          \
  case ROCSHMEM_TILE_ELEMENT_UINT:                                            \
    return TYPED_FN<unsigned int, Op>(                                        \
        ctx, team, dst_data, src_data, dst_strides, src_strides, start_coord,  \
        boundary, ndim, root);                                                \
  case ROCSHMEM_TILE_ELEMENT_LONG:                                            \
    return TYPED_FN<long, Op>(ctx, team, dst_data, src_data, dst_strides,      \
                              src_strides, start_coord, boundary, ndim,        \
                              root);                                          \
  case ROCSHMEM_TILE_ELEMENT_ULONG:                                           \
    return TYPED_FN<unsigned long, Op>(                                       \
        ctx, team, dst_data, src_data, dst_strides, src_strides, start_coord,  \
        boundary, ndim, root);                                                \
  case ROCSHMEM_TILE_ELEMENT_INT64:                                           \
  case ROCSHMEM_TILE_ELEMENT_LONGLONG:                                        \
    return TYPED_FN<long long, Op>(ctx, team, dst_data, src_data, dst_strides, \
                                   src_strides, start_coord, boundary, ndim,   \
                                   root);                                     \
  case ROCSHMEM_TILE_ELEMENT_UINT64:                                          \
  case ROCSHMEM_TILE_ELEMENT_ULONGLONG:                                       \
    return TYPED_FN<unsigned long long, Op>(                                  \
        ctx, team, dst_data, src_data, dst_strides, src_strides, start_coord,  \
        boundary, ndim, root);                                                \
  case ROCSHMEM_TILE_ELEMENT_FLOAT:                                           \
    return TYPED_FN<float, Op>(ctx, team, dst_data, src_data, dst_strides,     \
                               src_strides, start_coord, boundary, ndim,       \
                               root);                                         \
  case ROCSHMEM_TILE_ELEMENT_DOUBLE:                                          \
    return TYPED_FN<double, Op>(ctx, team, dst_data, src_data, dst_strides,    \
                                src_strides, start_coord, boundary, ndim, root)

#define IPC_TILE_REDUCE_DISPATCH_DEF(DISPATCH_FN, TYPED_FN)                   \
  template <ROCSHMEM_OP Op>                                                   \
  __device__ inline int DISPATCH_FN(                                          \
      IPCContext* ctx, rocshmem_team_t team, void* dst_data,                  \
      const void* src_data, const size_t* dst_strides,                        \
      const size_t* src_strides, const size_t* start_coord,                   \
      const size_t* boundary, int ndim,                                       \
      [[maybe_unused]] size_t element_size, int root, uint64_t flags) {       \
    const auto element_type = static_cast<ROCSHMEM_TILE_ELEMENT_TYPE>(        \
        (flags & ROCSHMEM_TILE_ELEMENT_TYPE_MASK) >>                          \
        ROCSHMEM_TILE_ELEMENT_TYPE_SHIFT);                                    \
                                                                               \
    switch (element_type) {                                                   \
      IPC_TILE_REDUCE_DISPATCH_CASES(TYPED_FN);                               \
      default:                                                                \
        LOGD_WARN("Tile reduce element type not specified for IPC backend");   \
        return ROCSHMEM_ERROR;                                                \
    }                                                                         \
  }

IPC_TILE_REDUCE_DISPATCH_DEF(ipc_tile_reduce_dispatch, ipc_tile_reduce_typed)
IPC_TILE_REDUCE_DISPATCH_DEF(ipc_tile_reduce_wave_dispatch,
                             ipc_tile_reduce_typed_wave)
IPC_TILE_REDUCE_DISPATCH_DEF(ipc_tile_reduce_wg_dispatch,
                             ipc_tile_reduce_typed_wg)

#undef IPC_TILE_REDUCE_DISPATCH_DEF
#undef IPC_TILE_REDUCE_DISPATCH_CASES

// SUM Reductions - Type-erased implementations
__device__ inline int IPCContext::tile_sum_reduce(rocshmem_team_t team,
                                                  void* dst_data,
                                                  const void* src_data,
                                                  const size_t* dst_strides,
                                                  const size_t* src_strides,
                                                  const size_t* start_coord,
                                                  const size_t* boundary,
                                                  int ndim,
                                                  size_t element_size,
                                                  int root,
                                                  uint64_t flags) {
  return ipc_tile_reduce_dispatch<ROCSHMEM_SUM>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int IPCContext::tile_sum_reduce_wave(rocshmem_team_t team,
                                                       void* dst_data,
                                                       const void* src_data,
                                                       const size_t* dst_strides,
                                                       const size_t* src_strides,
                                                       const size_t* start_coord,
                                                       const size_t* boundary,
                                                       int ndim,
                                                       size_t element_size,
                                                       int root,
                                                       uint64_t flags) {
  return ipc_tile_reduce_wave_dispatch<ROCSHMEM_SUM>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int IPCContext::tile_sum_reduce_wg(rocshmem_team_t team,
                                                     void* dst_data,
                                                     const void* src_data,
                                                     const size_t* dst_strides,
                                                     const size_t* src_strides,
                                                     const size_t* start_coord,
                                                     const size_t* boundary,
                                                     int ndim,
                                                     size_t element_size,
                                                     int root,
                                                     uint64_t flags) {
  return ipc_tile_reduce_wg_dispatch<ROCSHMEM_SUM>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

// MAX Reductions - Type-erased interface
__device__ inline int IPCContext::tile_max_reduce(rocshmem_team_t team,
                                                   void* dst_data,
                                                   const void* src_data,
                                                   const size_t* dst_strides,
                                                   const size_t* src_strides,
                                                   const size_t* start_coord,
                                                   const size_t* boundary,
                                                   int ndim,
                                                   size_t element_size,
                                                   int root,
                                                   uint64_t flags) {
  return ipc_tile_reduce_dispatch<ROCSHMEM_MAX>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int IPCContext::tile_max_reduce_wave(rocshmem_team_t team,
                                                        void* dst_data,
                                                        const void* src_data,
                                                        const size_t* dst_strides,
                                                        const size_t* src_strides,
                                                        const size_t* start_coord,
                                                        const size_t* boundary,
                                                        int ndim,
                                                        size_t element_size,
                                                        int root,
                                                        uint64_t flags) {
  return ipc_tile_reduce_wave_dispatch<ROCSHMEM_MAX>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int IPCContext::tile_max_reduce_wg(rocshmem_team_t team,
                                                      void* dst_data,
                                                      const void* src_data,
                                                      const size_t* dst_strides,
                                                      const size_t* src_strides,
                                                      const size_t* start_coord,
                                                      const size_t* boundary,
                                                      int ndim,
                                                      size_t element_size,
                                                      int root,
                                                      uint64_t flags) {
  return ipc_tile_reduce_wg_dispatch<ROCSHMEM_MAX>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

// MIN Reductions - Type-erased interface
__device__ inline int IPCContext::tile_min_reduce(rocshmem_team_t team,
                                                   void* dst_data,
                                                   const void* src_data,
                                                   const size_t* dst_strides,
                                                   const size_t* src_strides,
                                                   const size_t* start_coord,
                                                   const size_t* boundary,
                                                   int ndim,
                                                   size_t element_size,
                                                   int root,
                                                   uint64_t flags) {
  return ipc_tile_reduce_dispatch<ROCSHMEM_MIN>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int IPCContext::tile_min_reduce_wave(rocshmem_team_t team,
                                                        void* dst_data,
                                                        const void* src_data,
                                                        const size_t* dst_strides,
                                                        const size_t* src_strides,
                                                        const size_t* start_coord,
                                                        const size_t* boundary,
                                                        int ndim,
                                                        size_t element_size,
                                                        int root,
                                                        uint64_t flags) {
  return ipc_tile_reduce_wave_dispatch<ROCSHMEM_MIN>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int IPCContext::tile_min_reduce_wg(rocshmem_team_t team,
                                                      void* dst_data,
                                                      const void* src_data,
                                                      const size_t* dst_strides,
                                                      const size_t* src_strides,
                                                      const size_t* start_coord,
                                                      const size_t* boundary,
                                                      int ndim,
                                                      size_t element_size,
                                                      int root,
                                                      uint64_t flags) {
  return ipc_tile_reduce_wg_dispatch<ROCSHMEM_MIN>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_HPP_
