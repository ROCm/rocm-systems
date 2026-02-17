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

#ifndef LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "rocshmem/rocshmem.hpp"
#include "util.hpp"
#include "context_gda_device.hpp"
#include "gda_team.hpp"
#include "queue_pair.hpp"
#include "rocshmem_calc.hpp"
#include "backend_gda.hpp"

#include <hip/hip_runtime.h>

namespace rocshmem {

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <typename T>
__device__ void GDAContext::p(T *dest, T value, int pe) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    long L_offset{reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank]};
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset, reinterpret_cast<void *>(&value), sizeof(T));
    return;
  }
  putmem_nbi(dest, &value, sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put(T *dest, const T *source, size_t nelems, int pe) {
  putmem(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put_nbi(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ T GDAContext::g(const T *source, int pe) {
  T ret;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    const char *src_typed{reinterpret_cast<const char *>(source)};
    long L_offset{const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank]};
    ipcImpl_.ipcCopy(&ret, ipcImpl_.ipc_bases[local_pe] + L_offset, sizeof(T));
    return ret;
  }
  printf("rocshmem::gda:g not implemented\n");
  abort();
  //TODO the following is incorrect because ret is not ibv registered memory
  //getmem(&ret, source, sizeof(T), pe);
  return ret;
}

template <typename T>
__device__ void GDAContext::get(T *dest, const T *source, size_t nelems, int pe) {
  getmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void GDAContext::get_nbi(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

// Atomics
template <typename T>
__device__ void GDAContext::amo_add(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_add not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[pe].atomic_nofetch(base_heap[pe] + L_offset, value, 0, pe);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ void GDAContext::amo_set(void *dst, T value, int pe) {
  amo_swap(dst, value, pe);
}

template <typename T>
__device__ T GDAContext::amo_swap(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_set not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      /**
       * Guess that the remote memory is zero by setting condition to zero.
       * The compare-and-swap loop will execute at least twice if wrong.
       * It may run additional times if contention on memory location.
       */
      while ((ret_val = qps[pe].atomic_cas(base_heap[pe] + L_offset, value,
                         cond, pe)) != cond) {
        cond = ret_val;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::amo_fetch_and(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_fetch_and not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  T desired_val = cond & value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while ((ret_val = qps[pe].atomic_cas(base_heap[pe] + L_offset,
                         desired_val, cond, pe)) != cond) {
        cond = ret_val;
        desired_val = ret_val & value;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ void GDAContext::amo_and(void *dst, T value, int pe) {
  amo_fetch_and(dst, value, pe);
}

template <typename T>
__device__ T GDAContext::amo_fetch_or(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_fetch_or not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  T desired_val = cond | value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while ((ret_val = qps[pe].atomic_cas(base_heap[pe] + L_offset,
                         desired_val, cond, pe)) != cond) {
        cond = ret_val;
        desired_val = ret_val | value;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ void GDAContext::amo_or(void *dst, T value, int pe) {
  amo_fetch_or(dst, value, pe);
}

template <typename T>
__device__ T GDAContext::amo_fetch_xor(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_fetch_xor not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  T desired_val = cond ^ value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while ((ret_val = qps[pe].atomic_cas(base_heap[pe] + L_offset,
                         desired_val, cond, pe)) != cond) {
        cond = ret_val;
        desired_val = ret_val ^ value;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ void GDAContext::amo_xor(void *dst, T value, int pe) {
  amo_fetch_xor(dst, value, pe);
}

template <typename T>
__device__ void GDAContext::amo_cas(void *dst, T value, T cond, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_cas not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[pe].atomic_cas_nofetch(base_heap[pe] + L_offset, value, cond, pe);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ T GDAContext::amo_fetch_add(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_fadd not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  T ret_val = 0;
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val =  qps[pe].atomic_fetch(base_heap[pe] + L_offset, value, 0, pe);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  if constexpr (sizeof(T) != 8) { printf("rocshmem::gda:amo_fcas not implemented for non-64bit types.\n"); abort(); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val = qps[pe].atomic_cas(base_heap[pe] + L_offset, value, cond, pe);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

// Collectives TODO: loosely adapted from IPC, needs review
template <typename T, ROCSHMEM_OP Op>
__device__ void gda_compute_reduce(T *src, T *dst, int size, int wg_id, int wg_size) {
  for (int i = wg_id; i < size; i += wg_size) {
    OpWrap<Op>::Calc(src, dst, i);
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ void GDAContext::internal_direct_allreduce(
    T *dst, const T *src, int nelems, GDATeam *team_obj) {  // NOLINT(runtime/int)

  int stride = team_obj->tinfo_wrt_world->stride;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);

  int finish = PE_start + stride * PE_size;
  int pe = my_pe;

  int wg_id = get_flat_block_id();
  int wg_size = get_flat_block_size();
  int64_t flag_val = 1;

  for (int i = wg_id; i < nelems; i += wg_size) {
    dst[i] = src[i];
  }
  __syncthreads();

  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      putmem_wg(&pWrk[pe * nelems], reinterpret_cast<const void *>(src),
                nelems * sizeof(T), i);

      if (is_thread_zero_in_block()) {
        fence();
        putmem(&pSync[pe], &flag_val, sizeof(*pSync), i);
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
      gda_compute_reduce<T, Op>(ptr, dst, nelems, wg_id, wg_size);
      threadfence_system();
    }
  }

  __syncthreads();

  for (int i = wg_id; i < num_pes; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  threadfence_system();
  __syncthreads();
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
__device__ void GDAContext::internal_ring_allreduce(
    T *dst, const T *src, int nelems, GDATeam *team_obj,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size) {

  int stride = team_obj->tinfo_wrt_world->stride;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
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
    // Loop 2 in the algorithm above
    for (int iter = 0; iter < PE_size - 1; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      off_recv = (((my_pe_in_team - iter + 2 * PE_size) % PE_size) * chunk_size);

      putmem_wg(reinterpret_cast<void *>(&pWrk[off_send]),
                reinterpret_cast<void *>(&dst[off_send + off_seg]),
                chunk_size * sizeof(T), send_pe);

      if (is_thread_zero_in_block()) {
        fence();

        wait_val = seg + 100;
        putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
      gda_compute_reduce<T, Op>(&pWrk[off_recv], &dst[off_seg + off_recv],
                                chunk_size, wg_id, wg_size);
    }

    // Loop 2 in the example above
    for (int iter = PE_size - 1; iter < 2 * PE_size - 2; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      putmem_nbi_wg(reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    chunk_size * sizeof(T), send_pe);

      if (is_thread_zero_in_block()) {
        fence();
        wait_val = seg + 100;
        putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
    }
  }
  __syncthreads();

  for (int i = wg_id; i < 2 * num_pes - 2; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::reduce(rocshmem_team_t team, T *dest,
                                  const T *source, int nreduce) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int PE_size = team_obj->tinfo_wrt_world->size;

  size_t direct_pWrk = PE_size * nreduce;
  size_t direct_pSync = PE_size;
  size_t ring_pSync = 2 * PE_size;
  size_t provided_pWrk = max(nreduce / 2 + 1, ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
    internal_direct_allreduce<T, Op>(dest, source, nreduce, team_obj);
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
      // recalculate chunk_size
      chunk_size = seg_size / PE_size;

      if (n_seg > 0) {
        internal_ring_allreduce<T, Op>(dest, source, nreduce, team_obj, n_seg,
                                       seg_size, chunk_size);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;

        if (p_chunk > 0) {
          internal_ring_allreduce<T, Op>(p_dst, p_src, (p_chunk * PE_size),
                                         team_obj, 1, (p_chunk * PE_size),
                                         p_chunk);
        }

        if ((p_chunk * PE_size) < p_count) {
          // Final elements need to use direct_allreduce
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);

          internal_direct_allreduce<T, Op>(p_dst, p_src2, p_count, team_obj);
        }
      }
    } else {
      GPU_DPRINTF("Unsupported reduction size for GDA conduit.\n");
      return ROCSHMEM_ERROR;
    }
  }
  barrier_wg(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void GDAContext::internal_put_broadcast(
    T *dst, const T *src, int nelems, int pe_root, int pe_start,
    int stride, int pe_size) {  // NOLINT(runtime/int)
  if (my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (i != my_pe) {
        put_nbi_wg(dst, src, nelems, i);
      }
    }
  }
}

template <typename T>
__device__ void GDAContext::internal_get_broadcast(
  T *dst, const T *src, int nelems, int pe_root) {  // NOLINT(runtime/int)
  if (my_pe != pe_root) {
    get_wg(dst, src, nelems, pe_root);
  }
}

template <typename T>
__device__ void GDAContext::broadcast(rocshmem_team_t team, T *dst,
                                      const T *src, int nelems, int pe_root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int stride = team_obj->tinfo_wrt_world->stride;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->tinfo_wrt_world->size;
  long *p_sync = team_obj->bcast_pSync;

  // Passed pe_root is relative to team, convert to world root
  int pe_root_world = team_obj->get_pe_in_world(pe_root);
  internal_broadcast<T>(dst, src, nelems, pe_root_world, pe_start, stride,
               pe_size, p_sync);
}

template <typename T>
__device__ void GDAContext::internal_broadcast(T *dst, const T *src, int nelems,
                                      int pe_root, int pe_start,
                                      int stride, int pe_size,
                                      long *p_sync) {  // NOLINT(runtime/int)
  if (num_pes < 4) { //TODO: optimized for IPC
    internal_put_broadcast(dst, src, nelems, pe_root, pe_start, stride,
                           pe_size);
  } else {
    internal_get_broadcast(dst, src, nelems, pe_root);
  }

  // Synchronize on completion of broadcast
  internal_sync_wg(my_pe, pe_start, stride, pe_size, p_sync);
}

template <typename T>
__device__ void GDAContext::alltoall(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  if (gda_provider_ == GDAProvider::BNXT ||
      gda_provider_ == GDAProvider::IONIC) {
    alltoall_linear_thread_puts(team, dst, src, nelems);
  } else {
    alltoall_linear(team, dst, src, nelems);
  }
}
template <typename T>

__device__ void GDAContext::alltoallv(rocshmem_team_t team,
                                      T *dest, const size_t dest_nelems[],
                                      const size_t dest_displs[],
                                      T *source, const size_t source_nelems[],
                                      const size_t source_displs[]) {
  if (gda_provider_ == GDAProvider::MLX5 ||
      gda_provider_ == GDAProvider::IONIC) {
    printf("rocshmem::gda:alltoallv not implemented\n");
    abort();
  }

  alltoallv_get(team,
                dest, dest_nelems, dest_displs,
                source, source_nelems, source_displs);
}

template <typename T>
__device__ void GDAContext::alltoallv_copy(rocshmem_team_t team,
                                           T *dest, const size_t dest_nelems[],
                                           const size_t dest_displs[],
                                           T *source, const size_t source_nelems[],
                                           const size_t source_displs[]) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t alltoall_pSync_offset = (team_obj->alltoall_sequence_number % 2) * pe_size;
  T *tmp_buf = reinterpret_cast<T*>(team_obj->pWrk);
  int tmp_buf_off = (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / (pe_size * sizeof(T));

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[my_pe];
    size_t nelems = source_nelems[dest_pe] * sizeof(T);
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);

    if (nelems != 0) {
      T* src = (T*)((char*)source + (source_displs[j] * sizeof(T)));
      T* dst = (T*)((char*)&tmp_buf[my_pe * tmp_buf_off] + base_heap_offset);
      qps[dest_pe].put_nbi_single(dst, src, nelems, false);
    }

    qps[dest_pe].atomic_nofetch_single(amo_dst, 1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    volatile long *vol_ivars = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(vol_ivars) != 1) { }

    pe_quiet_single(dest_pe);

    pSync[alltoall_pSync_offset + dest_pe] = ROCSHMEM_SYNC_VALUE;
  }

  // Copy out of staging buffer
  __syncthreads();

  for (int j = 0; j < pe_size; j++) {
    size_t nelems = dest_nelems[j] * sizeof(T);

    if (nelems != 0) {
      T* dst = (T*)((char*) dest + dest_displs[j] * sizeof(T));
      T* src = (T*)((char*) &tmp_buf[j * tmp_buf_off]);
      memcpy_wg(dst, src, nelems);
    }
  }

  __syncthreads();

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }
}

template <typename T>
__device__ void GDAContext::alltoallv_get(rocshmem_team_t team,
                                          T *dest, const size_t dest_nelems[],
                                          const size_t dest_displs[],
                                          T *source, const size_t source_nelems[],
                                          const size_t source_displs[]) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int pe_size       = team_obj->num_pes;
  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t a2a_sn   = team_obj->alltoall_sequence_number;
  uint64_t alltoall_pSync_offset = (a2a_sn % 2) * pe_size;
  uint64_t *tmp_buf = (uint64_t*)team_obj->pWrk;

  const uint64_t displs_mask = 0x0000'FFFF'FFFF'FFFF;
  const uint64_t seq_mask = 0xFFFF;
  const uint64_t seq_shift = 48;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  /* Put Ctrl Message */
  for (int j = tid; j < pe_size; j+= step_size) {
    uint64_t *src;
    uint64_t *dst;
    uint64_t seq_bits;
    uint64_t displ_bits;

    int dest_pe = team_obj->get_pe_in_world(j);
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[my_pe];

    /* Pack Ctrl Message * 16 bits seq | 48bit displ */
    seq_bits = (seq_mask & (a2a_sn + 1)) << seq_shift;
    displ_bits = (displs_mask & source_displs[dest_pe]);
    uint64_t ctrl_msg = seq_bits | displ_bits;

    /* Prepare Ctrl Message */
    src = (uint64_t*)&ctrl_msg;
    dst = (uint64_t*)((char*)&tmp_buf[my_pe] + base_heap_offset);

    qps[dest_pe].put_nbi_single(dst, src, sizeof(uint64_t), true);

    /* Wait for Ctrl Message */
    uint64_t ctrl_value;
    volatile uint64_t *vol_ctrl = &tmp_buf[dest_pe];

    do {
      ctrl_value = uncached_load(vol_ctrl);
      seq_bits = (ctrl_value >> seq_shift) & seq_mask;
      displ_bits = ctrl_value & displs_mask;
    } while (seq_bits != (a2a_sn + 1));

    /* Get data */
    size_t nelems = dest_nelems[dest_pe] * sizeof(T);
    src = (uint64_t*)((char*)source + (displ_bits * sizeof(T)) + base_heap_offset);
    dst = (uint64_t*)((char*)dest + (dest_displs[j] * sizeof(T)));

    qps[dest_pe].get_nbi_single(dst, src, nelems, true);

    /* Put Completion */
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);
    qps[dest_pe].atomic_nofetch_single(amo_dst, 1);

    volatile long *vol_ivars = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(vol_ivars) != 1) { }

    pe_quiet_single(dest_pe);

    pSync[alltoall_pSync_offset + dest_pe] = ROCSHMEM_SYNC_VALUE;
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

template <typename T>
__device__ void GDAContext::alltoall_linear(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  int wf_id = get_flat_block_id() / WF_SIZE;
  int wf_count = (int) ceil((double)get_flat_block_size() / (double)WF_SIZE);

  // Have each PE put their designated data to the other PEs
  for (int j = wf_id; j < pe_size; j+= wf_count) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi_wave(&dst[my_pe_in_team * nelems], &src[j * nelems], nelems, dest_pe);
  }

  for (int j = wf_id; j < pe_size; j+= wf_count) {
    int dest_pe = team_obj->get_pe_in_world(j);
    pe_quiet(dest_pe);
  }

  // wait until everyone has obtained their designated data
  internal_sync_wg(my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void GDAContext::alltoall_linear_thread_puts(rocshmem_team_t team, T *dst,
                                                        const T *src, int nelems) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t alltoall_pSync_offset = (team_obj->alltoall_sequence_number % 2) * pe_size;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[my_pe];
    qps[dest_pe].put_nbi_single(reinterpret_cast<char*>(&dst[my_pe_in_team * nelems]) + base_heap_offset,
                                &src[j * nelems], nelems * sizeof(T), false);
    qps[dest_pe].atomic_nofetch_single(reinterpret_cast<char *>(&pSync[alltoall_pSync_offset + my_pe_in_team]) + base_heap_offset,
                                       1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    volatile long *vol_ivars = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(vol_ivars) != 1) { }

    pe_quiet_single(dest_pe);

    pSync[alltoall_pSync_offset + dest_pe] = ROCSHMEM_SYNC_VALUE;
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

template <typename T>
__device__ void GDAContext::fcollect(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  fcollect_linear(team, dst, src, nelems);
}

template <typename T>
__device__ void GDAContext::fcollect_linear(rocshmem_team_t team, T *dst,
                                            const T *src, int nelems) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_start = team_obj->tinfo_wrt_world->pe_start;
  int pe_size = team_obj->num_pes;
  int stride = team_obj->tinfo_wrt_world->stride;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;

  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    put_nbi_wg(&dst[my_pe_in_team * nelems], src, nelems, dest_pe);
  }

  if (is_thread_zero_in_block()) {
    quiet();
  }
  // wait until everyone has obtained their designated data
  internal_sync_wg(my_pe, pe_start, stride, pe_size, pSync);
}

// Block/wave functions
template <typename T>
__device__ void GDAContext::put_wg(T *dest, const T *source, size_t nelems, int pe) {
  putmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

  template <typename T>
__device__ void GDAContext::put_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  putmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::get_wg(T *dest, const T *source, size_t nelems, int pe) {
  getmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::get_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::get_wave(T *dest, const T *source, size_t nelems, int pe) {
  getmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void GDAContext::get_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
  getmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

#define GDA_CONTEXT_PUT_SIGNAL_DEF(SUFFIX)                                                            \
  template <typename T>                                                                               \
  __device__ void GDAContext::put_signal##SUFFIX(T *dest, const T *source, size_t nelems,             \
                                                 uint64_t *sig_addr, uint64_t signal, int sig_op,     \
                                                 int pe) {                                            \
    putmem_signal##SUFFIX(dest, source, nelems * sizeof(T), sig_addr, signal, sig_op, pe);            \
  }                                                                                                   \
                                                                                                      \
  template <typename T>                                                                               \
  __device__ void GDAContext::put_signal_nbi##SUFFIX(T *dest, const T *source, size_t nelems,         \
                                                     uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                     int pe) {                                        \
    putmem_signal##SUFFIX(dest, source, nelems * sizeof(T), sig_addr, signal, sig_op, pe);            \
  }

GDA_CONTEXT_PUT_SIGNAL_DEF()
GDA_CONTEXT_PUT_SIGNAL_DEF(_wg)
GDA_CONTEXT_PUT_SIGNAL_DEF(_wave)

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_
