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
#include "constmem.hpp"
#include "log.hpp"
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
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    long L_offset{reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank]};
    ipcImpl_.ipcCopy<MemcpyKind::Put>(ipcImpl_.ipc_bases[local_pe] + L_offset, reinterpret_cast<void *>(&value), sizeof(T), local_pe);
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
  T ret{};
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    const char *src_typed{reinterpret_cast<const char *>(source)};
    long L_offset{const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank]};
    ipcImpl_.ipcCopy<MemcpyKind::Get>(&ret, ipcImpl_.ipc_bases[local_pe] + L_offset, sizeof(T), local_pe);
    return ret;
  }
  LOGD_ERROR_ABORT("gda::g not implemented");
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
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_add not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_nofetch(base_heap[pe] + L_offset, value, 0, wf_info);
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
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_set not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
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
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             base_heap[pe] + L_offset, value, cond, wf_info)) != cond) {
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
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_and not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  T desired_val = cond & value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             base_heap[pe] + L_offset, desired_val, cond, wf_info)) != cond) {
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
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_or not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  T desired_val = cond | value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             base_heap[pe] + L_offset, desired_val, cond, wf_info)) != cond) {
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
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fetch_xor not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  T cond = 0;
  T desired_val = cond ^ value;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             base_heap[pe] + L_offset, desired_val, cond, wf_info)) != cond) {
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
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_cas not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_cas_nofetch(base_heap[pe] + L_offset, value, cond, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ T GDAContext::amo_fetch_add(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fadd not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  T ret_val = 0;
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val =  qps[qp_index].atomic_fetch(base_heap[pe] + L_offset, value, 0, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fcas not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val = qps[qp_index].atomic_cas(base_heap[pe] + L_offset, value, cond, wf_info);
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
__device__ void GDAContext::internal_direct_allreduce(T *dst, const T *src,
    int nelems, GDATeam *team_obj, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)

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
        nelems * sizeof(T), i, i, wf_info);

      if (is_thread_zero_in_block()) {
        fence();
        internal_putmem(&pSync[pe], &flag_val, sizeof(*pSync), i, i, wf_info);
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

  for (int i = wg_id; i < constmem.num_pes; i += wg_size) {
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
__device__ void GDAContext::internal_ring_allreduce(T *dst, const T *src,
    int nelems, GDATeam *team_obj,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size, ActiveWFInfo &wf_info) {

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

      internal_putmem_wg(reinterpret_cast<void *>(&pWrk[off_send]),
        reinterpret_cast<void *>(&dst[off_send + off_seg]),
        chunk_size * sizeof(T), send_pe, send_pe, wf_info);

      if (is_thread_zero_in_block()) {
        fence();

        wait_val = seg + 100;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe,
          send_pe, wf_info);
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
      internal_putmem_nbi_wg(reinterpret_cast<void *>(&dst[off_send + off_seg]),
        reinterpret_cast<void *>(&dst[off_send + off_seg]),
        chunk_size * sizeof(T), send_pe, send_pe, wf_info);

      if (is_thread_zero_in_block()) {
        fence();
        wait_val = seg + 100;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe,
          send_pe, wf_info);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
    }
  }
  __syncthreads();

  for (int i = wg_id; i < 2 * constmem.num_pes - 2; i += wg_size) {
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

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
    internal_direct_allreduce<T, Op>(dest, source, nreduce, team_obj, wf_info);
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
          seg_size, chunk_size, wf_info);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;

        if (p_chunk > 0) {
          internal_ring_allreduce<T, Op>(p_dst, p_src, (p_chunk * PE_size),
            team_obj, 1, (p_chunk * PE_size), p_chunk, wf_info);
        }

        if ((p_chunk * PE_size) < p_count) {
          // Final elements need to use direct_allreduce
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);

          internal_direct_allreduce<T, Op>(p_dst, p_src2, p_count, team_obj, wf_info);
        }
      }
    } else {
      LOGD_WARN("Unsupported reduction size for GDA conduit.");
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
 * Only workgroup 0 (is_block_zero_in_grid) runs the reduction algorithm;
 * all workgroups participate in the per-chunk barrier_wg so the barrier
 * call counts match.  This prevents concurrent accumulation races when
 * multiple workgroups share the same team pSync/pWrk/dest buffers.
 */
template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::reduce_scatter_wg(rocshmem_team_t team, T *dest,
                                             const T *source, int nreduce) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int PE_size   = team_obj->tinfo_wrt_world->size;
  int PE_start  = team_obj->tinfo_wrt_world->pe_start;
  int stride    = team_obj->tinfo_wrt_world->stride;
  int team_rank = (my_pe - PE_start) / stride;

  long *pSync = team_obj->reduce_pSync;
  T    *pWrk  = reinterpret_cast<T *>(team_obj->pWrk);

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);

  int wg_id   = get_flat_block_id();
  int wg_size = get_flat_block_size();

  int pWrk_elems = (int)(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  int chunk_size = max(1, pWrk_elems / PE_size);
  int n_chunks   = (nreduce + chunk_size - 1) / chunk_size;
  int64_t flag_val = 1;
  int finish = PE_start + stride * PE_size;

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
                             count * sizeof(T), i, i, wf_info);
          if (is_thread_zero_in_block()) {
            fence();
            internal_putmem(&pSync[team_rank], &flag_val, sizeof(*pSync), i, i, wf_info);
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
          gda_compute_reduce<T, Op>(&pWrk[remote_rank * chunk_size],
                                    dest + offset, count, wg_id, wg_size);
          threadfence_system();
        }
      }
      __syncthreads();

      // Reset pSync before reuse.
      for (int j = wg_id; j < PE_size; j += wg_size) {
        pSync[j] = ROCSHMEM_SYNC_VALUE;
      }
      threadfence_system();
      __syncthreads();
      // Sync with workgroup 0 of other PEs
      barrier_wg(team);
    }
  }

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void GDAContext::internal_put_broadcast(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (constmem.my_pe != i)
        internal_putmem_nbi_wg(dst, src, nelems * sizeof(T), i, i, wf_info);
    }
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nelems * sizeof(T));
  }
}

template <typename T>
__device__ void GDAContext::internal_get_broadcast(T *dst, const T *src,
    int nelems, int pe_root, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    memcpy_wg<MemcpyKind::Put>(dst, const_cast<T *>(src), nelems * sizeof(T));
  } else {
    internal_getmem_wg(dst, src, nelems * sizeof(T), pe_root, pe_root, wf_info);
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
__device__ void GDAContext::internal_broadcast(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)
  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) { //TODO: optimized for IPC
    internal_put_broadcast(dst, src, nelems, pe_root, pe_start, stride,
      pe_size, wf_info);
  } else {
    internal_get_broadcast(dst, src, nelems, pe_root, wf_info);
  }

  // Synchronize on completion of broadcast
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

template <typename T>
__device__ void GDAContext::alltoall(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  alltoall_linear_thread_puts(team, dst, src, nelems);
}

template <typename T>
__device__ void GDAContext::alltoallv(rocshmem_team_t team,
                                      T *dest, const size_t dest_nelems[],
                                      const size_t dest_displs[],
                                      T *source, const size_t source_nelems[],
                                      const size_t source_displs[]) {
  if (constmem.alltoall_wg_algo == gda::ALLTOALLV_WG_ALGO_COPY) {
    alltoallv_copy(team,
                   dest, dest_nelems, dest_displs,
                   source, source_nelems, source_displs);
  } else {
    alltoallv_get(team,
                  dest, dest_nelems, dest_displs,
                  source, source_nelems, source_displs);
  }
}

template <typename T>
__device__ void GDAContext::alltoallv_copy(rocshmem_team_t team, T *dest,
    const size_t dest_nelems[], const size_t dest_displs[], T *source,
    const size_t source_nelems[], const size_t source_displs[]) {
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
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];
    size_t nelems = source_nelems[dest_pe] * sizeof(T);
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);

    if (nelems != 0) {
      T* src = (T*)((char*)source + (source_displs[j] * sizeof(T)));
      T* dst = (T*)((char*)&tmp_buf[constmem.my_pe * tmp_buf_off] + base_heap_offset);
      qps[dest_pe].put_nbi_single(dst, src, nelems, false);
    }

    qps[dest_pe].atomic_nofetch_single(amo_dst, 1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    long *sync_flags = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flags) != 1) { }

    qps[dest_pe].quiet_single();

    pSync[alltoall_pSync_offset + dest_pe] = ROCSHMEM_SYNC_VALUE;
  }

  // Copy out of staging buffer
  __syncthreads();

  for (int j = 0; j < pe_size; j++) {
    size_t nelems = dest_nelems[j] * sizeof(T);

    if (nelems != 0) {
      T* dst = (T*)((char*) dest + dest_displs[j] * sizeof(T));
      T* src = (T*)((char*) &tmp_buf[j * tmp_buf_off]);
      memcpy_wg<MemcpyKind::Put>(dst, src, nelems);
    }
  }

  __syncthreads();

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }
}

template <typename T>
__device__ void GDAContext::alltoallv_get(rocshmem_team_t team, T *dest,
    [[maybe_unused]] const size_t dest_nelems[], const size_t dest_displs[], T *source,
    [[maybe_unused]] const size_t source_nelems[], const size_t source_displs[]) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  int pe_size = team_obj->num_pes;
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
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];

    /* Pack Ctrl Message * 16 bits seq | 48bit displ */
    seq_bits = (seq_mask & (a2a_sn + 1)) << seq_shift;
    displ_bits = (displs_mask & source_displs[dest_pe]);
    uint64_t ctrl_msg = seq_bits | displ_bits;

    /* Prepare Ctrl Message */
    src = (uint64_t*)&ctrl_msg;
    dst = (uint64_t*)((char*)&tmp_buf[constmem.my_pe] + base_heap_offset);

    qps[dest_pe].put_nbi_single(dst, src, sizeof(uint64_t), true);

    /* Wait for Ctrl Message */
    uint64_t ctrl_value;
    uint64_t *vol_ctrl = &tmp_buf[dest_pe];

    do {
      ctrl_value = uncached_load(vol_ctrl);
      seq_bits = (ctrl_value >> seq_shift) & seq_mask;
      displ_bits = ctrl_value & displs_mask;
    } while (seq_bits != ((a2a_sn + 1) & seq_mask));

    /* Get data */
    size_t nelems = dest_nelems[dest_pe] * sizeof(T);
    src = (uint64_t*)((char*)source + (displ_bits * sizeof(T)) + base_heap_offset);
    dst = (uint64_t*)((char*)dest + (dest_displs[j] * sizeof(T)));

    qps[dest_pe].get_nbi_single(dst, src, nelems, true);

    /* Put Completion */
    char* amo_dst = ((char*)&pSync[alltoall_pSync_offset + my_pe_in_team] + base_heap_offset);
    qps[dest_pe].atomic_nofetch_single(amo_dst, 1);

    long *sync_flags = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flags) != 1) { }

    qps[dest_pe].quiet_single();

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

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  // Have each PE put their designated data to the other PEs
  for (int j = wf_id; j < pe_size; j+= wf_count) {
    int dest_pe = team_obj->get_pe_in_world(j);
    internal_putmem_nbi_wave(&dst[my_pe_in_team * nelems], &src[j * nelems],
      nelems * sizeof(T), dest_pe, dest_pe, wf_info);
  }

  for (int j = wf_id; j < pe_size; j+= wf_count) {
    int dest_pe = team_obj->get_pe_in_world(j);
    qps[dest_pe].quiet(wf_info);
  }

  // wait until everyone has obtained their designated data
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync, wf_info);
}

template <typename T>
__device__ void GDAContext::alltoall_linear_thread_puts(rocshmem_team_t team,
    T *dst, const T *src, int nelems) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int pe_size = team_obj->num_pes;
  long *pSync = team_obj->alltoall_pSync;
  int my_pe_in_team = team_obj->my_pe;
  uint64_t alltoall_pSync_offset = (team_obj->alltoall_sequence_number % 2) * pe_size;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  // Have each PE put their designated data to the other PEs
  for (int j = tid; j < pe_size; j += step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);
    uint64_t base_heap_offset = base_heap[dest_pe] - base_heap[constmem.my_pe];
    qps[dest_pe].put_nbi_single(
      reinterpret_cast<char*>(&dst[my_pe_in_team * nelems]) + base_heap_offset,
      &src[j * nelems], nelems * sizeof(T), false);
    qps[dest_pe].atomic_nofetch_single(
      reinterpret_cast<char *>(&pSync[alltoall_pSync_offset + my_pe_in_team]) +
      base_heap_offset, 1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    long *sync_flags = &pSync[alltoall_pSync_offset + dest_pe];
    while (uncached_load(sync_flags) != 1) { }

    qps[dest_pe].quiet_single();

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

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  // Have each PE put their designated data to the other PEs
  for (int j = 0; j < pe_size; j++) {
    int dest_pe = team_obj->get_pe_in_world(j);
    internal_putmem_nbi_wg(&dst[my_pe_in_team * nelems], src,
      nelems * sizeof(T), dest_pe, dest_pe, wf_info);
  }

  if (is_thread_zero_in_block()) {
    // Iterate through 0th qp of each PE
    for (int j = 0; j < pe_size; j++) {
      int dest_pe = team_obj->get_pe_in_world(j);
      qps[dest_pe].quiet(wf_info);
    }
  }
  // wait until everyone has obtained their designated data
  internal_sync_wg(constmem.my_pe, pe_start, stride, pe_size, pSync, wf_info);
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

// Internal functions used by collective and signal operations
template <typename T>
__device__ void GDAContext::internal_amo_add(void *dst, T value, int pe,
    int qp_index, ActiveWFInfo &wf_info) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_add not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_nofetch(base_heap[pe] + L_offset, value, 0, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ T GDAContext::internal_amo_fetch_add(void *dst, T value, int pe,
    int qp_index, ActiveWFInfo &wf_info) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fadd not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
  T ret_val = 0;
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val =  qps[qp_index].atomic_fetch(base_heap[pe] + L_offset, value, 0, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::internal_amo_swap(void *dst, T value, int pe,
    int qp_index, ActiveWFInfo &wf_info) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_set not implemented for non-64bit types"); }//TODO:support for non-uint64t
  uint64_t L_offset = reinterpret_cast<char *>(dst) - base_heap[constmem.my_pe];
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
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_cas(
             base_heap[pe] + L_offset, value, cond, wf_info)) != cond) {
        cond = ret_val;
      }
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

/******************************************************************************
 ****************************** INLINE FUNCTIONS ******************************
 *****************************************************************************/

/**
 * @brief Get the Queue Pair index for a given PE based on a atomic counter
 *        This ensures even distribution of requests across multiple QPs
 *        allocated per PE.
 * @param pe The target PE
 * @return The Queue Pair index
 *
 * Explanation of QP indexing scheme:
 *  num_qps_per_pe = 4
 *  num_pes        = 3
 *
 *  Layout of QPs per PE:
 *
 *             PE0          PE1          PE2
 *           ───────      ───────      ───────
 *  QP0  ─> [ QP0,0 ]    [ QP0,1 ]    [ QP0,2 ]
 *  QP1  ─> [ QP1,0 ]    [ QP1,1 ]    [ QP1,2 ]
 *  QP2  ─> [ QP2,0 ]    [ QP2,1 ]  **[ QP2,2 ]** <-- highlighted (3rd QP of PE2)
 *  QP3  ─> [ QP3,0 ]    [ QP3,1 ]    [ QP3,2 ]
 *
 *  Legend:
 *    - num_qps_per_pe = 4  →  Four Queue Pairs per PE
 *    - num_pes = 3         →  Three Processing Elements (PE0–PE2)
 *    - QP[i,j]             →  i-th QP of PE j
 *    - **[ QP2,2 ]**       →  The 3rd QP (QP index 2) of PE2
 */
__device__ __forceinline__ uint32_t GDAContext::get_qp_index(int pe,
    ActiveWFInfo wf_info) {

  uint32_t qp_index   {0};

  if(wf_info.pe_group_logical_lane_id == 0) {
    // Only the leader lane updates the counter (Does it require atomics?)
    // uint32_t local_qp_counter = __hip_atomic_fetch_add(&qp_counter[pe], 1,
    //                                        __ATOMIC_RELAXED,
    //                                        __HIP_MEMORY_SCOPE_AGENT);
    // local_qp_counter %= num_qps_per_pe;
    // qp_index = (local_qp_counter * num_pes) + pe;
    qp_index = (qp_counter[pe]++ % num_qps_per_pe) * constmem.num_pes + pe;
  }

  // Broadcast the qp_index value to other lanes in the wavefront
  // that are targeting the same PE
  qp_index = __shfl_sync(wf_info.pe_group_mask, qp_index, wf_info.pe_group_first_phys_lane_id);

  return qp_index;
}

/******************************************************************************
 **************** TILE API STUB IMPLEMENTATIONS (NOT IMPLEMENTED) *************
 *****************************************************************************/

// RMA PUT operations - Type-erased interface
__device__ inline int GDAContext::tile_put([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                    [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                    [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                    [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                    [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_put_wave([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                         [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                         [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                         [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                         [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_put_wg([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                       [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                       [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                       [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                       [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// RMA GET operations - Type-erased interface
__device__ inline int GDAContext::tile_get([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                    [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                    [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                    [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                    [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_get_wave([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                         [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                         [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                         [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                         [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_get_wg([[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                       [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                       [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                       [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                       [[maybe_unused]] int pe, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// Allgather operations - Type-erased interface
__device__ inline int GDAContext::tile_allgather([[maybe_unused]] rocshmem_team_t team,
                                          [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                          [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                          [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                          [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                          [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_allgather_wave([[maybe_unused]] rocshmem_team_t team,
                                               [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                               [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                               [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                               [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                               [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_allgather_wg([[maybe_unused]] rocshmem_team_t team,
                                             [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                             [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                             [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                             [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                             [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// Broadcast operations - Type-erased interface
__device__ inline int GDAContext::tile_broadcast([[maybe_unused]] rocshmem_team_t team,
                                          [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                          [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                          [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                          [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                          [[maybe_unused]] int pe_root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_broadcast_wave([[maybe_unused]] rocshmem_team_t team,
                                               [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                               [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                               [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                               [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                               [[maybe_unused]] int pe_root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_broadcast_wg([[maybe_unused]] rocshmem_team_t team,
                                             [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                             [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                             [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                             [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                             [[maybe_unused]] int pe_root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// SUM Reduction operations - Type-erased interface
__device__ inline int GDAContext::tile_sum_reduce([[maybe_unused]] rocshmem_team_t team,
                                           [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                           [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                           [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                           [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                           [[maybe_unused]] int root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_sum_reduce_wave([[maybe_unused]] rocshmem_team_t team,
                                                [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                                [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                                [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                                [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                                [[maybe_unused]] int root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_sum_reduce_wg([[maybe_unused]] rocshmem_team_t team,
                                              [[maybe_unused]] void* dst_data, [[maybe_unused]] const void* src_data,
                                              [[maybe_unused]] const size_t* dst_strides, [[maybe_unused]] const size_t* src_strides,
                                              [[maybe_unused]] const size_t* start_coord, [[maybe_unused]] const size_t* boundary,
                                              [[maybe_unused]] int ndim, [[maybe_unused]] size_t element_size,
                                              [[maybe_unused]] int root, [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// MAX Reduction operations - Type-erased interface
__device__ inline int GDAContext::tile_max_reduce([[maybe_unused]] rocshmem_team_t team,
                                                   [[maybe_unused]] void* dst_data,
                                                   [[maybe_unused]] const void* src_data,
                                                   [[maybe_unused]] const size_t* dst_strides,
                                                   [[maybe_unused]] const size_t* src_strides,
                                                   [[maybe_unused]] const size_t* start_coord,
                                                   [[maybe_unused]] const size_t* boundary,
                                                   [[maybe_unused]] int ndim,
                                                   [[maybe_unused]] size_t element_size,
                                                   [[maybe_unused]] int root,
                                                   [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_max_reduce_wave([[maybe_unused]] rocshmem_team_t team,
                                                        [[maybe_unused]] void* dst_data,
                                                        [[maybe_unused]] const void* src_data,
                                                        [[maybe_unused]] const size_t* dst_strides,
                                                        [[maybe_unused]] const size_t* src_strides,
                                                        [[maybe_unused]] const size_t* start_coord,
                                                        [[maybe_unused]] const size_t* boundary,
                                                        [[maybe_unused]] int ndim,
                                                        [[maybe_unused]] size_t element_size,
                                                        [[maybe_unused]] int root,
                                                        [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_max_reduce_wg([[maybe_unused]] rocshmem_team_t team,
                                                      [[maybe_unused]] void* dst_data,
                                                      [[maybe_unused]] const void* src_data,
                                                      [[maybe_unused]] const size_t* dst_strides,
                                                      [[maybe_unused]] const size_t* src_strides,
                                                      [[maybe_unused]] const size_t* start_coord,
                                                      [[maybe_unused]] const size_t* boundary,
                                                      [[maybe_unused]] int ndim,
                                                      [[maybe_unused]] size_t element_size,
                                                      [[maybe_unused]] int root,
                                                      [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// MIN Reduction operations - Type-erased interface
__device__ inline int GDAContext::tile_min_reduce([[maybe_unused]] rocshmem_team_t team,
                                                   [[maybe_unused]] void* dst_data,
                                                   [[maybe_unused]] const void* src_data,
                                                   [[maybe_unused]] const size_t* dst_strides,
                                                   [[maybe_unused]] const size_t* src_strides,
                                                   [[maybe_unused]] const size_t* start_coord,
                                                   [[maybe_unused]] const size_t* boundary,
                                                   [[maybe_unused]] int ndim,
                                                   [[maybe_unused]] size_t element_size,
                                                   [[maybe_unused]] int root,
                                                   [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_min_reduce_wave([[maybe_unused]] rocshmem_team_t team,
                                                        [[maybe_unused]] void* dst_data,
                                                        [[maybe_unused]] const void* src_data,
                                                        [[maybe_unused]] const size_t* dst_strides,
                                                        [[maybe_unused]] const size_t* src_strides,
                                                        [[maybe_unused]] const size_t* start_coord,
                                                        [[maybe_unused]] const size_t* boundary,
                                                        [[maybe_unused]] int ndim,
                                                        [[maybe_unused]] size_t element_size,
                                                        [[maybe_unused]] int root,
                                                        [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

__device__ inline int GDAContext::tile_min_reduce_wg([[maybe_unused]] rocshmem_team_t team,
                                                      [[maybe_unused]] void* dst_data,
                                                      [[maybe_unused]] const void* src_data,
                                                      [[maybe_unused]] const size_t* dst_strides,
                                                      [[maybe_unused]] const size_t* src_strides,
                                                      [[maybe_unused]] const size_t* start_coord,
                                                      [[maybe_unused]] const size_t* boundary,
                                                      [[maybe_unused]] int ndim,
                                                      [[maybe_unused]] size_t element_size,
                                                      [[maybe_unused]] int root,
                                                      [[maybe_unused]] uint64_t flags) {
  LOGD_WARN("Tile API not implemented for GDA backend");
  return ROCSHMEM_ERROR;
}

// Rooted SUM Reduction operations
// Rooted MAX Reduction operations
// Rooted MIN Reduction operations
}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_
