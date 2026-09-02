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
#include "tile_layout.hpp"
#include "tile_memcpy_device.hpp"
#include "context_gda_device.hpp"
#include "gda_team.hpp"
#include "queue_pair_provider.hpp"
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
  char *remote{nullptr};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe) &&
      (remote = ipcImpl_.ipcPeerPtr(dest, local_pe)) != nullptr) {
    ipcImpl_.ipcCopy<MemcpyKind::Put>(remote, reinterpret_cast<void *>(&value), sizeof(T), local_pe);
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
  char *remote{nullptr};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe) &&
      (remote = ipcImpl_.ipcPeerPtr(source, local_pe)) != nullptr) {
    ipcImpl_.ipcCopy<MemcpyKind::Get>(&ret, remote, sizeof(T), local_pe);
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
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_add(dst, value, wf_info);
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
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_compare_swap(
             dst, cond, value, wf_info)) != cond) {
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
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_compare_swap(
             dst, cond, desired_val, wf_info)) != cond) {
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
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_compare_swap(
             dst, cond, desired_val, wf_info)) != cond) {
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
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_compare_swap(
             dst, cond, desired_val, wf_info)) != cond) {
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
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_compare_swap_nofetch(dst, cond, value, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ T GDAContext::amo_fetch_add(void *dst, T value, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fadd not implemented for non-64bit types"); }//TODO:support for non-uint64t
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  T ret_val = 0;
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val =  qps[qp_index].atomic_fetch_add(dst, value, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
  return ret_val;
}

template <typename T>
__device__ T GDAContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fcas not implemented for non-64bit types"); }//TODO:support for non-uint64t
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  T ret_val;
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val = qps[qp_index].atomic_compare_swap(dst, cond, value, wf_info);
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
__device__ void GDAContext::internal_direct_allreduce_wg(T *dst, const T *src,
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

template <typename T, ROCSHMEM_OP Op>
__device__ void GDAContext::internal_direct_allreduce_wave(T *dst, const T *src,
    int nelems, GDATeam *team_obj, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)

  int stride = team_obj->tinfo_wrt_world->stride;
  int PE_start = team_obj->tinfo_wrt_world->pe_start;
  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);

  int finish = PE_start + stride * PE_size;
  int pe = constmem.my_pe;
  int wf_tid = get_flat_block_id() % WF_SIZE;
  int64_t flag_val = 1;

  // Initialize local dst from src
  for (int i = wf_tid; i < nelems; i += WF_SIZE) {
    dst[i] = src[i];
  }

  // Put src to pWrk on all other PEs and signal via pSync
  for (int i = PE_start; i < finish; i += stride) {
    if (i != pe) {
      internal_putmem_nbi_wave(&pWrk[pe * nelems],
        reinterpret_cast<const void *>(src), nelems * sizeof(T), i, i, wf_info);
      if (is_thread_zero_in_wave()) {
        qps[i].quiet(wf_info);
        internal_putmem(&pSync[pe], &flag_val, sizeof(*pSync), i, i, wf_info);
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
    for (int i = 0; i < constmem.num_pes; i++) {
      pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    __threadfence_system();
  }
}

template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::reduce_wave(rocshmem_team_t team, T *dest,
                                       const T *source, int nreduce) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int PE_size = team_obj->tinfo_wrt_world->size;

  size_t direct_pWrk = PE_size * nreduce;
  size_t direct_pSync = PE_size;
  size_t provided_pWrk = max(nreduce / 2 + 1, ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE);
  size_t provided_pSync = ROCSHMEM_REDUCE_SYNC_SIZE;

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);

  size_t ring_pSync = 2 * PE_size;

  if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
    internal_direct_allreduce_wave<T, Op>(dest, source, nreduce, team_obj, wf_info);
  } else {
    if (ring_pSync <= ROCSHMEM_REDUCE_SYNC_SIZE) {
      size_t ring_pWrk = ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE;
      int chunk_size = ring_pWrk / PE_size;
      int seg_size = chunk_size * PE_size;

      int n_seg = nreduce / seg_size;
      int n_seg_up = (nreduce - 1) / seg_size + 1;

      if (n_seg > 0) {
        internal_ring_allreduce_wave<T, Op>(dest, source, nreduce, team_obj,
          n_seg, seg_size, chunk_size, wf_info);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;

        if (p_chunk > 0) {
          internal_ring_allreduce_wave<T, Op>(p_dst, p_src, 
            (p_chunk * PE_size), team_obj, 1, 
            (p_chunk * PE_size), p_chunk, wf_info);
        }

        if ((p_chunk * PE_size) < p_count) {
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);
          internal_direct_allreduce_wave<T, Op>(p_dst, p_src2, p_count, team_obj, wf_info);
        }
      }
    } else {
      LOGD_WARN("Unsupported reduction size for GDA wave conduit.");
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
__device__ void GDAContext::internal_ring_allreduce_wg(T *dst, const T *src,
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
      
      fence();

      if (is_thread_zero_in_block()) {
        wait_val = seg + 100;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe,
          send_pe, wf_info);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      fence();
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
      fence();
      if (is_thread_zero_in_block()) {
        wait_val = seg + 10;
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe,
          send_pe, wf_info);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      fence();
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
__device__ void GDAContext::internal_ring_allreduce_wave(T *dst, const T *src,
    int nelems, GDATeam *team_obj,  // NOLINT(runtime/int)
    int n_seg, int seg_size, int chunk_size, ActiveWFInfo &wf_info) {

  int PE_size = team_obj->tinfo_wrt_world->size;
  long *pSync = team_obj->reduce_pSync;
  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  int my_pe_in_team = team_obj->my_pe;

  int off_seg, off_send, off_recv;
  int send_pe = (my_pe_in_team + 1) % PE_size;
  send_pe = team_obj->get_pe_in_world(send_pe);
  long wait_val;  // NOLINT(runtime/int)

  int wf_tid = get_flat_block_id() % WF_SIZE;

  for (int i = wf_tid; i < nelems; i += WF_SIZE) {
    dst[i] = src[i];
  }

  for (int seg = 0; seg < n_seg; seg++) {
    off_seg = seg * seg_size;
    // Loop 1: reduce-scatter
    for (int iter = 0; iter < PE_size - 1; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      off_recv = (((my_pe_in_team - iter + 2 * PE_size) % PE_size) * chunk_size);

      internal_putmem_wave(reinterpret_cast<void *>(&pWrk[off_send]),
        reinterpret_cast<void *>(&dst[off_send + off_seg]),
        chunk_size * sizeof(T), send_pe, send_pe, wf_info);
      fence();
      if (is_thread_zero_in_wave()) {
        qps[send_pe].quiet(wf_info);
        wait_val = seg + 100;
        internal_putmem_wave(&pSync[iter], &wait_val, sizeof(*pSync), send_pe,
          send_pe, wf_info);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      fence();
      for (int j = wf_tid; j < chunk_size; j += WF_SIZE) {
        OpWrap<Op>::Calc(&pWrk[off_recv], &dst[off_seg + off_recv], j);
      }
    }

    // Loop 2: all-gather
    for (int iter = PE_size - 1; iter < 2 * PE_size - 2; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      internal_putmem_nbi_wave(reinterpret_cast<void *>(&dst[off_send + off_seg]),
        reinterpret_cast<void *>(&dst[off_send + off_seg]),
        chunk_size * sizeof(T), send_pe, send_pe, wf_info);
      fence();
      if (is_thread_zero_in_wave()) {
        qps[send_pe].quiet(wf_info);
        wait_val = seg + 10;
        internal_putmem_wave(&pSync[iter], &wait_val, sizeof(*pSync), send_pe,
          send_pe, wf_info);
#if defined(__gfx90a__)
        __threadfence_system();
#endif /* __gfx90a__ */
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
    }
  }

  if (is_thread_zero_in_wave()) {
    for (int i = 0; i < 2 * constmem.num_pes - 2; i++) {
      pSync[i] = ROCSHMEM_SYNC_VALUE;
    }
    __threadfence_system();
  }
}

template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::reduce_wg(rocshmem_team_t team, T *dest,
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
    internal_direct_allreduce_wg<T, Op>(dest, source, nreduce, team_obj, wf_info);
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
          seg_size, chunk_size, wf_info);
      }
      if (n_seg_up > n_seg) {
        T *p_dst = (dest + (n_seg * seg_size));
        const T *p_src = (source + (n_seg * seg_size));
        int p_count = nreduce - (n_seg * seg_size);
        int p_chunk = p_count / PE_size;

        if (p_chunk > 0) {
          internal_ring_allreduce_wg<T, Op>(p_dst, p_src, (p_chunk * PE_size),
            team_obj, 1, (p_chunk * PE_size), p_chunk, wf_info);
        }

        if ((p_chunk * PE_size) < p_count) {
          // Final elements need to use direct_allreduce
          p_count -= (p_chunk * PE_size);
          p_dst += (p_chunk * PE_size);
          const T *p_src2 = p_src + (p_chunk * PE_size);

          internal_direct_allreduce_wg<T, Op>(p_dst, p_src2, p_count, team_obj, wf_info);
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

template <typename T, ROCSHMEM_OP Op>
__device__ int GDAContext::reduce_scatter_wave(rocshmem_team_t team, T *dest,
                                               const T *source, int nreduce) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

  int PE_size   = team_obj->tinfo_wrt_world->size;
  int PE_start  = team_obj->tinfo_wrt_world->pe_start;
  int stride    = team_obj->tinfo_wrt_world->stride;
  int team_rank = (my_pe - PE_start) / stride;

  long *pSync = team_obj->reduce_pSync;
  T    *pWrk  = reinterpret_cast<T *>(team_obj->pWrk);

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wave);

  int wave_tid  = get_flat_block_id() % WF_SIZE;

  int pWrk_elems = (int)(ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double) / sizeof(T));
  int chunk_size = max(1, pWrk_elems / PE_size);
  int n_chunks   = (nreduce + chunk_size - 1) / chunk_size;
  int64_t flag_val = 1;
  int finish = PE_start + stride * PE_size;

  for (int c = 0; c < n_chunks; c++) {
    int offset = c * chunk_size;
    int count  = min(chunk_size, nreduce - offset);

    // Seed dest[offset..offset+count) from my own contribution.
    for (int j = wave_tid; j < count; j += WF_SIZE) {
      dest[offset + j] = source[team_rank * nreduce + offset + j];
    }
    __builtin_amdgcn_wave_barrier();

    // Send my contribution for each remote PE's output block, then signal.
    for (int i = PE_start; i < finish; i += stride) {
      if (i != my_pe) {
        int remote_rank = (i - PE_start) / stride;
        internal_putmem_wave(&pWrk[team_rank * chunk_size],
                              reinterpret_cast<const void *>(
                                  source + remote_rank * nreduce + offset),
                              count * sizeof(T), i, i, wf_info);
        if (is_thread_zero_in_wave()) {
          fence();
          internal_putmem(&pSync[team_rank], &flag_val, sizeof(*pSync), i, i, wf_info);
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
        T *src_chunk = &pWrk[remote_rank * chunk_size];
        T *dst_chunk = dest + offset;
        for (int j = wave_tid; j < count; j += WF_SIZE) {
          OpWrap<Op>::Calc(src_chunk, dst_chunk, j);
        }
        threadfence_system();
      }
    }
    __builtin_amdgcn_wave_barrier();

    // Reset pSync before reuse.
    for (int j = wave_tid; j < PE_size; j += WF_SIZE) {
      pSync[j] = ROCSHMEM_SYNC_VALUE;
    }
    threadfence_system();
    // Quiet all QPs used this chunk, then sync with wave 0 of other PEs.
    if (is_thread_zero_in_wave()) {
      for (int i = PE_start; i < finish; i += stride) {
        if (i != my_pe) {
          qps[i].quiet(wf_info);
        }
      }
      sync_wave(team);
    }
  }

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void GDAContext::internal_put_broadcast_wave(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe == pe_root) {
    int finish = pe_start + stride * pe_size;
    for (int i = pe_start; i < finish; i += stride) {
      if (i != constmem.my_pe) {
        internal_putmem_nbi_wave(dst, src, nelems * sizeof(T), i, i, wf_info);
      }
    }
    memcpy_wave<MemcpyKind::Put>(dst, const_cast<T *>(src), nelems * sizeof(T));
  }
}

template <typename T>
__device__ void GDAContext::internal_get_broadcast_wave(T *dst, const T *src,
    int nelems, int pe_root, ActiveWFInfo &wf_info) {  // NOLINT(runtime/int)
  if (constmem.my_pe != pe_root) {
    internal_getmem_wave(dst, src, nelems * sizeof(T), pe_root, pe_root, wf_info);
  } else {
    memcpy_wave<MemcpyKind::Get>(dst, const_cast<T *>(src), nelems * sizeof(T));
  }
}

template <typename T>
__device__ int GDAContext::broadcast_wave(rocshmem_team_t team, T *dest, 
    const T* source, int nelems, int PE_root) {
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
  internal_broadcast_wave<T>(dest, source, nelems, pe_root_world, pe_start, stride,
               pe_size, p_sync);
  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void GDAContext::internal_broadcast_wave(T *dst, const T *src,
    int nelems, int pe_root, int pe_start, int stride, int pe_size,
    long *p_sync) {  // NOLINT(runtime/int)

  ActiveWFInfo wf_info(ctx_id_, ThreadScope::wg);
  if (constmem.num_pes < 4) { //TODO: optimized for IPC
    internal_put_broadcast_wave(dst, src, nelems, pe_root, pe_start, stride,
      pe_size, wf_info);
  } else {
    internal_get_broadcast_wave(dst, src, nelems, pe_root, wf_info);
  }

  // Synchronize on completion of broadcast
  internal_sync_wave(constmem.my_pe, pe_start, stride, pe_size, p_sync, wf_info);
}

template <typename T>
__device__ void GDAContext::broadcast_wg(rocshmem_team_t team, T *dst,
    const T *src, int nelems, int pe_root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);

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
__device__ void GDAContext::alltoall_wg(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  alltoallmem_linear_thread_puts_wg(team, dst, src, nelems * sizeof(T));
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
    size_t nelems = source_nelems[j] * sizeof(T);
    long *amo_dst = &pSync[alltoall_pSync_offset + my_pe_in_team];

    if (nelems != 0) {
      T *src = &source[source_displs[j]];
      T *dst = &tmp_buf[my_pe_in_team * tmp_buf_off];
      qps[dest_pe].put_nbi_single(dst, src, nelems, PostOpt{RingDB<false>});
    }

    qps[dest_pe].atomic_add_single(amo_dst, 1);
  }

  // wait until everyone has obtained their designated data
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    long *sync_flags = &pSync[alltoall_pSync_offset + j];
    while (uncached_load(sync_flags) != 1) { }

    qps[dest_pe].quiet_single();

    pSync[alltoall_pSync_offset + j] = ROCSHMEM_SYNC_VALUE;
  }

  // Copy out of staging buffer
  __syncthreads();

  for (int j = 0; j < pe_size; j++) {
    size_t nelems = dest_nelems[j] * sizeof(T);

    if (nelems != 0) {
      T *dst = &dest[dest_displs[j]];
      T *src = &tmp_buf[j * tmp_buf_off];
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
  uint64_t *tmp_buf = reinterpret_cast<uint64_t*>(team_obj->pWrk);

  static constexpr uint64_t displs_mask = 0x0000'FFFF'FFFF'FFFF;
  static constexpr uint64_t seq_mask = 0xFFFF;
  static constexpr uint64_t seq_shift = 48;

  int tid = get_flat_block_id();
  int step_size = min(get_flat_block_size(), WF_SIZE);

  /* Put Ctrl Message */
  for (int j = tid; j < pe_size; j+= step_size) {
    int dest_pe = team_obj->get_pe_in_world(j);

    /* Pack Ctrl Message * 16 bits seq | 48bit displ */
    uint64_t seq_bits = (seq_mask & (a2a_sn + 1)) << seq_shift;
    uint64_t displ_bits = (displs_mask & source_displs[j]);
    uint64_t ctrl_msg = seq_bits | displ_bits;

    /* Prepare Ctrl Message */
    uint64_t *ctrl_src = &ctrl_msg;
    uint64_t *ctrl_dst = &tmp_buf[my_pe_in_team];

    static_assert(QueuePair::can_inline<QueuePair::OpCode::RDMA_WRITE>(sizeof(ctrl_msg)),
                  "alltoallv_get control message must be posted inline");
    qps[dest_pe].put_nbi_single(ctrl_dst, ctrl_src, sizeof(ctrl_msg), PostOpt{RingDB<true>});

    /* Wait for Ctrl Message */
    uint64_t ctrl_value;
    uint64_t *vol_ctrl = &tmp_buf[j];

    do {
      ctrl_value = uncached_load(vol_ctrl);
      seq_bits = (ctrl_value >> seq_shift) & seq_mask;
      displ_bits = ctrl_value & displs_mask;
    } while (seq_bits != ((a2a_sn + 1) & seq_mask));

    /* Get data */
    size_t nelems = dest_nelems[j] * sizeof(T);
    T *src = &source[displ_bits];
    T *dst = &dest[dest_displs[j]];

    qps[dest_pe].get_nbi_single(dst, src, nelems, PostOpt{RingDB<true>});

    /* Put Completion */
    long *amo_dst = &pSync[alltoall_pSync_offset + my_pe_in_team];
    qps[dest_pe].atomic_add_single(amo_dst, 1);

    long *sync_flags = &pSync[alltoall_pSync_offset + j];
    while (uncached_load(sync_flags) != 1) { }

    qps[dest_pe].quiet_single();

    pSync[alltoall_pSync_offset + j] = ROCSHMEM_SYNC_VALUE;
  }

  if (is_thread_zero_in_block()) {
    team_obj->alltoall_sequence_number++;
  }

  __syncthreads();
}

template <typename T>
__device__ void GDAContext::alltoall_linear_wg(rocshmem_team_t team, T *dst,
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
__device__ int GDAContext::alltoall_wave(rocshmem_team_t team, 
                          T* dest, const T* source, int nelems) {
  if (dest == nullptr || source == nullptr)
    return ROCSHMEM_ERROR;

  alltoallmem_linear_thread_puts_wave(team, dest, source, nelems * sizeof(T));

  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void GDAContext::fcollect_wg(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  fcollectmem_linear_wg(team, dst, src, nelems * sizeof(T));
}

template <typename T>
__device__ int GDAContext::fcollect_wave(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  if (dst == nullptr || src == nullptr || team == ROCSHMEM_TEAM_INVALID)
    return ROCSHMEM_ERROR;

  fcollectmem_linear_wave(team, dst, src, nelems * sizeof(T));

  return ROCSHMEM_SUCCESS;
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
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      qps[qp_index].atomic_add(dst, value, wf_info);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <typename T>
__device__ T GDAContext::internal_amo_fetch_add(void *dst, T value, int pe,
    int qp_index, ActiveWFInfo &wf_info) {
  if constexpr (sizeof(T) != 8) { LOGD_ERROR_ABORT("gda::amo_fadd not implemented for non-64bit types"); }//TODO:support for non-uint64t
  T ret_val = 0;
  bool need_turn {true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      ret_val =  qps[qp_index].atomic_fetch_add(dst, value, wf_info);
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
      while (wf_info.update(pe), (ret_val = qps[qp_index].atomic_compare_swap(
             dst, cond, value, wf_info)) != cond) {
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
 ******************** TILE API RMA IMPLEMENTATIONS ****************************
 *****************************************************************************/

// Whether it is worth striping the chunks across workers. Below one chunk per
// worker, callers fall back to a leader-sequential path rather than leave most
// workers idle. Unequal per-worker chunk counts are safe either way: each
// posting round is a collective over whichever lanes are still in the loop.
__device__ __forceinline__ bool gda_tile_use_multi_wqe(size_t num_chunks,
                                                       int worker_count) {
  return worker_count > 1 &&
         num_chunks >= static_cast<size_t>(worker_count);
}

// Synchronize outstanding tile PUT operations (IPC quiet or GDA QP quiet)
__device__ __forceinline__ void GDAContext::tile_finish_put(int pe, int qp_index,
                                                            ActiveWFInfo &wf_info) {
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe)) {
    ipcImpl_.ipcQuiet();
  } else {
    qps[qp_index].quiet(wf_info);
  }
}

// GET completion uses the same quiet path as PUT
__device__ __forceinline__ void GDAContext::tile_finish_get(int pe, int qp_index,
                                                            ActiveWFInfo &wf_info) {
  tile_finish_put(pe, qp_index, wf_info);
}

__device__ inline void GDAContext::tile_put_chunk_nbi(char *dst, const char *src,
                                                     size_t bytes, int pe,
                                                     int qp_index) {
  /*
   * Built here rather than by the caller: in a striped loop the set of lanes
   * still posting shrinks on the final round, and the group must match the
   * execution mask at the point the WQEs are written.
   */
  ActiveWFInfo wf_info(pe);
  auto [dst_raddr, dst_rkey] = qps[qp_index].get_raddr_info(dst);
  uint32_t src_lkey =
      (static_cast<int32_t>(bytes) <=
       static_cast<int32_t>(qps[qp_index].inline_threshold))
          ? 0
          : qps[qp_index].get_lkey(reinterpret_cast<uintptr_t>(src));
  qps[qp_index].put_nbi(reinterpret_cast<void *>(dst_raddr), dst_rkey,
                        src, src_lkey, bytes, wf_info, true);
}

__device__ inline void GDAContext::tile_get_chunk_nbi(char *dst, const char *src,
                                                     size_t bytes, int pe,
                                                     int qp_index) {
  ActiveWFInfo wf_info(pe);
  qps[qp_index].get_nbi(dst, src, bytes, wf_info);
}

__device__ inline int GDAContext::tile_qp_index_for_worker(int pe, int worker_id,
                                                          int worker_count) {
  if (worker_count <= WF_SIZE) {
    int qp_index = pe;
    if (worker_id == 0) {
      const ThreadScope scope =
          (worker_count == 1) ? ThreadScope::thread : ThreadScope::wave;
      ActiveWFInfo lead(pe, scope);
      qp_index = static_cast<int>(get_qp_index(pe, lead));
    }
    if (worker_count > 1) {
      qp_index = __shfl(qp_index, 0);
    }
    return qp_index;
  }
  const int wave_id = worker_id / WF_SIZE;
  return (wave_id % static_cast<int>(num_qps_per_pe)) * constmem.num_pes + pe;
}

__device__ inline void GDAContext::tile_quiet_gda_workers(int pe, int worker_id,
                                                         int worker_count,
                                                         int wave_qp_index) {
  if (worker_count <= WF_SIZE) {
    if (worker_count > 1) {
      __builtin_amdgcn_wave_barrier();
    }
    if (worker_id == 0) {
      qps[wave_qp_index].quiet_single();
    }
  } else {
    __syncthreads();
    if (worker_id == 0) {
      for (uint32_t i = 0; i < num_qps_per_pe; i++) {
        qps[i * constmem.num_pes + pe].quiet_single();
      }
    }
  }
}

__device__ inline void GDAContext::tile_put_contig_slices_nbi(
    char *dst, const char *src, size_t bytes, int pe, int qp_index,
    int worker_id, int worker_count) {
  constexpr size_t kMinSlice = 64;
  if (worker_count == 1 || bytes < kMinSlice * static_cast<size_t>(worker_count)) {
    if (worker_id == 0 && bytes != 0) {
      tile_put_chunk_nbi(dst, src, bytes, pe, qp_index);
    }
    return;
  }
  const size_t n = static_cast<size_t>(worker_count);
  const size_t chunk = bytes / n;
  const size_t start = static_cast<size_t>(worker_id) * chunk;
  const size_t len =
      (worker_id == worker_count - 1) ? (bytes - start) : chunk;
  if (len != 0) {
    tile_put_chunk_nbi(dst + start, src + start, len, pe, qp_index);
  }
}

__device__ inline void GDAContext::tile_get_contig_slices_nbi(
    char *dst, const char *src, size_t bytes, int pe, int qp_index,
    int worker_id, int worker_count) {
  constexpr size_t kMinSlice = 64;
  if (worker_count == 1 || bytes < kMinSlice * static_cast<size_t>(worker_count)) {
    if (worker_id == 0 && bytes != 0) {
      tile_get_chunk_nbi(dst, src, bytes, pe, qp_index);
    }
    return;
  }
  const size_t n = static_cast<size_t>(worker_count);
  const size_t chunk = bytes / n;
  const size_t start = static_cast<size_t>(worker_id) * chunk;
  const size_t len =
      (worker_id == worker_count - 1) ? (bytes - start) : chunk;
  if (len != 0) {
    tile_get_chunk_nbi(dst + start, src + start, len, pe, qp_index);
  }
}

__device__ inline void GDAContext::tile_put_rows_nbi(
    char *dst_base, const char *src_base, size_t dst_row_stride_bytes,
    size_t src_row_stride_bytes, size_t num_rows, size_t row_bytes,
    int pe, int qp_index, int worker_id, int worker_count) {
  for (size_t i = static_cast<size_t>(worker_id); i < num_rows;
       i += static_cast<size_t>(worker_count)) {
    tile_put_chunk_nbi(dst_base + i * dst_row_stride_bytes,
                       src_base + i * src_row_stride_bytes, row_bytes, pe,
                       qp_index);
  }
}

__device__ inline void GDAContext::tile_put_cols_nbi(
    char *dst_base, const char *src_base, size_t dst_col_stride_bytes,
    size_t src_col_stride_bytes, size_t num_cols, size_t col_bytes,
    int pe, int qp_index, int worker_id, int worker_count) {
  for (size_t j = static_cast<size_t>(worker_id); j < num_cols;
       j += static_cast<size_t>(worker_count)) {
    tile_put_chunk_nbi(dst_base + j * dst_col_stride_bytes,
                       src_base + j * src_col_stride_bytes, col_bytes, pe,
                       qp_index);
  }
}

__device__ inline void GDAContext::tile_get_rows_nbi(
    char *dst_base, const char *src_base, size_t dst_row_stride_bytes,
    size_t src_row_stride_bytes, size_t num_rows, size_t row_bytes,
    int pe, int qp_index, int worker_id, int worker_count) {
  for (size_t i = static_cast<size_t>(worker_id); i < num_rows;
       i += static_cast<size_t>(worker_count)) {
    tile_get_chunk_nbi(dst_base + i * dst_row_stride_bytes,
                       src_base + i * src_row_stride_bytes, row_bytes, pe,
                       qp_index);
  }
}

__device__ inline void GDAContext::tile_get_cols_nbi(
    char *dst_base, const char *src_base, size_t dst_col_stride_bytes,
    size_t src_col_stride_bytes, size_t num_cols, size_t col_bytes,
    int pe, int qp_index, int worker_id, int worker_count) {
  for (size_t j = static_cast<size_t>(worker_id); j < num_cols;
       j += static_cast<size_t>(worker_count)) {
    tile_get_chunk_nbi(dst_base + j * dst_col_stride_bytes,
                       src_base + j * src_col_stride_bytes, col_bytes, pe,
                       qp_index);
  }
}

__device__ inline void GDAContext::tile_put_strided_2d_nbi(
    char *dst_base, const char *src_base, size_t dst_s0, size_t dst_s1,
    size_t src_s0, size_t src_s1, size_t extent0, size_t extent1,
    size_t element_size, int pe, int qp_index, int worker_id,
    int worker_count) {
  const size_t n = extent0 * extent1;
  for (size_t idx = static_cast<size_t>(worker_id); idx < n;
       idx += static_cast<size_t>(worker_count)) {
    const size_t i = idx / extent1;
    const size_t j = idx % extent1;
    tile_put_chunk_nbi(dst_base + (i * dst_s0 + j * dst_s1) * element_size,
                       src_base + (i * src_s0 + j * src_s1) * element_size,
                       element_size, pe, qp_index);
  }
}

__device__ inline void GDAContext::tile_get_strided_2d_nbi(
    char *dst_base, const char *src_base, size_t dst_s0, size_t dst_s1,
    size_t src_s0, size_t src_s1, size_t extent0, size_t extent1,
    size_t element_size, int pe, int qp_index, int worker_id,
    int worker_count) {
  const size_t n = extent0 * extent1;
  for (size_t idx = static_cast<size_t>(worker_id); idx < n;
       idx += static_cast<size_t>(worker_count)) {
    const size_t i = idx / extent1;
    const size_t j = idx % extent1;
    tile_get_chunk_nbi(dst_base + (i * dst_s0 + j * dst_s1) * element_size,
                       src_base + (i * src_s0 + j * src_s1) * element_size,
                       element_size, pe, qp_index);
  }
}

__device__ inline void GDAContext::tile_put_gda_workers(
    void *dst_data, const void *src_data, const size_t *dst_strides,
    const size_t *src_strides, const size_t *start_coord,
    const size_t *boundary, int ndim, size_t element_size, int pe,
    int worker_id, int worker_count) {
  const int qp_index = tile_qp_index_for_worker(pe, worker_id, worker_count);

  if (ndim == 2) {
    const TileView view = tile_make_view(
        dst_data, src_data, dst_strides, src_strides, start_coord, boundary,
        ndim, element_size, true);
    const size_t src_s0 = view.src_s0;
    const size_t src_s1 = view.src_s1;
    const size_t dst_s0 = view.dst_s0;
    const size_t dst_s1 = view.dst_s1;
    const size_t ext0 = view.ext0;
    const size_t ext1 = view.ext1;
    char *src_base = view.src_base;
    char *dst_base = view.dst_base;
    const TileLayout layout = tile_classify(view);

    switch (layout) {
      case TileLayout::Contiguous:
        tile_put_contig_slices_nbi(dst_base, src_base, ext0 * ext1 * element_size,
                                   pe, qp_index, worker_id, worker_count);
        break;
      case TileLayout::RowContig: {
        const size_t row_bytes = ext1 * element_size;
        if (gda_tile_use_multi_wqe(ext0, worker_count)) {
          tile_put_rows_nbi(dst_base, src_base, dst_s0 * element_size,
                            src_s0 * element_size, ext0, row_bytes, pe,
                            qp_index, worker_id, worker_count);
        } else if (worker_id == 0) {
          for (size_t i = 0; i < ext0; i++) {
            tile_put_chunk_nbi(dst_base + i * dst_s0 * element_size,
                               src_base + i * src_s0 * element_size, row_bytes,
                               pe, qp_index);
          }
        }
        break;
      }
      case TileLayout::ColContig: {
        const size_t col_bytes = ext0 * element_size;
        if (gda_tile_use_multi_wqe(ext1, worker_count)) {
          tile_put_cols_nbi(dst_base, src_base, dst_s1 * element_size,
                            src_s1 * element_size, ext1, col_bytes, pe,
                            qp_index, worker_id, worker_count);
        } else if (worker_id == 0) {
          for (size_t j = 0; j < ext1; j++) {
            tile_put_chunk_nbi(dst_base + j * dst_s1 * element_size,
                               src_base + j * src_s1 * element_size, col_bytes,
                               pe, qp_index);
          }
        }
        break;
      }
      case TileLayout::Strided:
      default: {
        const size_t n = ext0 * ext1;
        if (gda_tile_use_multi_wqe(n, worker_count)) {
          tile_put_strided_2d_nbi(dst_base, src_base, dst_s0, dst_s1, src_s0,
                                  src_s1, ext0, ext1, element_size, pe,
                                  qp_index, worker_id, worker_count);
        } else if (worker_id == 0) {
          tile_put_strided_2d_nbi(dst_base, src_base, dst_s0, dst_s1, src_s0,
                                  src_s1, ext0, ext1, element_size, pe,
                                  qp_index, 0, 1);
        }
        break;
      }
    }
  } else if (ndim == 1) {
    const TileView view = tile_make_view(
        dst_data, src_data, dst_strides, src_strides, start_coord, boundary,
        ndim, element_size, true);
    const size_t ext = view.ext0;
    char *src_ptr = view.src_base;
    char *dst_ptr = view.dst_base;
    if (view.src_s0 == 1 && view.dst_s0 == 1) {
      tile_put_contig_slices_nbi(dst_ptr, src_ptr, ext * element_size, pe,
                                 qp_index, worker_id, worker_count);
    } else if (gda_tile_use_multi_wqe(ext, worker_count)) {
      tile_put_rows_nbi(dst_ptr, src_ptr, view.dst_s0 * element_size,
                        view.src_s0 * element_size, ext, element_size,
                        pe, qp_index, worker_id, worker_count);
    } else if (worker_id == 0) {
      tile_put_rows_nbi(dst_ptr, src_ptr, view.dst_s0 * element_size,
                        view.src_s0 * element_size, ext, element_size,
                        pe, qp_index, 0, 1);
    }
  }

  tile_quiet_gda_workers(pe, worker_id, worker_count, qp_index);
}

__device__ inline void GDAContext::tile_get_gda_workers(
    void *dst_data, const void *src_data, const size_t *dst_strides,
    const size_t *src_strides, const size_t *start_coord,
    const size_t *boundary, int ndim, size_t element_size, int pe,
    int worker_id, int worker_count) {
  const int qp_index = tile_qp_index_for_worker(pe, worker_id, worker_count);

  if (ndim == 2) {
    const TileView view = tile_make_view(
        dst_data, src_data, dst_strides, src_strides, start_coord, boundary,
        ndim, element_size, false);
    const size_t src_s0 = view.src_s0;
    const size_t src_s1 = view.src_s1;
    const size_t dst_s0 = view.dst_s0;
    const size_t dst_s1 = view.dst_s1;
    const size_t ext0 = view.ext0;
    const size_t ext1 = view.ext1;
    char *src_base = view.src_base;
    char *dst_base = view.dst_base;
    const TileLayout layout = tile_classify(view);

    switch (layout) {
      case TileLayout::Contiguous:
        tile_get_contig_slices_nbi(dst_base, src_base, ext0 * ext1 * element_size,
                                   pe, qp_index, worker_id, worker_count);
        break;
      case TileLayout::RowContig: {
        const size_t row_bytes = ext1 * element_size;
        if (gda_tile_use_multi_wqe(ext0, worker_count)) {
          tile_get_rows_nbi(dst_base, src_base, dst_s0 * element_size,
                            src_s0 * element_size, ext0, row_bytes, pe,
                            qp_index, worker_id, worker_count);
        } else if (worker_id == 0) {
          for (size_t i = 0; i < ext0; i++) {
            tile_get_chunk_nbi(dst_base + i * dst_s0 * element_size,
                               src_base + i * src_s0 * element_size, row_bytes,
                               pe, qp_index);
          }
        }
        break;
      }
      case TileLayout::ColContig: {
        const size_t col_bytes = ext0 * element_size;
        if (gda_tile_use_multi_wqe(ext1, worker_count)) {
          tile_get_cols_nbi(dst_base, src_base, dst_s1 * element_size,
                            src_s1 * element_size, ext1, col_bytes, pe,
                            qp_index, worker_id, worker_count);
        } else if (worker_id == 0) {
          for (size_t j = 0; j < ext1; j++) {
            tile_get_chunk_nbi(dst_base + j * dst_s1 * element_size,
                               src_base + j * src_s1 * element_size, col_bytes,
                               pe, qp_index);
          }
        }
        break;
      }
      case TileLayout::Strided:
      default: {
        const size_t n = ext0 * ext1;
        if (gda_tile_use_multi_wqe(n, worker_count)) {
          tile_get_strided_2d_nbi(dst_base, src_base, dst_s0, dst_s1, src_s0,
                                  src_s1, ext0, ext1, element_size, pe,
                                  qp_index, worker_id, worker_count);
        } else if (worker_id == 0) {
          tile_get_strided_2d_nbi(dst_base, src_base, dst_s0, dst_s1, src_s0,
                                  src_s1, ext0, ext1, element_size, pe,
                                  qp_index, 0, 1);
        }
        break;
      }
    }
  } else if (ndim == 1) {
    const TileView view = tile_make_view(
        dst_data, src_data, dst_strides, src_strides, start_coord, boundary,
        ndim, element_size, false);
    const size_t ext = view.ext0;
    char *src_ptr = view.src_base;
    char *dst_ptr = view.dst_base;
    if (view.src_s0 == 1 && view.dst_s0 == 1) {
      tile_get_contig_slices_nbi(dst_ptr, src_ptr, ext * element_size, pe,
                                 qp_index, worker_id, worker_count);
    } else if (gda_tile_use_multi_wqe(ext, worker_count)) {
      tile_get_rows_nbi(dst_ptr, src_ptr, view.dst_s0 * element_size,
                        view.src_s0 * element_size, ext, element_size,
                        pe, qp_index, worker_id, worker_count);
    } else if (worker_id == 0) {
      tile_get_rows_nbi(dst_ptr, src_ptr, view.dst_s0 * element_size,
                        view.src_s0 * element_size, ext, element_size,
                        pe, qp_index, 0, 1);
    }
  }

  tile_quiet_gda_workers(pe, worker_id, worker_count, qp_index);
}

// RMA Operations - Type-erased implementations
__device__ inline int GDAContext::tile_put(void* dst_data, const void* src_data,
                                           const size_t* dst_strides, const size_t* src_strides,
                                           const size_t* start_coord, const size_t* boundary,
                                           int ndim, size_t element_size, int pe,
                                           [[maybe_unused]] uint64_t flags) {
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  const TileView view = tile_make_view(
      dst_data, src_data, dst_strides, src_strides, start_coord, boundary, ndim,
      element_size, true);

  if (view.ndim == 2) {
    switch (tile_classify(view)) {
      case TileLayout::Contiguous:
        internal_putmem_nbi(view.dst_base, view.src_base,
                            view.ext0 * view.ext1 * view.element_size, pe,
                            qp_index, wf_info);
        break;
      case TileLayout::RowContig:
        for (size_t i = 0; i < view.ext0; i++) {
          internal_putmem_nbi(
              view.dst_base + i * view.dst_s0 * view.element_size,
              view.src_base + i * view.src_s0 * view.element_size,
              view.ext1 * view.element_size, pe, qp_index, wf_info);
        }
        break;
      case TileLayout::ColContig:
        for (size_t j = 0; j < view.ext1; j++) {
          internal_putmem_nbi(
              view.dst_base + j * view.dst_s1 * view.element_size,
              view.src_base + j * view.src_s1 * view.element_size,
              view.ext0 * view.element_size, pe, qp_index, wf_info);
        }
        break;
      case TileLayout::Strided:
      default:
        for (size_t i = 0; i < view.ext0; i++) {
          for (size_t j = 0; j < view.ext1; j++) {
            internal_putmem_nbi(
                view.dst_base +
                    (i * view.dst_s0 + j * view.dst_s1) * view.element_size,
                view.src_base +
                    (i * view.src_s0 + j * view.src_s1) * view.element_size,
                view.element_size, pe, qp_index, wf_info);
          }
        }
        break;
    }
  } else if (view.ndim == 1) {
    if (tile_classify(view) == TileLayout::Contiguous) {
      internal_putmem_nbi(view.dst_base, view.src_base,
                          view.ext0 * view.element_size, pe, qp_index, wf_info);
    } else {
      for (size_t i = 0; i < view.ext0; i++) {
        internal_putmem_nbi(
            view.dst_base + i * view.dst_s0 * view.element_size,
            view.src_base + i * view.src_s0 * view.element_size,
            view.element_size, pe, qp_index, wf_info);
      }
    }
  }

  tile_finish_put(pe, qp_index, wf_info);
  return ROCSHMEM_SUCCESS;
}

__device__ inline int GDAContext::tile_put_wave(void* dst_data, const void* src_data,
                                                const size_t* dst_strides, const size_t* src_strides,
                                                const size_t* start_coord, const size_t* boundary,
                                                int ndim, size_t element_size, int pe,
                                                [[maybe_unused]] uint64_t flags) {
  int local_pe{-1};
  const bool ipc_avail = ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe);

  // IPC fast path: direct peer access via shmem_ptr
  if (ipc_avail) {
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
  } else {
    const int wave_tid = get_flat_block_id() % WF_SIZE;
    tile_put_gda_workers(dst_data, src_data, dst_strides, src_strides, start_coord,
                         boundary, ndim, element_size, pe, wave_tid, WF_SIZE);
  }

  return ROCSHMEM_SUCCESS;
}

__device__ inline int GDAContext::tile_put_wg(void* dst_data, const void* src_data,
                                              const size_t* dst_strides, const size_t* src_strides,
                                              const size_t* start_coord, const size_t* boundary,
                                              int ndim, size_t element_size, int pe,
                                              [[maybe_unused]] uint64_t flags) {
  int local_pe{-1};
  const bool ipc_avail = ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe);

  // IPC fast path: direct peer access via shmem_ptr
  if (ipc_avail) {
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
  } else {
    tile_put_gda_workers(dst_data, src_data, dst_strides, src_strides, start_coord,
                         boundary, ndim, element_size, pe, get_flat_block_id(),
                         get_flat_block_size());
    __builtin_amdgcn_s_barrier();
  }

  return ROCSHMEM_SUCCESS;
}

// RMA GET operations - Type-erased implementations
__device__ inline int GDAContext::tile_get(void* dst_data, const void* src_data,
                                           const size_t* dst_strides, const size_t* src_strides,
                                           const size_t* start_coord, const size_t* boundary,
                                           int ndim, size_t element_size, int pe,
                                           [[maybe_unused]] uint64_t flags) {
  ActiveWFInfo wf_info(pe);
  int qp_index = get_qp_index(pe, wf_info);
  const TileView view = tile_make_view(
      dst_data, src_data, dst_strides, src_strides, start_coord, boundary, ndim,
      element_size, false);

  if (view.ndim == 2) {
    switch (tile_classify(view)) {
      case TileLayout::Contiguous:
        internal_getmem_nbi(view.dst_base, view.src_base,
                            view.ext0 * view.ext1 * view.element_size, pe,
                            qp_index, wf_info);
        break;
      case TileLayout::RowContig:
        for (size_t i = 0; i < view.ext0; i++) {
          internal_getmem_nbi(
              view.dst_base + i * view.dst_s0 * view.element_size,
              view.src_base + i * view.src_s0 * view.element_size,
              view.ext1 * view.element_size, pe, qp_index, wf_info);
        }
        break;
      case TileLayout::ColContig:
        for (size_t j = 0; j < view.ext1; j++) {
          internal_getmem_nbi(
              view.dst_base + j * view.dst_s1 * view.element_size,
              view.src_base + j * view.src_s1 * view.element_size,
              view.ext0 * view.element_size, pe, qp_index, wf_info);
        }
        break;
      case TileLayout::Strided:
      default:
        for (size_t i = 0; i < view.ext0; i++) {
          for (size_t j = 0; j < view.ext1; j++) {
            internal_getmem_nbi(
                view.dst_base +
                    (i * view.dst_s0 + j * view.dst_s1) * view.element_size,
                view.src_base +
                    (i * view.src_s0 + j * view.src_s1) * view.element_size,
                view.element_size, pe, qp_index, wf_info);
          }
        }
        break;
    }
  } else if (view.ndim == 1) {
    if (tile_classify(view) == TileLayout::Contiguous) {
      internal_getmem_nbi(view.dst_base, view.src_base,
                          view.ext0 * view.element_size, pe, qp_index, wf_info);
    } else {
      for (size_t i = 0; i < view.ext0; i++) {
        internal_getmem_nbi(
            view.dst_base + i * view.dst_s0 * view.element_size,
            view.src_base + i * view.src_s0 * view.element_size,
            view.element_size, pe, qp_index, wf_info);
      }
    }
  }

  tile_finish_get(pe, qp_index, wf_info);
  return ROCSHMEM_SUCCESS;
}

__device__ inline int GDAContext::tile_get_wave(void* dst_data, const void* src_data,
                                                const size_t* dst_strides, const size_t* src_strides,
                                                const size_t* start_coord, const size_t* boundary,
                                                int ndim, size_t element_size, int pe,
                                                [[maybe_unused]] uint64_t flags) {
  int local_pe{-1};
  const bool ipc_avail = ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe);

  // IPC fast path: direct peer access via shmem_ptr
  if (ipc_avail) {
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
  } else {
    const int wave_tid = get_flat_block_id() % WF_SIZE;
    tile_get_gda_workers(dst_data, src_data, dst_strides, src_strides, start_coord,
                         boundary, ndim, element_size, pe, wave_tid, WF_SIZE);
  }

  return ROCSHMEM_SUCCESS;
}

__device__ inline int GDAContext::tile_get_wg(void* dst_data, const void* src_data,
                                              const size_t* dst_strides, const size_t* src_strides,
                                              const size_t* start_coord, const size_t* boundary,
                                              int ndim, size_t element_size, int pe,
                                              [[maybe_unused]] uint64_t flags) {
  int local_pe{-1};
  const bool ipc_avail = ipcImpl_.isIpcAvailable(constmem.my_pe, pe, &local_pe);

  // IPC fast path: direct peer access via shmem_ptr
  if (ipc_avail) {
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
  } else {
    tile_get_gda_workers(dst_data, src_data, dst_strides, src_strides, start_coord,
                         boundary, ndim, element_size, pe, get_flat_block_id(),
                         get_flat_block_size());
    __builtin_amdgcn_s_barrier();
  }

  return ROCSHMEM_SUCCESS;
}

// Collective Allgather - Type-erased implementations
__device__ inline int GDAContext::tile_allgather(rocshmem_team_t team,
                                                 void* dst_data,
                                                 const void* src_data,
                                                 const size_t* dst_strides,
                                                 const size_t* src_strides,
                                                 const size_t* start_coord,
                                                 const size_t* boundary,
                                                 int ndim,
                                                 size_t element_size,
                                                 uint64_t flags) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
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

__device__ inline int GDAContext::tile_allgather_wave(rocshmem_team_t team,
                                                      void* dst_data,
                                                      const void* src_data,
                                                      const size_t* dst_strides,
                                                      const size_t* src_strides,
                                                      const size_t* start_coord,
                                                      const size_t* boundary,
                                                      int ndim,
                                                      size_t element_size,
                                                      uint64_t flags) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
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

__device__ inline int GDAContext::tile_allgather_wg(rocshmem_team_t team,
                                                    void* dst_data,
                                                    const void* src_data,
                                                    const size_t* dst_strides,
                                                    const size_t* src_strides,
                                                    const size_t* start_coord,
                                                    const size_t* boundary,
                                                    int ndim,
                                                    size_t element_size,
                                                    uint64_t flags) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
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
__device__ inline int GDAContext::tile_broadcast(rocshmem_team_t team,
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
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
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

__device__ inline int GDAContext::tile_broadcast_wave(rocshmem_team_t team,
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
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
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

__device__ inline int GDAContext::tile_broadcast_wg(rocshmem_team_t team,
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
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
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
__device__ inline size_t gda_tile_num_elements(const size_t* start_coord,
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

__device__ inline size_t gda_tile_dst_offset(size_t flat_idx,
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

__device__ inline size_t gda_tile_src_offset(size_t flat_idx,
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

__device__ inline bool gda_tile_is_contiguous(const size_t* strides,
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
__device__ inline int GDAContext::tile_reduce_typed_impl(
    rocshmem_team_t team, const void* src_data, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root,
    size_t segment_start, size_t segment_elems, size_t segment_capacity,
    int worker_id, int worker_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  const int team_size = team_obj->num_pes;
  const int my_pe_in_team = team_obj->my_pe;
  const int root_pe_world = team_obj->get_pe_in_world(root);

  if (root < 0 || root >= team_size || ndim <= 0) {
    LOGD_WARN("Invalid tile reduce arguments for GDA backend");
    return ROCSHMEM_ERROR;
  }

  T *pWrk = reinterpret_cast<T *>(team_obj->pWrk);
  T *my_pWrk = &pWrk[my_pe_in_team * segment_capacity];

  const bool src_contiguous =
      gda_tile_is_contiguous(src_strides, start_coord, boundary, ndim);
  const size_t first_src_offset = gda_tile_src_offset(
      segment_start, src_strides, start_coord, boundary, ndim);
  const char *src_base = static_cast<const char *>(src_data) +
                         first_src_offset * sizeof(T);

  // All workers must participate (wave shuffle in QP selection).
  const int qp_index =
      tile_qp_index_for_worker(root_pe_world, worker_id, worker_count);
  const size_t segment_bytes = segment_elems * sizeof(T);

  if (src_contiguous && my_pe_in_team != root) {
    tile_put_contig_slices_nbi(reinterpret_cast<char *>(my_pWrk), src_base,
                               segment_bytes, root_pe_world, qp_index,
                               worker_id, worker_count);
    tile_quiet_gda_workers(root_pe_world, worker_id, worker_count, qp_index);
  } else if (src_contiguous && my_pe_in_team == root) {
    const T *src_typed = reinterpret_cast<const T *>(src_base);
    if (worker_count == 1) {
      __builtin_memcpy(my_pWrk, src_typed, segment_bytes);
    } else {
      for (size_t elem = worker_id; elem < segment_elems; elem += worker_count)
        my_pWrk[elem] = src_typed[elem];
    }
  } else {
    // Pack strided source into the contiguous PE slot, then bulk-put to root.
    for (size_t elem = worker_id; elem < segment_elems; elem += worker_count) {
      const size_t tile_elem = segment_start + elem;
      const size_t src_offset =
          gda_tile_src_offset(tile_elem, src_strides, start_coord, boundary, ndim);
      my_pWrk[elem] =
          *reinterpret_cast<const T *>(static_cast<const char *>(src_data) +
                                       src_offset * sizeof(T));
    }

    if (my_pe_in_team != root) {
      if (worker_count == WF_SIZE) {
        __builtin_amdgcn_wave_barrier();
      } else if (worker_count > 1) {
        __syncthreads();
      }
      tile_put_contig_slices_nbi(reinterpret_cast<char *>(my_pWrk),
                                 reinterpret_cast<const char *>(my_pWrk),
                                 segment_bytes, root_pe_world, qp_index,
                                 worker_id, worker_count);
      tile_quiet_gda_workers(root_pe_world, worker_id, worker_count, qp_index);
    }
  }

  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline void gda_tile_reduce_root_compute(
    rocshmem_team_t team, void* dst_data, const size_t* dst_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root,
    size_t segment_start, size_t segment_elems, size_t segment_capacity,
    int worker_id, int worker_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
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
        gda_tile_dst_offset(tile_elem, dst_strides, start_coord, boundary, ndim);
    char *dst_base = static_cast<char *>(dst_data);
    *reinterpret_cast<T *>(dst_base + dst_offset * sizeof(T)) = reduced_value;
  }
}

__device__ inline void gda_tile_reduce_reset_psync(rocshmem_team_t team,
                                                   int root,
                                                   int worker_id,
                                                   int worker_count) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  if (team_obj->my_pe != root) {
    return;
  }

  long *pSync = team_obj->reduce_pSync;
  for (int i = worker_id; i < team_obj->num_pes; i += worker_count) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int GDAContext::tile_reduce_typed(
    rocshmem_team_t team, void* dst_data, const void* src_data,
    const size_t* dst_strides, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root) {
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  long *pSync = team_obj->reduce_pSync;
  const int root_pe_world = team_obj->get_pe_in_world(root);
  const size_t tile_elements =
      gda_tile_num_elements(start_coord, boundary, ndim);
  const size_t pwrk_capacity =
      (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / sizeof(T);
  const size_t segment_capacity = pwrk_capacity / team_obj->num_pes;

  if (tile_elements == 0) {
    LOGD_WARN("Tile reduce: invalid coordinate range (underflow or overflow)");
    return ROCSHMEM_ERROR;
  }

  if (segment_capacity == 0) {
    LOGD_WARN("Tile reduce type exceeds GDA pWrk capacity");
    return ROCSHMEM_ERROR;
  }

  if (team_obj->num_pes > static_cast<int>(ROCSHMEM_REDUCE_SYNC_SIZE)) {
    LOGD_WARN("Tile reduce team size exceeds GDA reduce_pSync capacity");
    return ROCSHMEM_ERROR;
  }

  ActiveWFInfo wf_info(root_pe_world, ThreadScope::thread);
  const int qp_index = root_pe_world;

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
                      root_pe_world, qp_index, wf_info);
    }

    gda_tile_reduce_root_compute<T, Op>(
        team, dst_data, dst_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, 0, 1);
    sync(team);
  }

  gda_tile_reduce_reset_psync(team, root, 0, 1);
  sync(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int GDAContext::tile_reduce_typed_wave(
    rocshmem_team_t team, void* dst_data, const void* src_data,
    const size_t* dst_strides, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root) {
  const int wave_id = get_flat_block_id() % WF_SIZE;
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  long *pSync = team_obj->reduce_pSync;
  const int root_pe_world = team_obj->get_pe_in_world(root);
  const size_t tile_elements =
      gda_tile_num_elements(start_coord, boundary, ndim);
  const size_t pwrk_capacity =
      (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / sizeof(T);
  const size_t segment_capacity = pwrk_capacity / team_obj->num_pes;

  if (tile_elements == 0) {
    LOGD_WARN("Tile reduce: invalid coordinate range (underflow or overflow)");
    return ROCSHMEM_ERROR;
  }

  if (segment_capacity == 0) {
    LOGD_WARN("Tile reduce type exceeds GDA pWrk capacity");
    return ROCSHMEM_ERROR;
  }

  if (team_obj->num_pes > static_cast<int>(ROCSHMEM_REDUCE_SYNC_SIZE)) {
    LOGD_WARN("Tile reduce team size exceeds GDA reduce_pSync capacity");
    return ROCSHMEM_ERROR;
  }

  ActiveWFInfo wf_info(root_pe_world, ThreadScope::wave);
  const int qp_index = root_pe_world;

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
                        root_pe_world, qp_index, wf_info);
      }
    }

    __builtin_amdgcn_wave_barrier();
    gda_tile_reduce_root_compute<T, Op>(
        team, dst_data, dst_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, wave_id, WF_SIZE);
    sync_wave(team);
  }

  gda_tile_reduce_reset_psync(team, root, wave_id, WF_SIZE);
  sync_wave(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int GDAContext::tile_reduce_typed_wg(
    rocshmem_team_t team, void* dst_data, const void* src_data,
    const size_t* dst_strides, const size_t* src_strides,
    const size_t* start_coord, const size_t* boundary, int ndim, int root) {
  const int thread_id = get_flat_block_id();
  const int block_size = get_flat_block_size();
  GDATeam *team_obj = reinterpret_cast<GDATeam *>(team);
  long *pSync = team_obj->reduce_pSync;
  const int root_pe_world = team_obj->get_pe_in_world(root);
  const size_t tile_elements =
      gda_tile_num_elements(start_coord, boundary, ndim);
  const size_t pwrk_capacity =
      (ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)) / sizeof(T);
  const size_t segment_capacity = pwrk_capacity / team_obj->num_pes;

  if (tile_elements == 0) {
    LOGD_WARN("Tile reduce: invalid coordinate range (underflow or overflow)");
    return ROCSHMEM_ERROR;
  }

  if (segment_capacity == 0) {
    LOGD_WARN("Tile reduce type exceeds GDA pWrk capacity");
    return ROCSHMEM_ERROR;
  }

  if (team_obj->num_pes > static_cast<int>(ROCSHMEM_REDUCE_SYNC_SIZE)) {
    LOGD_WARN("Tile reduce team size exceeds GDA reduce_pSync capacity");
    return ROCSHMEM_ERROR;
  }

  ActiveWFInfo wf_info(root_pe_world, ThreadScope::wg);
  const int qp_index = root_pe_world;

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
                        root_pe_world, qp_index, wf_info);
      }
    }

    __syncthreads();
    gda_tile_reduce_root_compute<T, Op>(
        team, dst_data, dst_strides, start_coord, boundary, ndim, root,
        segment_start, segment_elems, segment_capacity, thread_id, block_size);
    sync_wg(team);
  }

  gda_tile_reduce_reset_psync(team, root, thread_id, block_size);
  sync_wg(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T, ROCSHMEM_OP Op>
__device__ inline int gda_tile_reduce_typed(GDAContext* ctx,
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
__device__ inline int gda_tile_reduce_typed_wave(GDAContext* ctx,
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
__device__ inline int gda_tile_reduce_typed_wg(GDAContext* ctx,
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

#define GDA_TILE_REDUCE_DISPATCH_CASES(TYPED_FN)                              \
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

#define GDA_TILE_REDUCE_DISPATCH_DEF(DISPATCH_FN, TYPED_FN)                   \
  template <ROCSHMEM_OP Op>                                                   \
  __device__ inline int DISPATCH_FN(                                          \
      GDAContext* ctx, rocshmem_team_t team, void* dst_data,                  \
      const void* src_data, const size_t* dst_strides,                        \
      const size_t* src_strides, const size_t* start_coord,                   \
      const size_t* boundary, int ndim,                                       \
      [[maybe_unused]] size_t element_size, int root, uint64_t flags) {       \
    const auto element_type = static_cast<ROCSHMEM_TILE_ELEMENT_TYPE>(        \
        (flags & ROCSHMEM_TILE_ELEMENT_TYPE_MASK) >>                          \
        ROCSHMEM_TILE_ELEMENT_TYPE_SHIFT);                                    \
                                                                               \
    switch (element_type) {                                                   \
      GDA_TILE_REDUCE_DISPATCH_CASES(TYPED_FN);                               \
      default:                                                                \
        LOGD_WARN("Tile reduce element type not specified for GDA backend");   \
        return ROCSHMEM_ERROR;                                                \
    }                                                                         \
  }

GDA_TILE_REDUCE_DISPATCH_DEF(gda_tile_reduce_dispatch, gda_tile_reduce_typed)
GDA_TILE_REDUCE_DISPATCH_DEF(gda_tile_reduce_wave_dispatch,
                             gda_tile_reduce_typed_wave)
GDA_TILE_REDUCE_DISPATCH_DEF(gda_tile_reduce_wg_dispatch,
                             gda_tile_reduce_typed_wg)

#undef GDA_TILE_REDUCE_DISPATCH_DEF
#undef GDA_TILE_REDUCE_DISPATCH_CASES

// SUM Reductions - Type-erased implementations
__device__ inline int GDAContext::tile_sum_reduce(rocshmem_team_t team,
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
  return gda_tile_reduce_dispatch<ROCSHMEM_SUM>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int GDAContext::tile_sum_reduce_wave(rocshmem_team_t team,
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
  return gda_tile_reduce_wave_dispatch<ROCSHMEM_SUM>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int GDAContext::tile_sum_reduce_wg(rocshmem_team_t team,
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
  return gda_tile_reduce_wg_dispatch<ROCSHMEM_SUM>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

// MAX Reductions - Type-erased interface
__device__ inline int GDAContext::tile_max_reduce(rocshmem_team_t team,
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
  return gda_tile_reduce_dispatch<ROCSHMEM_MAX>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int GDAContext::tile_max_reduce_wave(rocshmem_team_t team,
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
  return gda_tile_reduce_wave_dispatch<ROCSHMEM_MAX>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int GDAContext::tile_max_reduce_wg(rocshmem_team_t team,
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
  return gda_tile_reduce_wg_dispatch<ROCSHMEM_MAX>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

// MIN Reductions - Type-erased interface
__device__ inline int GDAContext::tile_min_reduce(rocshmem_team_t team,
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
  return gda_tile_reduce_dispatch<ROCSHMEM_MIN>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int GDAContext::tile_min_reduce_wave(rocshmem_team_t team,
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
  return gda_tile_reduce_wave_dispatch<ROCSHMEM_MIN>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ inline int GDAContext::tile_min_reduce_wg(rocshmem_team_t team,
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
  return gda_tile_reduce_wg_dispatch<ROCSHMEM_MIN>(
      this, team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

// Rooted SUM Reduction operations
// Rooted MAX Reduction operations
// Rooted MIN Reduction operations
}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_CONTEXT_TMPL_DEVICE_HPP_
