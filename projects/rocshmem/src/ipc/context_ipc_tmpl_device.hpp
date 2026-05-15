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
#include "context_ipc_device.hpp"
#include "util.hpp"
#include "ipc_team.hpp"
#include "rocshmem_calc.hpp"

#include <hip/hip_runtime.h>

namespace rocshmem {

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <typename T>
__device__ void IPCContext::p(T *dest, T value, int pe) {
  putmem_nbi(dest, &value, sizeof(T), pe);
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
  getmem(&ret, source, sizeof(T), pe);
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
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOAdd(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ void IPCContext::amo_set(void *dest, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOSet(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ T IPCContext::amo_swap(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOSwap(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_and(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOFetchAnd(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ void IPCContext::amo_and(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOAnd(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_or(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOFetchOr(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ void IPCContext::amo_or(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOOr(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_xor(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOFetchXor(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ void IPCContext::amo_xor(void *dest, T value, int pe) {
  uint64_t L_offset =
      reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOXor(
      reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ void IPCContext::amo_cas(void *dest, T value, T cond, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  ipcImpl_.ipcAMOCas(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), cond, value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_add(void *dest, T value, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOFetchAdd(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), value);
}

template <typename T>
__device__ T IPCContext::amo_fetch_cas(void *dest, T value, T cond, int pe) {
  uint64_t L_offset = reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
  return ipcImpl_.ipcAMOFetchCas(reinterpret_cast<T *>(ipcImpl_.ipc_bases[pe] + L_offset), cond, value);
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
__device__ void IPCContext::internal_direct_allreduce(
    T *dst, const T *src, int nelems, IPCTeam *team_obj) {  // NOLINT(runtime/int)

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

  for (int i = wg_id; i < num_pes; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
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
__device__ void IPCContext::internal_ring_allreduce(
    T *dst, const T *src, int nelems, IPCTeam *team_obj,  // NOLINT(runtime/int)
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
    // Loop 1 in the algorithm above
    for (int iter = 0; iter < PE_size - 1; iter++) {
      off_send = (((my_pe_in_team + 1 - iter + 2 * PE_size) % PE_size) * chunk_size);
      off_recv = (((my_pe_in_team - iter + 2 * PE_size) % PE_size) * chunk_size);

      internal_putmem_wg(reinterpret_cast<void *>(&pWrk[off_send]),
                    reinterpret_cast<void *>(&dst[off_send + off_seg]),
                    chunk_size * sizeof(T), send_pe);

      if (is_thread_zero_in_block()) {
        wait_val = seg + 100;
        fence(send_pe);
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
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

      if (is_thread_zero_in_block()) {
        wait_val = seg + 10;
        fence(send_pe);
        internal_putmem(&pSync[iter], &wait_val, sizeof(*pSync), send_pe);
        wait_until(&pSync[iter], ROCSHMEM_CMP_EQ, wait_val);
      }
      __syncthreads();
    }
  }

  for (int i = wg_id; i < 2 * num_pes - 2; i += wg_size) {
    pSync[i] = ROCSHMEM_SYNC_VALUE;
  }
  __syncthreads();
}

template <typename T, ROCSHMEM_OP Op>
__device__ int IPCContext::reduce(rocshmem_team_t team, T *dest,
                                  const T *source, int nreduce) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

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
          internal_ring_allreduce<T, Op>(p_dst, p_src,
                                         (p_chunk * PE_size), team_obj, 1,
                                         (p_chunk * PE_size), p_chunk);
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
      GPU_DPRINTF("Unsupported reduction size for IPC conduit.\n");
      return ROCSHMEM_ERROR;
    }
  }
  barrier_wg(team);
  return ROCSHMEM_SUCCESS;
}

template <typename T>
__device__ void IPCContext::internal_put_broadcast(
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
__device__ void IPCContext::internal_get_broadcast(
  T *dst, const T *src, int nelems, int pe_root) {  // NOLINT(runtime/int)
  if (my_pe != pe_root) {
    get_wg(dst, src, nelems, pe_root);
  }
}

template <typename T>
__device__ void IPCContext::broadcast(rocshmem_team_t team, T *dst,
                                      const T *src, int nelems, int pe_root) {
  IPCTeam *team_obj = reinterpret_cast<IPCTeam *>(team);

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
__device__ void IPCContext::internal_broadcast(T *dst, const T *src, int nelems,
                                      int pe_root, int pe_start,
                                      int stride, int pe_size,
                                      long *p_sync) {  // NOLINT(runtime/int)
  if (num_pes < 4) {
    internal_put_broadcast(dst, src, nelems, pe_root, pe_start, stride,
                           pe_size);
  } else {
    internal_get_broadcast(dst, src, nelems, pe_root);
  }

  // Synchronize on completion of broadcast
  internal_sync_wg(my_pe, pe_start, stride, pe_size, p_sync);
}

template <typename T>
__device__ void IPCContext::alltoall(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  alltoall_linear(team, dst, src, nelems);
}

template <typename T>
__device__ void IPCContext::alltoallv(rocshmem_team_t team,
                                      T *dest, const size_t dest_nelems[],
                                      const size_t dest_displs[],
                                      T *source, const size_t source_nelems[],
                                      const size_t source_displs[]) {
  printf("rocshmem::ipc:alltoallv not implemented\n");
  abort();
}

template <typename T>
__device__ void IPCContext::alltoall_linear(rocshmem_team_t team, T *dst,
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
  internal_sync_wg(my_pe, pe_start, stride, pe_size, pSync);
}

template <typename T>
__device__ void IPCContext::fcollect(rocshmem_team_t team, T *dst,
                                     const T *src, int nelems) {
  fcollect_linear(team, dst, src, nelems);
}

template <typename T>
__device__ void IPCContext::fcollect_linear(rocshmem_team_t team, T *dst,
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

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_CONTEXT_TMPL_DEVICE_HPP_
