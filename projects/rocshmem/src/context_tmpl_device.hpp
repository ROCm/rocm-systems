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

#ifndef LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP_
#define LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "backend_type.hpp"
#if defined(USE_GDA)
#include "gda/context_gda_device.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/context_ro_device.hpp"
#endif
#if defined(USE_IPC)
#include "ipc/context_ipc_device.hpp"
#endif

namespace rocshmem {

/*
 * Context dispatch implementations for the template functions. Needs to
 * be in a header and not cpp because it is a template.
 */
template <typename T>
__device__ void Context::p(T *dest, T value, int pe) {
  ctxStats.incStat(NUM_P);

  /*
   * TODO: Need to handle _p a bit differently for coalescing, since the
   * owner of a coalesced message needs val from all absorbed messages.
   */
  DISPATCH(p(dest, value, pe));
}

template <typename T>
__device__ T Context::g(T *source, int pe) {
  ctxStats.incStat(NUM_G);

  /*
   * TODO: Need to handle _g a bit differently for coalescing, since the
   * owner of a coalesced message needs val from all absorbed messages.
   */
  DISPATCH_RET(g(source, pe));
}

// The only way to get multi-arg templates to feed into a macro
template <typename T, ROCSHMEM_OP Op>
__device__ void Context::to_all(T *dest, const T *source, int nreduce,
                                int PE_start, int logPE_stride, int PE_size,
                                T *pWrk,
                                long *pSync) {  // NOLINT(runtime/int)
  if (nreduce == 0) {
    return;
  }

  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_TO_ALL);
  }

  DISPATCH(to_all<PAIR(T, Op)>(dest, source, nreduce, PE_start, logPE_stride,
                               PE_size, pWrk, pSync));
}

template <typename T, ROCSHMEM_OP Op>
__device__ int Context::reduce(rocshmem_team_t team, T *dest, const T *source,
                               int nreduce) {
  if (nreduce == 0) {
    return ROCSHMEM_SUCCESS;
  }

  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_TO_ALL);
  }

  DISPATCH_RET(reduce<PAIR(T, Op)>(team, dest, source, nreduce));
}

template <typename T>
__device__ void Context::put(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT);

  DISPATCH(put(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::put_nbi(T *dest, const T *source, size_t nelems,
                                 int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI);

  DISPATCH(put_nbi(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::get(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET);

  DISPATCH(get(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::get_nbi(T *dest, const T *source, size_t nelems,
                                 int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI);

  DISPATCH(get_nbi(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::alltoall(rocshmem_team_t team, T *dest,
                                  const T *source, int nelems) {
  if (nelems == 0) {
    return;
  }

  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_ALLTOALL);
  }

  DISPATCH(alltoall<T>(team, dest, source, nelems));
}

template <typename T>
__device__ void Context::alltoallv(rocshmem_team_t team,
                                   T *dest, const size_t dest_nelems[],
                                   const size_t dest_displs[],
                                   T *source, const size_t source_nelems[],
                                   const size_t source_displs[]) {

  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_ALLTOALLV);
  }

  DISPATCH(alltoallv<T>(team,
                        dest, dest_nelems, dest_displs,
                        source, source_nelems, source_displs));
}

template <typename T>
__device__ void Context::fcollect(rocshmem_team_t team, T *dest,
                                  const T *source, int nelems) {
  if (nelems == 0) {
    return;
  }

  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_FCOLLECT);
  }

  DISPATCH(fcollect<T>(team, dest, source, nelems));
}

template <typename T>
__device__ void Context::broadcast(rocshmem_team_t team, T *dest,
                                   const T *source, int nelems, int pe_root) {
  if (nelems == 0) {
    return;
  }

  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_BROADCAST);
  }

  DISPATCH(broadcast<T>(team, dest, source, nelems, pe_root));
}

template <typename T>
__device__ void Context::broadcast(T *dest, const T *source, int nelems,
                                   int pe_root, int pe_start, int log_pe_stride,
                                   int pe_size,
                                   long *p_sync) {  // NOLINT(runtime/int)
  if (nelems == 0) {
    return;
  }

  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_BROADCAST);
  }

  DISPATCH(broadcast<T>(dest, source, nelems, pe_root, pe_start, log_pe_stride,
                        pe_size, p_sync));
}

template <typename T>
__device__ __forceinline__ void Context::wait_until(T *ivars, int cmp,
                                                    T val) {
  while (!test(ivars, cmp, val)) {
  }
}

__device__ __forceinline__ size_t status_entry(size_t nelems,
                                               const int *status) {
  size_t i{0};
  while (i < nelems) {
    if (status[i] == 0) {
      return i;
    }
    i++;
  }
  return i;
}

template <typename T>
__device__ __forceinline__
size_t Context::wait_until_any(T *ivars, size_t nelems,
                               const int *status,
                               int cmp, T val) {
  // zero nelems error condition
  if (!nelems) {
    return SIZE_MAX;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return SIZE_MAX;
  }

  while (true) {
    for (size_t i{pos}; i < nelems; i++) {
      // skip entries marked with non-zero status
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val)) {
        return i;
      }
    }
  }
}

template <typename T>
__device__ __forceinline__
void Context::wait_until_all(T *ivars, size_t nelems,
                             const int *status,
                             int cmp, T val) {
  // zero nelems error condition
  if (!nelems) {
    return;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return;
  }

  for (size_t i{pos}; i < nelems; i++) {
    if (status[i]) {
      continue;
    }
    while (!test(ivars + i, cmp, val)) {
    }
  }
}

template <typename T>
__device__ __forceinline__
size_t Context::wait_until_some(T *ivars, size_t nelems,
                              size_t* indices,
                              const int *status,
                              int cmp, T val) {
  // zero nelems error condition
  if (!nelems) {
    return 0;
  }

  size_t pos{status_entry(nelems, status)};

  // invalid (empty) status array error condition
  if (pos == nelems) {
    return 0;
  }

  bool done {false};
  size_t ncompleted {0};
  while (!done) {
    for (size_t i{pos}; i < nelems; i++) {
      // skip entries marked with non-zero status
      if (status[i]) {
        continue;
      }
      if (test(ivars + i, cmp, val)) {
        done = true;
        indices[ncompleted] = i;
        ncompleted++;
      }
    }
  }
  return ncompleted;
}

template <typename T>
__device__ __forceinline__
void Context::wait_until_all_vector(T *ivars, size_t nelems,
                                    const int *status,
                                    int cmp, T* vals) {
  ;
}

template <typename T>
__device__ __forceinline__
size_t Context::wait_until_any_vector(T *ivars, size_t nelems,
                                      const int *status,
                                      int cmp, T* vals) {
  return 0;
}

template <typename T>
__device__ __forceinline__
size_t Context::wait_until_some_vector(T *ivars, size_t nelems,
                                     size_t* indices,
                                     const int *status,
                                     int cmp, T* vals) {
  return 0;
}

template <typename T>
__device__ __forceinline__ int Context::test(T *ivars, int cmp,
                                             T val) {
  int ret = 0;
  volatile T *vol_ivars = reinterpret_cast<T *>(ivars);
  switch (cmp) {
    case ROCSHMEM_CMP_EQ:
      if (uncached_load(vol_ivars) == val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_NE:
      if (uncached_load(vol_ivars) != val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_GT:
      if (uncached_load(vol_ivars) > val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_GE:
      if (uncached_load(vol_ivars) >= val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_LT:
      if (uncached_load(vol_ivars) < val) {
        ret = 1;
      }
      break;
    case ROCSHMEM_CMP_LE:
      if (uncached_load(vol_ivars) <= val) {
        ret = 1;
      }
      break;
    default:
      break;
  }
  return ret;
}

template <typename T>
__device__ void Context::put_wg(T *dest, const T *source, size_t nelems,
                                int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_WG);

  DISPATCH(put_wg(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::put_nbi_wg(T *dest, const T *source, size_t nelems,
                                    int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI_WG);

  DISPATCH(put_nbi_wg(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::get_wg(T *dest, const T *source, size_t nelems,
                                int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_WG);

  DISPATCH(get_wg(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::get_nbi_wg(T *dest, const T *source, size_t nelems,
                                    int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI_WG);

  DISPATCH(get_nbi_wg(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::put_wave(T *dest, const T *source, size_t nelems,
                                  int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_WAVE);

  DISPATCH(put_wave(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::put_nbi_wave(T *dest, const T *source, size_t nelems,
                                      int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI_WAVE);

  DISPATCH(put_nbi_wave(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::get_wave(T *dest, const T *source, size_t nelems,
                                  int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_WAVE);

  DISPATCH(get_wave(dest, source, nelems, pe));
}

template <typename T>
__device__ void Context::get_nbi_wave(T *dest, const T *source, size_t nelems,
                                      int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI_WAVE);

  DISPATCH(get_nbi_wave(dest, source, nelems, pe));
}

template <typename T>
__device__ T Context::amo_fetch_add(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_FADD);

  DISPATCH_RET(amo_fetch_add(dst, value, pe));
}

template <typename T>
__device__ void Context::amo_add(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_ADD);

  DISPATCH(amo_add(dst, value, pe));
}

template <typename T>
__device__ void Context::amo_set(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_SET);

  DISPATCH(amo_set(dst, value, pe));
}

template <typename T>
__device__ T Context::amo_swap(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_SWAP);

  DISPATCH_RET(amo_swap(dst, value, pe));
}

template <typename T>
__device__ T Context::amo_fetch_and(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_FETCH_AND);

  DISPATCH_RET(amo_fetch_and(dst, value, pe));
}

template <typename T>
__device__ void Context::amo_and(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_AND);

  DISPATCH(amo_and(dst, value, pe));
}

template <typename T>
__device__ T Context::amo_fetch_or(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_FETCH_OR);

  DISPATCH_RET(amo_fetch_or(dst, value, pe));
}

template <typename T>
__device__ void Context::amo_or(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_OR);

  DISPATCH(amo_or(dst, value, pe));
}

template <typename T>
__device__ T Context::amo_fetch_xor(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_FETCH_XOR);

  DISPATCH_RET(amo_fetch_xor(dst, value, pe));
}

template <typename T>
__device__ void Context::amo_xor(void *dst, T value, int pe) {
  ctxStats.incStat(NUM_ATOMIC_XOR);

  DISPATCH(amo_xor(dst, value, pe));
}

template <typename T>
__device__ T Context::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  ctxStats.incStat(NUM_ATOMIC_FCSWAP);

  DISPATCH_RET(amo_fetch_cas(dst, value, cond, pe));
}

template <typename T>
__device__ void Context::amo_cas(void *dst, T value, T cond, int pe) {
  ctxStats.incStat(NUM_ATOMIC_CSWAP);

  DISPATCH(amo_cas(dst, value, cond, pe));
}

#define CONTEXT_PUT_SIGNAL_DEF(SUFFIX, STATS_SUFFIX)                                           \
  template <typename T>                                                                        \
  __device__ void Context::put_signal##SUFFIX(T *dest, const T *source, size_t nelems,         \
                                              uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                              int pe) {                                        \
    if (nelems == 0) {                                                                         \
      return;                                                                                  \
    }                                                                                          \
                                                                                               \
    ctxStats.incStat(NUM_PUT_SIGNAL##STATS_SUFFIX);                                            \
                                                                                               \
    DISPATCH(put_signal##SUFFIX(dest, source, nelems, sig_addr, signal, sig_op, pe));          \
  }

CONTEXT_PUT_SIGNAL_DEF(,)
CONTEXT_PUT_SIGNAL_DEF(_wg, _WG)
CONTEXT_PUT_SIGNAL_DEF(_wave, _WAVE)
CONTEXT_PUT_SIGNAL_DEF(_nbi, _NBI)
CONTEXT_PUT_SIGNAL_DEF(_nbi_wg, _NBI_WG)
CONTEXT_PUT_SIGNAL_DEF(_nbi_wave, _NBI_WAVE)

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP_
