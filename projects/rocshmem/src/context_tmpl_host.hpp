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

#ifndef LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP_
#define LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP_

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "backend_type.hpp"
#if defined(USE_GDA)
#include "gda/context_gda_host.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/context_ro_host.hpp"
#endif
#if defined(USE_IPC)
#include "ipc/context_ipc_host.hpp"
#endif

namespace rocshmem {

template <typename T>
__host__ void Context::p(T *dest, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_P);

  HOST_DISPATCH(p(dest, value, pe));
}

template <typename T>
__host__ T Context::g(const T *source, int pe) {
  ctxHostStats.incStat(NUM_HOST_G);

  HOST_DISPATCH_RET(g(source, pe));
}

template <typename T>
__host__ void Context::put(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_PUT);

  HOST_DISPATCH(put(dest, source, nelems, pe));
}

template <typename T>
__host__ void Context::get(T *dest, const T *source, size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_GET);

  HOST_DISPATCH(get(dest, source, nelems, pe));
}

template <typename T>
__host__ void Context::put_nbi(T *dest, const T *source, size_t nelems,
                               int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_PUT_NBI);

  HOST_DISPATCH(put_nbi(dest, source, nelems, pe));
}

template <typename T>
__host__ void Context::get_nbi(T *dest, const T *source, size_t nelems,
                               int pe) {
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_GET_NBI);

  HOST_DISPATCH(get_nbi(dest, source, nelems, pe));
}

template <typename T>
__host__ T Context::amo_fetch_add(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_FADD);

  HOST_DISPATCH_RET(amo_fetch_add(dst, value, pe));
}

template <typename T>
__host__ void Context::amo_add(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_ADD);

  HOST_DISPATCH(amo_add(dst, value, pe));
}

template <typename T>
__host__ void Context::amo_set(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_ADD);

  HOST_DISPATCH(amo_set(dst, value, pe));
}

template <typename T>
__host__ T Context::amo_swap(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_ADD);

  HOST_DISPATCH_RET(amo_swap(dst, value, pe));
}

template <typename T>
__host__ T Context::amo_fetch_and(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_FETCH_AND);

  HOST_DISPATCH_RET(amo_fetch_and(dst, value, pe));
}

template <typename T>
__host__ void Context::amo_and(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_AND);

  HOST_DISPATCH(amo_and(dst, value, pe));
}

template <typename T>
__host__ T Context::amo_fetch_or(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_FETCH_OR);

  HOST_DISPATCH_RET(amo_fetch_or(dst, value, pe));
}

template <typename T>
__host__ void Context::amo_or(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_OR);

  HOST_DISPATCH(amo_or(dst, value, pe));
}

template <typename T>
__host__ T Context::amo_fetch_xor(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_FETCH_XOR);

  HOST_DISPATCH_RET(amo_fetch_xor(dst, value, pe));
}

template <typename T>
__host__ void Context::amo_xor(void *dst, T value, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_XOR);

  HOST_DISPATCH(amo_xor(dst, value, pe));
}

template <typename T>
__host__ T Context::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_FCSWAP);

  HOST_DISPATCH_RET(amo_fetch_cas(dst, value, cond, pe));
}

template <typename T>
__host__ void Context::amo_cas(void *dst, T value, T cond, int pe) {
  ctxHostStats.incStat(NUM_HOST_ATOMIC_CSWAP);

  HOST_DISPATCH(amo_cas(dst, value, cond, pe));
}

template <typename T>
__host__ void Context::broadcast(T *dest, const T *source, int nelems,
                                 int pe_root, int pe_start, int log_pe_stride,
                                 int pe_size,
                                 long *p_sync) {  // NOLINT(runtime/int)
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_BROADCAST);

  HOST_DISPATCH(broadcast<T>(dest, source, nelems, pe_root, pe_start,
                             log_pe_stride, pe_size, p_sync));
}

template <typename T>
__host__ void Context::broadcast(rocshmem_team_t team, T *dest,
                                 const T *source, int nelems,
                                 int pe_root) {  // NOLINT(runtime/int)
  if (nelems == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_BROADCAST);

  HOST_DISPATCH(broadcast<T>(team, dest, source, nelems, pe_root));
}

template <typename T, ROCSHMEM_OP Op>
__host__ void Context::to_all(T *dest, const T *source, int nreduce,
                              int PE_start, int logPE_stride, int PE_size,
                              T *pWrk,
                              long *pSync) {  // NOLINT(runtime/int)
  if (nreduce == 0) {
    return;
  }

  ctxHostStats.incStat(NUM_HOST_TO_ALL);

  HOST_DISPATCH(to_all<PAIR(T, Op)>(dest, source, nreduce, PE_start,
                                    logPE_stride, PE_size, pWrk, pSync));
}

template <typename T, ROCSHMEM_OP Op>
__host__ int Context::reduce(rocshmem_team_t team, T *dest, const T *source,
                             int nreduce) {  // NOLINT(runtime/int)
  if (nreduce == 0) {
    return ROCSHMEM_SUCCESS;
  }

  ctxHostStats.incStat(NUM_HOST_TO_ALL);

  HOST_DISPATCH_RET(reduce<PAIR(T, Op)>(team, dest, source, nreduce));
}

template <typename T>
__host__ void Context::wait_until(T *ivars, int cmp, T val) {
  ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL);

  HOST_DISPATCH(wait_until<T>(ivars, cmp, val));
}

template <typename T>
__host__ size_t Context::wait_until_any(T *ivars, size_t nelems,
                                        const int* status,
                                        int cmp, T val) {
  ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL_ANY);

  HOST_DISPATCH_RET(wait_until_any<T>(ivars, nelems, status, cmp, val));
}

template <typename T>
__host__ void Context::wait_until_all(T *ivars, size_t nelems,
                                      const int* status,
                                      int cmp, T val) {
  ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL_ALL);

  HOST_DISPATCH(wait_until_all<T>(ivars, nelems, status, cmp, val));
}

template <typename T>
__host__ size_t Context::wait_until_some(T *ivars, size_t nelems,
                                         size_t* indices,
                                         const int* status,
                                         int cmp, T val) {
  ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL_SOME);

  HOST_DISPATCH_RET(wait_until_some<T>(ivars, nelems, indices, status, cmp, val));
}

template <typename T>
__host__ void Context::wait_until_all_vector(T *ivars, size_t nelems,
                                             const int *status,
                                             int cmp, T* vals) {
  ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL_ALL_VECTOR);

  HOST_DISPATCH(wait_until_all_vector<T>(ivars, nelems, status, cmp, vals));
}

template <typename T>
__host__ size_t Context::wait_until_any_vector(T *ivars, size_t nelems,
                                               const int *status,
                                               int cmp, T* vals) {
  ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL_ANY_VECTOR);

  HOST_DISPATCH_RET(wait_until_any_vector<T>(ivars, nelems, status, cmp, vals));
}

template <typename T>
__host__ size_t Context::wait_until_some_vector(T *ivars, size_t nelems,
                                                size_t* indices,
                                                const int *status,
                                                int cmp, T* vals) {
  ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL_SOME_VECTOR);

  HOST_DISPATCH_RET(wait_until_some_vector<T>(ivars, nelems, indices, status, cmp, vals));
}

template <typename T>
__host__ int Context::test(T *ivars, int cmp, T val) {
  ctxHostStats.incStat(NUM_HOST_TEST);

  HOST_DISPATCH_RET(test<T>(ivars, cmp, val));
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP_
