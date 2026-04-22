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

#ifndef LIBRARY_SRC_TEMPLATES_HOST_HPP_
#define LIBRARY_SRC_TEMPLATES_HOST_HPP_

#include "rocshmem/rocshmem.hpp"

/**
 * @file templates_host.hpp
 * @brief Internal header that declares templates for rocSHMEM's implementation
 * of the user-facing host APIs.
 *
 * This file contains templates for the OpenSHMEM APIs that take have
 * hardcoded data types into the function name.
 */

/******************************************************************************
 **************************** HOST FUNCTIONS **********************************
 *****************************************************************************/

namespace rocshmem {

template <typename T>
__host__ void rocshmem_put(rocshmem_ctx_t ctx, T *dest, const T *source,
                            size_t nelems, int pe);

template <typename T>
__host__ void rocshmem_put(T *dest, const T *source, size_t nelems, int pe);

template <typename T>
__host__ void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe);

template <typename T>
__host__ void rocshmem_p(T *dest, T value, int pe);

template <typename T>
__host__ void rocshmem_get(rocshmem_ctx_t ctx, T *dest, const T *source,
                            size_t nelems, int pe);

template <typename T>
__host__ void rocshmem_get(T *dest, const T *source, size_t nelems, int pe);

template <typename T>
__host__ T rocshmem_g(rocshmem_ctx_t ctx, const T *source, int pe);

template <typename T>
__host__ T rocshmem_g(const T *source, int pe);

template <typename T>
__host__ void rocshmem_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *src,
                                size_t nelems, int pe);

template <typename T>
__host__ void rocshmem_put_nbi(T *dest, const T *src, size_t nelems, int pe);

template <typename T>
__host__ void rocshmem_get_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                size_t nelems, int pe);

template <typename T>
__host__ void rocshmem_get_nbi(T *dest, const T *source, size_t nelems,
                                int pe);

template <typename T>
__host__ T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val,
                                      int pe);

template <typename T>
__host__ T rocshmem_atomic_fetch_add(T *dest, T val, int pe);

template <typename T>
__host__ T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond,
                                         T val, int pe);

template <typename T>
__host__ T rocshmem_atomic_compare_swap(T *dest, T cond, T val, int pe);

template <typename T>
__host__ T rocshmem_atomic_fetch_inc(rocshmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__host__ T rocshmem_atomic_fetch_inc(T *dest, int pe);

template <typename T>
__host__ T rocshmem_atomic_fetch(rocshmem_ctx_t ctx, T *source, int pe);

template <typename T>
__host__ T rocshmem_atomic_fetch(T *source, int pe);

template <typename T>
__host__ void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val, int pe);

template <typename T>
__host__ void rocshmem_atomic_add(T *dest, T val, int pe);

template <typename T>
__host__ void rocshmem_atomic_inc(rocshmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__host__ void rocshmem_atomic_inc(T *dest, int pe);

template <typename T>
__host__ void rocshmem_atomic_set(T *dest, T val, int pe);

template <typename T>
__host__ void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T val, int pe);

template <typename T>
__host__ void rocshmem_broadcast(rocshmem_ctx_t ctx, T *dest, const T *source,
                                  int nelement, int PE_root, int PE_start,
                                  int logPE_stride, int PE_size, long *pSync);

template <typename T, ROCSHMEM_OP Op>
__host__ void rocshmem_to_all(rocshmem_ctx_t ctx, T *dest, const T *source,
                               int nreduce, int PE_start, int logPE_stride,
                               int PE_size, T *pWrk, long *pSync);

template <typename T>
__host__ void rocshmem_wait_until(T *ivars, int cmp, T val);

template <typename T>
__host__ void wait_until_all(T* ivars, size_t nelems, const int *status,
                             int cmp, T val);

template <typename T>
__host__ size_t wait_until_any(T* ivars, size_t nelems, const int *status,
                               int cmp, T val);

template <typename T>
__host__ size_t wait_until_some(T* ivars, size_t nelems, size_t* indices,
                              const int *status, int cmp, T val);

template <typename T>
__host__ void wait_until_all_vector(T* ivars, size_t nelems, const int *status,
                                    int cmp, T* vals);

template <typename T>
__host__ size_t wait_until_any_vector(T* ivars, size_t nelems, const int *status,
                                      int cmp, T* vals);

template <typename T>
__host__ size_t wait_until_some_vector(T* ivars, size_t nelems,
                                     size_t* indices, const int *status,
                                     int cmp, T* vals);

template <typename T>
__host__ int rocshmem_test(T *ivars, int cmp, T val);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TEMPLATES_HOST_HPP_
