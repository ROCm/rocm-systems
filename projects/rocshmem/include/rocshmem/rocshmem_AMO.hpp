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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_AMO_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_AMO_HPP

namespace rocshmem {

/**
 * @name SHMEM_ATOMIC_FETCH
 * @brief Atomically return the value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The value of \p dest.
 */
__device__ ATTR_NO_INLINE float rocshmem_ctx_float_atomic_fetch(
    rocshmem_ctx_t ctx, float *source, int pe);
__device__ ATTR_NO_INLINE float rocshmem_float_atomic_fetch(
    float *source, int pe);
__host__ float rocshmem_ctx_float_atomic_fetch(
    rocshmem_ctx_t ctx, float *source, int pe);
__host__ float rocshmem_float_atomic_fetch(
    float *source, int pe);

__device__ ATTR_NO_INLINE double rocshmem_ctx_double_atomic_fetch(
    rocshmem_ctx_t ctx, double *source, int pe);
__device__ ATTR_NO_INLINE double rocshmem_double_atomic_fetch(
    double *source, int pe);
__host__ double rocshmem_ctx_double_atomic_fetch(
    rocshmem_ctx_t ctx, double *source, int pe);
__host__ double rocshmem_double_atomic_fetch(
    double *source, int pe);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_atomic_fetch(
    rocshmem_ctx_t ctx, int *source, int pe);
__device__ ATTR_NO_INLINE int rocshmem_int_atomic_fetch(
    int *source, int pe);
__host__ int rocshmem_ctx_int_atomic_fetch(
    rocshmem_ctx_t ctx, int *source, int pe);
__host__ int rocshmem_int_atomic_fetch(
    int *source, int pe);

__device__ ATTR_NO_INLINE long rocshmem_ctx_long_atomic_fetch(
    rocshmem_ctx_t ctx, long *source, int pe);
__device__ ATTR_NO_INLINE long rocshmem_long_atomic_fetch(
    long *source, int pe);
__host__ long rocshmem_ctx_long_atomic_fetch(
    rocshmem_ctx_t ctx, long *source, int pe);
__host__ long rocshmem_long_atomic_fetch(
    long *source, int pe);

__device__ ATTR_NO_INLINE long long rocshmem_ctx_longlong_atomic_fetch(
    rocshmem_ctx_t ctx, long long *source, int pe);
__device__ ATTR_NO_INLINE long long rocshmem_longlong_atomic_fetch(
    long long *source, int pe);
__host__ long long rocshmem_ctx_longlong_atomic_fetch(
    rocshmem_ctx_t ctx, long long *source, int pe);
__host__ long long rocshmem_longlong_atomic_fetch(
    long long *source, int pe);

__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_fetch(
    rocshmem_ctx_t ctx, unsigned int *source, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_fetch(
    unsigned int *source, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_fetch(
    rocshmem_ctx_t ctx, unsigned int *source, int pe);
__host__ unsigned int rocshmem_uint_atomic_fetch(
    unsigned int *source, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_fetch(
    rocshmem_ctx_t ctx, unsigned long *source, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_fetch(
    unsigned long *source, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_fetch(
    rocshmem_ctx_t ctx, unsigned long *source, int pe);
__host__ unsigned long rocshmem_ulong_atomic_fetch(
    unsigned long *source, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_fetch(
    rocshmem_ctx_t ctx, unsigned long long *source, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_fetch(
    unsigned long long *source, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_fetch(
    rocshmem_ctx_t ctx, unsigned long long *source, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_fetch(
    unsigned long long *source, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_fetch(
    rocshmem_ctx_t ctx, int32_t *source, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_fetch(
    int32_t *source, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_fetch(
    rocshmem_ctx_t ctx, int32_t *source, int pe);
__host__ int32_t rocshmem_int32_atomic_fetch(
    int32_t *source, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_fetch(
    rocshmem_ctx_t ctx, int64_t *source, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_fetch(
    int64_t *source, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_fetch(
    rocshmem_ctx_t ctx, int64_t *source, int pe);
__host__ int64_t rocshmem_int64_atomic_fetch(
    int64_t *source, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_fetch(
    rocshmem_ctx_t ctx, uint32_t *source, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_fetch(
    uint32_t *source, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_fetch(
    rocshmem_ctx_t ctx, uint32_t *source, int pe);
__host__ uint32_t rocshmem_uint32_atomic_fetch(
    uint32_t *source, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_fetch(
    rocshmem_ctx_t ctx, uint64_t *source, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_fetch(
    uint64_t *source, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_fetch(
    rocshmem_ctx_t ctx, uint64_t *source, int pe);
__host__ uint64_t rocshmem_uint64_atomic_fetch(
    uint64_t *source, int pe);

__device__ ATTR_NO_INLINE size_t rocshmem_ctx_size_atomic_fetch(
    rocshmem_ctx_t ctx, size_t *source, int pe);
__device__ ATTR_NO_INLINE size_t rocshmem_size_atomic_fetch(
    size_t *source, int pe);
__host__ size_t rocshmem_ctx_size_atomic_fetch(
    rocshmem_ctx_t ctx, size_t *source, int pe);
__host__ size_t rocshmem_size_atomic_fetch(
    size_t *source, int pe);

__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ctx_ptrdiff_atomic_fetch(
    rocshmem_ctx_t ctx, ptrdiff_t *source, int pe);
__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ptrdiff_atomic_fetch(
    ptrdiff_t *source, int pe);
__host__ ptrdiff_t rocshmem_ctx_ptrdiff_atomic_fetch(
    rocshmem_ctx_t ctx, ptrdiff_t *source, int pe);
__host__ ptrdiff_t rocshmem_ptrdiff_atomic_fetch(
    ptrdiff_t *source, int pe);


/**
 * @name SHMEM_ATOMIC_SET
 * @brief Atomically set the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_atomic_set(
    rocshmem_ctx_t ctx, float *dest, float value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_atomic_set(
    float *dest, float value, int pe);
__host__ void rocshmem_ctx_float_atomic_set(
    rocshmem_ctx_t ctx, float *dest, float value, int pe);
__host__ void rocshmem_float_atomic_set(
    float *dest, float value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_atomic_set(
    rocshmem_ctx_t ctx, double *dest, double value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_atomic_set(
    double *dest, double value, int pe);
__host__ void rocshmem_ctx_double_atomic_set(
    rocshmem_ctx_t ctx, double *dest, double value, int pe);
__host__ void rocshmem_double_atomic_set(
    double *dest, double value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_atomic_set(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_atomic_set(
    int *dest, int value, int pe);
__host__ void rocshmem_ctx_int_atomic_set(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__host__ void rocshmem_int_atomic_set(
    int *dest, int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_atomic_set(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_atomic_set(
    long *dest, long value, int pe);
__host__ void rocshmem_ctx_long_atomic_set(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__host__ void rocshmem_long_atomic_set(
    long *dest, long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_atomic_set(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_atomic_set(
    long long *dest, long long value, int pe);
__host__ void rocshmem_ctx_longlong_atomic_set(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__host__ void rocshmem_longlong_atomic_set(
    long long *dest, long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_atomic_set(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_atomic_set(
    unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_ctx_uint_atomic_set(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_uint_atomic_set(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_atomic_set(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_atomic_set(
    unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ctx_ulong_atomic_set(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ulong_atomic_set(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_atomic_set(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_atomic_set(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ctx_ulonglong_atomic_set(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ulonglong_atomic_set(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_atomic_set(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_atomic_set(
    int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_ctx_int32_atomic_set(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_int32_atomic_set(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_atomic_set(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_atomic_set(
    int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_ctx_int64_atomic_set(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_int64_atomic_set(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_atomic_set(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_atomic_set(
    uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_ctx_uint32_atomic_set(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_uint32_atomic_set(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_atomic_set(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_atomic_set(
    uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_ctx_uint64_atomic_set(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_uint64_atomic_set(
    uint64_t *dest, uint64_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_atomic_set(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_atomic_set(
    size_t *dest, size_t value, int pe);
__host__ void rocshmem_ctx_size_atomic_set(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__host__ void rocshmem_size_atomic_set(
    size_t *dest, size_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_atomic_set(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_atomic_set(
    ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ void rocshmem_ctx_ptrdiff_atomic_set(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ void rocshmem_ptrdiff_atomic_set(
    ptrdiff_t *dest, ptrdiff_t value, int pe);


/**
 * @name SHMEM_ATOMIC_COMPARE_SWAP
 * @brief Atomically compares if the value in \p dest with \p cond is equal
 * then put \p val in \p dest. The operation returns the older value of \p dest
 * to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] cond    The value to be compare with.
 * @param[in] val     The value to be atomically swapped.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest.
 */
__device__ ATTR_NO_INLINE int rocshmem_ctx_int_atomic_compare_swap(
    rocshmem_ctx_t ctx, int *dest, int cond, int value, int pe);
__device__ ATTR_NO_INLINE int rocshmem_int_atomic_compare_swap(
    int *dest, int cond, int value, int pe);
__host__ int rocshmem_ctx_int_atomic_compare_swap(
    rocshmem_ctx_t ctx, int *dest, int cond, int value, int pe);
__host__ int rocshmem_int_atomic_compare_swap(
    int *dest, int cond, int value, int pe);

__device__ ATTR_NO_INLINE long rocshmem_ctx_long_atomic_compare_swap(
    rocshmem_ctx_t ctx, long *dest, long cond, long value, int pe);
__device__ ATTR_NO_INLINE long rocshmem_long_atomic_compare_swap(
    long *dest, long cond, long value, int pe);
__host__ long rocshmem_ctx_long_atomic_compare_swap(
    rocshmem_ctx_t ctx, long *dest, long cond, long value, int pe);
__host__ long rocshmem_long_atomic_compare_swap(
    long *dest, long cond, long value, int pe);

__device__ ATTR_NO_INLINE long long rocshmem_ctx_longlong_atomic_compare_swap(
    rocshmem_ctx_t ctx, long long *dest, long long cond, long long value, int pe);
__device__ ATTR_NO_INLINE long long rocshmem_longlong_atomic_compare_swap(
    long long *dest, long long cond, long long value, int pe);
__host__ long long rocshmem_ctx_longlong_atomic_compare_swap(
    rocshmem_ctx_t ctx, long long *dest, long long cond, long long value, int pe);
__host__ long long rocshmem_longlong_atomic_compare_swap(
    long long *dest, long long cond, long long value, int pe);

__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_compare_swap(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int cond, unsigned int value, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_compare_swap(
    unsigned int *dest, unsigned int cond, unsigned int value, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_compare_swap(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int cond, unsigned int value, int pe);
__host__ unsigned int rocshmem_uint_atomic_compare_swap(
    unsigned int *dest, unsigned int cond, unsigned int value, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_compare_swap(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long cond, unsigned long value, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_compare_swap(
    unsigned long *dest, unsigned long cond, unsigned long value, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_compare_swap(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long cond, unsigned long value, int pe);
__host__ unsigned long rocshmem_ulong_atomic_compare_swap(
    unsigned long *dest, unsigned long cond, unsigned long value, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_compare_swap(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long cond, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_compare_swap(
    unsigned long long *dest, unsigned long long cond, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_compare_swap(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long cond, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_compare_swap(
    unsigned long long *dest, unsigned long long cond, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_compare_swap(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t cond, int32_t value, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_compare_swap(
    int32_t *dest, int32_t cond, int32_t value, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_compare_swap(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t cond, int32_t value, int pe);
__host__ int32_t rocshmem_int32_atomic_compare_swap(
    int32_t *dest, int32_t cond, int32_t value, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_compare_swap(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t cond, int64_t value, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_compare_swap(
    int64_t *dest, int64_t cond, int64_t value, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_compare_swap(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t cond, int64_t value, int pe);
__host__ int64_t rocshmem_int64_atomic_compare_swap(
    int64_t *dest, int64_t cond, int64_t value, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_compare_swap(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t cond, uint32_t value, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_compare_swap(
    uint32_t *dest, uint32_t cond, uint32_t value, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_compare_swap(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t cond, uint32_t value, int pe);
__host__ uint32_t rocshmem_uint32_atomic_compare_swap(
    uint32_t *dest, uint32_t cond, uint32_t value, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_compare_swap(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t cond, uint64_t value, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_compare_swap(
    uint64_t *dest, uint64_t cond, uint64_t value, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_compare_swap(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t cond, uint64_t value, int pe);
__host__ uint64_t rocshmem_uint64_atomic_compare_swap(
    uint64_t *dest, uint64_t cond, uint64_t value, int pe);

__device__ ATTR_NO_INLINE size_t rocshmem_ctx_size_atomic_compare_swap(
    rocshmem_ctx_t ctx, size_t *dest, size_t cond, size_t value, int pe);
__device__ ATTR_NO_INLINE size_t rocshmem_size_atomic_compare_swap(
    size_t *dest, size_t cond, size_t value, int pe);
__host__ size_t rocshmem_ctx_size_atomic_compare_swap(
    rocshmem_ctx_t ctx, size_t *dest, size_t cond, size_t value, int pe);
__host__ size_t rocshmem_size_atomic_compare_swap(
    size_t *dest, size_t cond, size_t value, int pe);

__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ctx_ptrdiff_atomic_compare_swap(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t cond, ptrdiff_t value, int pe);
__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ptrdiff_atomic_compare_swap(
    ptrdiff_t *dest, ptrdiff_t cond, ptrdiff_t value, int pe);
__host__ ptrdiff_t rocshmem_ctx_ptrdiff_atomic_compare_swap(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t cond, ptrdiff_t value, int pe);
__host__ ptrdiff_t rocshmem_ptrdiff_atomic_compare_swap(
    ptrdiff_t *dest, ptrdiff_t cond, ptrdiff_t value, int pe);


/**
 * @name SHMEM_ATOMIC_SWAP
 * @brief Atomically swap the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */
__device__ ATTR_NO_INLINE float rocshmem_ctx_float_atomic_swap(
    rocshmem_ctx_t ctx, float *dest, float value, int pe);
__device__ ATTR_NO_INLINE float rocshmem_float_atomic_swap(
    float *dest, float value, int pe);
__host__ float rocshmem_ctx_float_atomic_swap(
    rocshmem_ctx_t ctx, float *dest, float value, int pe);
__host__ float rocshmem_float_atomic_swap(
    float *dest, float value, int pe);

__device__ ATTR_NO_INLINE double rocshmem_ctx_double_atomic_swap(
    rocshmem_ctx_t ctx, double *dest, double value, int pe);
__device__ ATTR_NO_INLINE double rocshmem_double_atomic_swap(
    double *dest, double value, int pe);
__host__ double rocshmem_ctx_double_atomic_swap(
    rocshmem_ctx_t ctx, double *dest, double value, int pe);
__host__ double rocshmem_double_atomic_swap(
    double *dest, double value, int pe);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_atomic_swap(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__device__ ATTR_NO_INLINE int rocshmem_int_atomic_swap(
    int *dest, int value, int pe);
__host__ int rocshmem_ctx_int_atomic_swap(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__host__ int rocshmem_int_atomic_swap(
    int *dest, int value, int pe);

__device__ ATTR_NO_INLINE long rocshmem_ctx_long_atomic_swap(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__device__ ATTR_NO_INLINE long rocshmem_long_atomic_swap(
    long *dest, long value, int pe);
__host__ long rocshmem_ctx_long_atomic_swap(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__host__ long rocshmem_long_atomic_swap(
    long *dest, long value, int pe);

__device__ ATTR_NO_INLINE long long rocshmem_ctx_longlong_atomic_swap(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__device__ ATTR_NO_INLINE long long rocshmem_longlong_atomic_swap(
    long long *dest, long long value, int pe);
__host__ long long rocshmem_ctx_longlong_atomic_swap(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__host__ long long rocshmem_longlong_atomic_swap(
    long long *dest, long long value, int pe);

__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_swap(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_swap(
    unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_swap(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_uint_atomic_swap(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_swap(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_swap(
    unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_swap(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ulong_atomic_swap(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_swap(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_swap(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_swap(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_swap(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_swap(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_swap(
    int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_swap(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_int32_atomic_swap(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_swap(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_swap(
    int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_swap(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_int64_atomic_swap(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_swap(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_swap(
    uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_swap(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_uint32_atomic_swap(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_swap(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_swap(
    uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_swap(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_uint64_atomic_swap(
    uint64_t *dest, uint64_t value, int pe);

__device__ ATTR_NO_INLINE size_t rocshmem_ctx_size_atomic_swap(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__device__ ATTR_NO_INLINE size_t rocshmem_size_atomic_swap(
    size_t *dest, size_t value, int pe);
__host__ size_t rocshmem_ctx_size_atomic_swap(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__host__ size_t rocshmem_size_atomic_swap(
    size_t *dest, size_t value, int pe);

__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ctx_ptrdiff_atomic_swap(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ptrdiff_atomic_swap(
    ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ ptrdiff_t rocshmem_ctx_ptrdiff_atomic_swap(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ ptrdiff_t rocshmem_ptrdiff_atomic_swap(
    ptrdiff_t *dest, ptrdiff_t value, int pe);


/**
 * @name SHMEM_ATOMIC_FETCH_INC
 * @brief Atomically add 1 to \p dest on \p pe. The operation
 * returns the older value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest before it was incremented by 1.
 */
__device__ ATTR_NO_INLINE int rocshmem_ctx_int_atomic_fetch_inc(
    rocshmem_ctx_t ctx, int *dest, int pe);
__device__ ATTR_NO_INLINE int rocshmem_int_atomic_fetch_inc(
    int *dest, int pe);
__host__ int rocshmem_ctx_int_atomic_fetch_inc(
    rocshmem_ctx_t ctx, int *dest, int pe);
__host__ int rocshmem_int_atomic_fetch_inc(
    int *dest, int pe);

__device__ ATTR_NO_INLINE long rocshmem_ctx_long_atomic_fetch_inc(
    rocshmem_ctx_t ctx, long *dest, int pe);
__device__ ATTR_NO_INLINE long rocshmem_long_atomic_fetch_inc(
    long *dest, int pe);
__host__ long rocshmem_ctx_long_atomic_fetch_inc(
    rocshmem_ctx_t ctx, long *dest, int pe);
__host__ long rocshmem_long_atomic_fetch_inc(
    long *dest, int pe);

__device__ ATTR_NO_INLINE long long rocshmem_ctx_longlong_atomic_fetch_inc(
    rocshmem_ctx_t ctx, long long *dest, int pe);
__device__ ATTR_NO_INLINE long long rocshmem_longlong_atomic_fetch_inc(
    long long *dest, int pe);
__host__ long long rocshmem_ctx_longlong_atomic_fetch_inc(
    rocshmem_ctx_t ctx, long long *dest, int pe);
__host__ long long rocshmem_longlong_atomic_fetch_inc(
    long long *dest, int pe);

__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_fetch_inc(
    rocshmem_ctx_t ctx, unsigned int *dest, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_fetch_inc(
    unsigned int *dest, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_fetch_inc(
    rocshmem_ctx_t ctx, unsigned int *dest, int pe);
__host__ unsigned int rocshmem_uint_atomic_fetch_inc(
    unsigned int *dest, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_fetch_inc(
    rocshmem_ctx_t ctx, unsigned long *dest, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_fetch_inc(
    unsigned long *dest, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_fetch_inc(
    rocshmem_ctx_t ctx, unsigned long *dest, int pe);
__host__ unsigned long rocshmem_ulong_atomic_fetch_inc(
    unsigned long *dest, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_inc(
    rocshmem_ctx_t ctx, unsigned long long *dest, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_fetch_inc(
    unsigned long long *dest, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_inc(
    rocshmem_ctx_t ctx, unsigned long long *dest, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_fetch_inc(
    unsigned long long *dest, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_fetch_inc(
    rocshmem_ctx_t ctx, int32_t *dest, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_fetch_inc(
    int32_t *dest, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_fetch_inc(
    rocshmem_ctx_t ctx, int32_t *dest, int pe);
__host__ int32_t rocshmem_int32_atomic_fetch_inc(
    int32_t *dest, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_fetch_inc(
    rocshmem_ctx_t ctx, int64_t *dest, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_fetch_inc(
    int64_t *dest, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_fetch_inc(
    rocshmem_ctx_t ctx, int64_t *dest, int pe);
__host__ int64_t rocshmem_int64_atomic_fetch_inc(
    int64_t *dest, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_fetch_inc(
    rocshmem_ctx_t ctx, uint32_t *dest, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_fetch_inc(
    uint32_t *dest, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_fetch_inc(
    rocshmem_ctx_t ctx, uint32_t *dest, int pe);
__host__ uint32_t rocshmem_uint32_atomic_fetch_inc(
    uint32_t *dest, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_fetch_inc(
    rocshmem_ctx_t ctx, uint64_t *dest, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_fetch_inc(
    uint64_t *dest, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_fetch_inc(
    rocshmem_ctx_t ctx, uint64_t *dest, int pe);
__host__ uint64_t rocshmem_uint64_atomic_fetch_inc(
    uint64_t *dest, int pe);

__device__ ATTR_NO_INLINE size_t rocshmem_ctx_size_atomic_fetch_inc(
    rocshmem_ctx_t ctx, size_t *dest, int pe);
__device__ ATTR_NO_INLINE size_t rocshmem_size_atomic_fetch_inc(
    size_t *dest, int pe);
__host__ size_t rocshmem_ctx_size_atomic_fetch_inc(
    rocshmem_ctx_t ctx, size_t *dest, int pe);
__host__ size_t rocshmem_size_atomic_fetch_inc(
    size_t *dest, int pe);

__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ctx_ptrdiff_atomic_fetch_inc(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, int pe);
__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ptrdiff_atomic_fetch_inc(
    ptrdiff_t *dest, int pe);
__host__ ptrdiff_t rocshmem_ctx_ptrdiff_atomic_fetch_inc(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, int pe);
__host__ ptrdiff_t rocshmem_ptrdiff_atomic_fetch_inc(
    ptrdiff_t *dest, int pe);


/**
 * @name SHMEM_ATOMIC_INC
 * @brief Atomically add 1 to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_int_atomic_inc(
    rocshmem_ctx_t ctx, int *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_atomic_inc(
    int *dest, int pe);
__host__ void rocshmem_ctx_int_atomic_inc(
    rocshmem_ctx_t ctx, int *dest, int pe);
__host__ void rocshmem_int_atomic_inc(
    int *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_atomic_inc(
    rocshmem_ctx_t ctx, long *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_atomic_inc(
    long *dest, int pe);
__host__ void rocshmem_ctx_long_atomic_inc(
    rocshmem_ctx_t ctx, long *dest, int pe);
__host__ void rocshmem_long_atomic_inc(
    long *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_atomic_inc(
    rocshmem_ctx_t ctx, long long *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_atomic_inc(
    long long *dest, int pe);
__host__ void rocshmem_ctx_longlong_atomic_inc(
    rocshmem_ctx_t ctx, long long *dest, int pe);
__host__ void rocshmem_longlong_atomic_inc(
    long long *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_atomic_inc(
    rocshmem_ctx_t ctx, unsigned int *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_atomic_inc(
    unsigned int *dest, int pe);
__host__ void rocshmem_ctx_uint_atomic_inc(
    rocshmem_ctx_t ctx, unsigned int *dest, int pe);
__host__ void rocshmem_uint_atomic_inc(
    unsigned int *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_atomic_inc(
    rocshmem_ctx_t ctx, unsigned long *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_atomic_inc(
    unsigned long *dest, int pe);
__host__ void rocshmem_ctx_ulong_atomic_inc(
    rocshmem_ctx_t ctx, unsigned long *dest, int pe);
__host__ void rocshmem_ulong_atomic_inc(
    unsigned long *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_atomic_inc(
    rocshmem_ctx_t ctx, unsigned long long *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_atomic_inc(
    unsigned long long *dest, int pe);
__host__ void rocshmem_ctx_ulonglong_atomic_inc(
    rocshmem_ctx_t ctx, unsigned long long *dest, int pe);
__host__ void rocshmem_ulonglong_atomic_inc(
    unsigned long long *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_atomic_inc(
    rocshmem_ctx_t ctx, int32_t *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_atomic_inc(
    int32_t *dest, int pe);
__host__ void rocshmem_ctx_int32_atomic_inc(
    rocshmem_ctx_t ctx, int32_t *dest, int pe);
__host__ void rocshmem_int32_atomic_inc(
    int32_t *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_atomic_inc(
    rocshmem_ctx_t ctx, int64_t *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_atomic_inc(
    int64_t *dest, int pe);
__host__ void rocshmem_ctx_int64_atomic_inc(
    rocshmem_ctx_t ctx, int64_t *dest, int pe);
__host__ void rocshmem_int64_atomic_inc(
    int64_t *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_atomic_inc(
    rocshmem_ctx_t ctx, uint32_t *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_atomic_inc(
    uint32_t *dest, int pe);
__host__ void rocshmem_ctx_uint32_atomic_inc(
    rocshmem_ctx_t ctx, uint32_t *dest, int pe);
__host__ void rocshmem_uint32_atomic_inc(
    uint32_t *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_atomic_inc(
    rocshmem_ctx_t ctx, uint64_t *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_atomic_inc(
    uint64_t *dest, int pe);
__host__ void rocshmem_ctx_uint64_atomic_inc(
    rocshmem_ctx_t ctx, uint64_t *dest, int pe);
__host__ void rocshmem_uint64_atomic_inc(
    uint64_t *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_atomic_inc(
    rocshmem_ctx_t ctx, size_t *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_atomic_inc(
    size_t *dest, int pe);
__host__ void rocshmem_ctx_size_atomic_inc(
    rocshmem_ctx_t ctx, size_t *dest, int pe);
__host__ void rocshmem_size_atomic_inc(
    size_t *dest, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_atomic_inc(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_atomic_inc(
    ptrdiff_t *dest, int pe);
__host__ void rocshmem_ctx_ptrdiff_atomic_inc(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, int pe);
__host__ void rocshmem_ptrdiff_atomic_inc(
    ptrdiff_t *dest, int pe);


/**
 * @name SHMEM_ATOMIC_FETCH_ADD
 * @brief Atomically add the value \p val to \p dest on \p pe. The operation
 * returns the older value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest before the \p val was added.
 */
__device__ ATTR_NO_INLINE int rocshmem_ctx_int_atomic_fetch_add(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__device__ ATTR_NO_INLINE int rocshmem_int_atomic_fetch_add(
    int *dest, int value, int pe);
__host__ int rocshmem_ctx_int_atomic_fetch_add(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__host__ int rocshmem_int_atomic_fetch_add(
    int *dest, int value, int pe);

__device__ ATTR_NO_INLINE long rocshmem_ctx_long_atomic_fetch_add(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__device__ ATTR_NO_INLINE long rocshmem_long_atomic_fetch_add(
    long *dest, long value, int pe);
__host__ long rocshmem_ctx_long_atomic_fetch_add(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__host__ long rocshmem_long_atomic_fetch_add(
    long *dest, long value, int pe);

__device__ ATTR_NO_INLINE long long rocshmem_ctx_longlong_atomic_fetch_add(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__device__ ATTR_NO_INLINE long long rocshmem_longlong_atomic_fetch_add(
    long long *dest, long long value, int pe);
__host__ long long rocshmem_ctx_longlong_atomic_fetch_add(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__host__ long long rocshmem_longlong_atomic_fetch_add(
    long long *dest, long long value, int pe);

__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_fetch_add(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_fetch_add(
    unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_fetch_add(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_uint_atomic_fetch_add(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_fetch_add(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_fetch_add(
    unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_fetch_add(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ulong_atomic_fetch_add(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_add(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_fetch_add(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_add(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_fetch_add(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_fetch_add(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_fetch_add(
    int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_fetch_add(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_int32_atomic_fetch_add(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_fetch_add(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_fetch_add(
    int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_fetch_add(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_int64_atomic_fetch_add(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_fetch_add(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_fetch_add(
    uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_fetch_add(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_uint32_atomic_fetch_add(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_fetch_add(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_fetch_add(
    uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_fetch_add(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_uint64_atomic_fetch_add(
    uint64_t *dest, uint64_t value, int pe);

__device__ ATTR_NO_INLINE size_t rocshmem_ctx_size_atomic_fetch_add(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__device__ ATTR_NO_INLINE size_t rocshmem_size_atomic_fetch_add(
    size_t *dest, size_t value, int pe);
__host__ size_t rocshmem_ctx_size_atomic_fetch_add(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__host__ size_t rocshmem_size_atomic_fetch_add(
    size_t *dest, size_t value, int pe);

__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ctx_ptrdiff_atomic_fetch_add(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__device__ ATTR_NO_INLINE ptrdiff_t rocshmem_ptrdiff_atomic_fetch_add(
    ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ ptrdiff_t rocshmem_ctx_ptrdiff_atomic_fetch_add(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ ptrdiff_t rocshmem_ptrdiff_atomic_fetch_add(
    ptrdiff_t *dest, ptrdiff_t value, int pe);


/**
 * @name SHMEM_ATOMIC_ADD
 * @brief Atomically add the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_int_atomic_add(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_atomic_add(
    int *dest, int value, int pe);
__host__ void rocshmem_ctx_int_atomic_add(
    rocshmem_ctx_t ctx, int *dest, int value, int pe);
__host__ void rocshmem_int_atomic_add(
    int *dest, int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_atomic_add(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_atomic_add(
    long *dest, long value, int pe);
__host__ void rocshmem_ctx_long_atomic_add(
    rocshmem_ctx_t ctx, long *dest, long value, int pe);
__host__ void rocshmem_long_atomic_add(
    long *dest, long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_atomic_add(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_atomic_add(
    long long *dest, long long value, int pe);
__host__ void rocshmem_ctx_longlong_atomic_add(
    rocshmem_ctx_t ctx, long long *dest, long long value, int pe);
__host__ void rocshmem_longlong_atomic_add(
    long long *dest, long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_atomic_add(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_atomic_add(
    unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_ctx_uint_atomic_add(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_uint_atomic_add(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_atomic_add(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_atomic_add(
    unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ctx_ulong_atomic_add(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ulong_atomic_add(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_atomic_add(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_atomic_add(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ctx_ulonglong_atomic_add(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ulonglong_atomic_add(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_atomic_add(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_atomic_add(
    int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_ctx_int32_atomic_add(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_int32_atomic_add(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_atomic_add(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_atomic_add(
    int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_ctx_int64_atomic_add(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_int64_atomic_add(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_atomic_add(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_atomic_add(
    uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_ctx_uint32_atomic_add(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_uint32_atomic_add(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_atomic_add(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_atomic_add(
    uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_ctx_uint64_atomic_add(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_uint64_atomic_add(
    uint64_t *dest, uint64_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_atomic_add(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_atomic_add(
    size_t *dest, size_t value, int pe);
__host__ void rocshmem_ctx_size_atomic_add(
    rocshmem_ctx_t ctx, size_t *dest, size_t value, int pe);
__host__ void rocshmem_size_atomic_add(
    size_t *dest, size_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_atomic_add(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_atomic_add(
    ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ void rocshmem_ctx_ptrdiff_atomic_add(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, ptrdiff_t value, int pe);
__host__ void rocshmem_ptrdiff_atomic_add(
    ptrdiff_t *dest, ptrdiff_t value, int pe);


/**
 * @name SHMEM_ATOMIC_FETCH_AND
 * @brief Atomically bitwise-and the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */
__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_fetch_and(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_fetch_and(
    unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_fetch_and(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_uint_atomic_fetch_and(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_fetch_and(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_fetch_and(
    unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_fetch_and(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ulong_atomic_fetch_and(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_and(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_fetch_and(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_and(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_fetch_and(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_fetch_and(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_fetch_and(
    int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_fetch_and(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_int32_atomic_fetch_and(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_fetch_and(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_fetch_and(
    int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_fetch_and(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_int64_atomic_fetch_and(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_fetch_and(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_fetch_and(
    uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_fetch_and(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_uint32_atomic_fetch_and(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_fetch_and(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_fetch_and(
    uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_fetch_and(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_uint64_atomic_fetch_and(
    uint64_t *dest, uint64_t value, int pe);


/**
 * @name SHMEM_ATOMIC_AND
 * @brief Atomically bitwise-and the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_atomic_and(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_atomic_and(
    unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_ctx_uint_atomic_and(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_uint_atomic_and(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_atomic_and(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_atomic_and(
    unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ctx_ulong_atomic_and(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ulong_atomic_and(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_atomic_and(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_atomic_and(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ctx_ulonglong_atomic_and(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ulonglong_atomic_and(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_atomic_and(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_atomic_and(
    int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_ctx_int32_atomic_and(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_int32_atomic_and(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_atomic_and(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_atomic_and(
    int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_ctx_int64_atomic_and(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_int64_atomic_and(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_atomic_and(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_atomic_and(
    uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_ctx_uint32_atomic_and(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_uint32_atomic_and(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_atomic_and(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_atomic_and(
    uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_ctx_uint64_atomic_and(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_uint64_atomic_and(
    uint64_t *dest, uint64_t value, int pe);


/**
 * @name SHMEM_ATOMIC_FETCH_OR
 * @brief Atomically bitwise-or the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */
__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_fetch_or(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_fetch_or(
    unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_fetch_or(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_uint_atomic_fetch_or(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_fetch_or(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_fetch_or(
    unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_fetch_or(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ulong_atomic_fetch_or(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_or(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_fetch_or(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_or(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_fetch_or(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_fetch_or(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_fetch_or(
    int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_fetch_or(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_int32_atomic_fetch_or(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_fetch_or(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_fetch_or(
    int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_fetch_or(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_int64_atomic_fetch_or(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_fetch_or(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_fetch_or(
    uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_fetch_or(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_uint32_atomic_fetch_or(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_fetch_or(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_fetch_or(
    uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_fetch_or(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_uint64_atomic_fetch_or(
    uint64_t *dest, uint64_t value, int pe);


/**
 * @name SHMEM_ATOMIC_OR
 * @brief Atomically bitwise-or the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_atomic_or(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_atomic_or(
    unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_ctx_uint_atomic_or(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_uint_atomic_or(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_atomic_or(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_atomic_or(
    unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ctx_ulong_atomic_or(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ulong_atomic_or(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_atomic_or(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_atomic_or(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ctx_ulonglong_atomic_or(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ulonglong_atomic_or(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_atomic_or(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_atomic_or(
    int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_ctx_int32_atomic_or(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_int32_atomic_or(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_atomic_or(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_atomic_or(
    int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_ctx_int64_atomic_or(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_int64_atomic_or(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_atomic_or(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_atomic_or(
    uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_ctx_uint32_atomic_or(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_uint32_atomic_or(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_atomic_or(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_atomic_or(
    uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_ctx_uint64_atomic_or(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_uint64_atomic_or(
    uint64_t *dest, uint64_t value, int pe);


/**
 * @name SHMEM_ATOMIC_FETCH_XOR
 * @brief Atomically bitwise-xor the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return original value
 */
__device__ ATTR_NO_INLINE unsigned int rocshmem_ctx_uint_atomic_fetch_xor(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE unsigned int rocshmem_uint_atomic_fetch_xor(
    unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_ctx_uint_atomic_fetch_xor(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ unsigned int rocshmem_uint_atomic_fetch_xor(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE unsigned long rocshmem_ctx_ulong_atomic_fetch_xor(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE unsigned long rocshmem_ulong_atomic_fetch_xor(
    unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ctx_ulong_atomic_fetch_xor(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ unsigned long rocshmem_ulong_atomic_fetch_xor(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_xor(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE unsigned long long rocshmem_ulonglong_atomic_fetch_xor(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ctx_ulonglong_atomic_fetch_xor(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ unsigned long long rocshmem_ulonglong_atomic_fetch_xor(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE int32_t rocshmem_ctx_int32_atomic_fetch_xor(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE int32_t rocshmem_int32_atomic_fetch_xor(
    int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_ctx_int32_atomic_fetch_xor(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ int32_t rocshmem_int32_atomic_fetch_xor(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE int64_t rocshmem_ctx_int64_atomic_fetch_xor(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE int64_t rocshmem_int64_atomic_fetch_xor(
    int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_ctx_int64_atomic_fetch_xor(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ int64_t rocshmem_int64_atomic_fetch_xor(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE uint32_t rocshmem_ctx_uint32_atomic_fetch_xor(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE uint32_t rocshmem_uint32_atomic_fetch_xor(
    uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_ctx_uint32_atomic_fetch_xor(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ uint32_t rocshmem_uint32_atomic_fetch_xor(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE uint64_t rocshmem_ctx_uint64_atomic_fetch_xor(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE uint64_t rocshmem_uint64_atomic_fetch_xor(
    uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_ctx_uint64_atomic_fetch_xor(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ uint64_t rocshmem_uint64_atomic_fetch_xor(
    uint64_t *dest, uint64_t value, int pe);


/**
 * @name SHMEM_ATOMIC_XOR
 * @brief Atomically bitwise-xor the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_atomic_xor(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_atomic_xor(
    unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_ctx_uint_atomic_xor(
    rocshmem_ctx_t ctx, unsigned int *dest, unsigned int value, int pe);
__host__ void rocshmem_uint_atomic_xor(
    unsigned int *dest, unsigned int value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_atomic_xor(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_atomic_xor(
    unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ctx_ulong_atomic_xor(
    rocshmem_ctx_t ctx, unsigned long *dest, unsigned long value, int pe);
__host__ void rocshmem_ulong_atomic_xor(
    unsigned long *dest, unsigned long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_atomic_xor(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_atomic_xor(
    unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ctx_ulonglong_atomic_xor(
    rocshmem_ctx_t ctx, unsigned long long *dest, unsigned long long value, int pe);
__host__ void rocshmem_ulonglong_atomic_xor(
    unsigned long long *dest, unsigned long long value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_atomic_xor(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_atomic_xor(
    int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_ctx_int32_atomic_xor(
    rocshmem_ctx_t ctx, int32_t *dest, int32_t value, int pe);
__host__ void rocshmem_int32_atomic_xor(
    int32_t *dest, int32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_atomic_xor(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_atomic_xor(
    int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_ctx_int64_atomic_xor(
    rocshmem_ctx_t ctx, int64_t *dest, int64_t value, int pe);
__host__ void rocshmem_int64_atomic_xor(
    int64_t *dest, int64_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_atomic_xor(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_atomic_xor(
    uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_ctx_uint32_atomic_xor(
    rocshmem_ctx_t ctx, uint32_t *dest, uint32_t value, int pe);
__host__ void rocshmem_uint32_atomic_xor(
    uint32_t *dest, uint32_t value, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_atomic_xor(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_atomic_xor(
    uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_ctx_uint64_atomic_xor(
    rocshmem_ctx_t ctx, uint64_t *dest, uint64_t value, int pe);
__host__ void rocshmem_uint64_atomic_xor(
    uint64_t *dest, uint64_t value, int pe);


}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_AMO_HPP
