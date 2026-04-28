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

#ifndef LIBRARY_SRC_TEMPLATES_HPP_
#define LIBRARY_SRC_TEMPLATES_HPP_

#include "rocshmem/rocshmem.hpp"

/**
 * @file templates.hpp
 * @brief Internal header that declares templates for rocSHMEM's implentation
 * of the user-facing device APIs.
 *
 * This file contains templates for the OpenSHMEM APIs that take have
 * hardcoded data types into the function name.
 */

/******************************************************************************
 **************************** DEVICE FUNCTIONS ********************************
 *****************************************************************************/

namespace rocshmem {

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_put(rocshmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_put(T *dest, const T *source, size_t nelems, int pe);

/**
 * @brief Writes a single value to \p dest at \p pe PE to \p dst at \p pe.
 * The caller must call into rocshmem_quiet() if remote completion is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] value  Value to write to dest at \p pe.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_p(rocshmem_ctx_t ctx, T *dest, T value, int pe);

template <typename T>
__device__ void rocshmem_p(T *dest, T value, int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_get(rocshmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_get(T *dest, const T *source, size_t nelems, int pe);

/**
 * @brief reads and returns single value from \p source at \p pe.
 * The calling work-group/thread will block until the operation completes.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] source sourcen address. Must be an address on the symmetric
 *                   heap.
 * @param[in] pe     PE of the remote process.
 *
 * @return the value read from remote \p source at \p pe.
 *
 */
template <typename T>
__device__ T rocshmem_g(rocshmem_ctx_t ctx, const T *source, int pe);

template <typename T>
__device__ T rocshmem_g(const T *source, int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_put_nbi(rocshmem_ctx_t ctx, T *dest, const T *src,
                                  size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_put_nbi(T *dest, const T *src, size_t nelems, int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_get_nbi(rocshmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_get_nbi(T *dest, const T *source, size_t nelems,
                                  int pe);

/**
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
 *
 */
template <typename T>
__device__ T rocshmem_atomic_fetch_add(rocshmem_ctx_t ctx, T *dest, T val,
                                        int pe);

template <typename T>
__device__ T rocshmem_atomic_fetch_add(T *dest, T val, int pe);

/**
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
 *
 */
template <typename T>
__device__ T rocshmem_atomic_compare_swap(rocshmem_ctx_t ctx, T *dest, T cond,
                                           T val, int pe);

template <typename T>
__device__ T rocshmem_atomic_compare_swap(T *dest, T cond, T val, int pe);

/**
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
 *
 */
template <typename T>
__device__ T rocshmem_atomic_fetch_inc(rocshmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__device__ T rocshmem_atomic_fetch_inc(T *dest, int pe);

/**
 * @brief Atomically return the value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] source  Source address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The value of \p dest.
 *
 */
template <typename T>
__device__ T rocshmem_atomic_fetch(rocshmem_ctx_t ctx, T *source, int pe);

template <typename T>
__device__ T rocshmem_atomic_fetch(T *source, int pe);

/**
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
 *
 */
template <typename T>
__device__ void rocshmem_atomic_add(rocshmem_ctx_t ctx, T *dest, T val,
                                     int pe);

template <typename T>
__device__ void rocshmem_atomic_add(T *dest, T val, int pe);

/**
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
 *
 */
template <typename T>
__device__ void rocshmem_atomic_inc(rocshmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__device__ void rocshmem_atomic_inc(T *dest, int pe);

/**
 * @brief Atomically set value for \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] value   Value to set.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 *
 */
template <typename T>
__device__ void rocshmem_atomic_set(rocshmem_ctx_t ctx, T *dest, T value,
                                     int pe);

template <typename T>
__device__ void rocshmem_atomic_set(T *dest, T value, int pe);

/**
 * @brief Block the caller until the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ivars Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return void
 *
 */
template <typename T>
__device__ void rocshmem_wait_until(T *ivars, int cmp, T val);

/**
 * @brief test if the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * rocSHMEM function.
 *
 * @param[in] ivars Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return 1 if the evaluation is true else 0
 *
 */
template <typename T>
__device__ int rocshmem_test(T *ivars, int cmp, T val);

/**
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcase completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelement     Size of the buffer to participate in the broadcast.
 * @param[in] PE_root      Zero-based ordinal of the PE, with respect to the
                           active set, from which the data is copied
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pSync        Temporary sync buffer provided to rocSHMEM. Must
                           be of size at least ROCSHMEM_REDUCE_SYNC_SIZE.
 *
 * @return void
 *
 */
template <typename T>
__device__ void rocshmem_wg_broadcast(rocshmem_ctx_t ctx, T *dest,
                                       const T *source, int nelement,
                                       int PE_root, int PE_start,
                                       int logPE_stride, int PE_size,
                                       long *pSync);

/**
 * @brief Perform an allreduce between PEs in the active set. The caller
 * is blocked until the reduction completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nreduce      Size of the buffer to participate in the reduction.
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pWrk         Temporary work buffer provided to rocSHMEM. Must
 *                         be of size at least max(size/2 + 1,
                           ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE).
 * @param[in] pSync        Temporary sync buffer provided to rocSHMEM. Must
                           be of size at least ROCSHMEM_REDUCE_SYNC_SIZE.
 * @param[in] handle       GPU side handle.
 *
 * @return void
 *
 */
template <typename T, ROCSHMEM_OP Op>
__device__ void rocshmem_wg_to_all(rocshmem_ctx_t ctx, T *dest,
                                    const T *source, int nreduce, int PE_start,
                                    int logPE_stride, int PE_size, T *pWrk,
                                    long *pSync);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must collectivlily participate in
 * the call using the same arguments
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_put_wave(rocshmem_ctx_t ctx, T *dest,
                                    const T *source, size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_put_wave(T *dest, const T *source, size_t nelems,
                                    int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-workgroub
 * (WG) granularity. However, All threads in a WG must collectivelly participate
 * in the call using the same arguments.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_put_wg(rocshmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_put_wg(T *dest, const T *source, size_t nelems,
                                  int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However,  all threads in a the wave must participate in the
 * call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_get_wave(rocshmem_ctx_t ctx, T *dest,
                                    const T *source, size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_get_wave(T *dest, const T *source, size_t nelems,
                                    int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However,  all threads in a the workgroup must participate in the
 * call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_get_wg(rocshmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_get_wg(T *dest, const T *source, size_t nelems,
                                  int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same args
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_put_nbi_wave(rocshmem_ctx_t ctx, T *dest,
                                        const T *src, size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_put_nbi_wave(T *dest, const T *src, size_t nelems,
                                        int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the same args
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_put_nbi_wg(rocshmem_ctx_t ctx, T *dest,
                                      const T *src, size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_put_nbi_wg(T *dest, const T *src, size_t nelems,
                                      int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same args
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_get_nbi_wave(rocshmem_ctx_t ctx, T *dest,
                                        const T *source, size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_get_nbi_wave(T *dest, const T *source, size_t nelems,
                                        int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the same args
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void rocshmem_get_nbi_wg(rocshmem_ctx_t ctx, T *dest,
                                      const T *source, size_t nelems, int pe);

template <typename T>
__device__ void rocshmem_get_nbi_wg(T *dest, const T *source, size_t nelems,
                                      int pe);

__device__ void rocshmem_putmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  rocshmem_ctx_putmem_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  rocshmem_ctx_putmem_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_wave(T *dest, const T *source, size_t nelems,
                                    int pe) {
  rocshmem_put_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_wg(T *dest, const T *source, size_t nelems,
                                  int pe) {
  rocshmem_put_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  rocshmem_ctx_getmem_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_wg(T *dest, const T *source, size_t nelems,
                                  int pe) {
  rocshmem_get_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_getmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  rocshmem_ctx_getmem_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_wave(T *dest, const T *source, size_t nelems,
                                    int pe) {
  rocshmem_get_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_putmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  rocshmem_ctx_putmem_nbi_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_put_nbi_wg(T *dest, const T *source, size_t nelems,
                                      int pe) {
  rocshmem_put_nbi_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_putmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  rocshmem_ctx_putmem_nbi_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems,
                                 pe);
}

template <typename T>
__device__ void rocshmem_put_nbi_wave(T *dest, const T *source, size_t nelems,
                                        int pe) {
  rocshmem_put_nbi_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_getmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  rocshmem_ctx_getmem_nbi_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void rocshmem_get_nbi_wg(T *dest, const T *source, size_t nelems,
                                      int pe) {
  rocshmem_get_nbi_wg(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void rocshmem_getmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  rocshmem_ctx_getmem_nbi_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems,
                                 pe);
}

template <typename T>
__device__ void rocshmem_get_nbi_wave(T *dest, const T *source, size_t nelems,
                                        int pe) {
  rocshmem_get_nbi_wave(ROCSHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_TEMPLATES_HPP_
