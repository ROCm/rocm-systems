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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_RMA_X_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_RMA_X_HPP

namespace rocshmem {

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must collectively participate
 * in the call using the same arguments
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_wave(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_wave(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_wave(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_wave(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_wave(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_wave(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_wave(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_wave(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_wave(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_wave(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_wave(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_wave(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_wave(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_wave(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_wave(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_wave(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_wave(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_wave(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_wave(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_wave(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_wave(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_wave(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_wave(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_wave(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_wave(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_wave(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * (WG) granularity. However, All threads in a WG must collectively participate
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
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_wg(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_wg(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_wg(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_wg(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_wg(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_wg(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_wg(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_wg(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_wg(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_wg(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_wg(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_wg(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_wg(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_wg(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_wg(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_wg(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_wg(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_wg(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_wg(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_wg(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_wg(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_wg(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_wg(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_wg(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_wg(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_wg(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must participate in the
 * call using the same parameters.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_wave(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_wave(void *dest,
                                                     const void *source,
                                                     size_t nelems, int pe);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * (WG) granularity. However, all threads in the workgroup must participate in
 * the call using the same parameters.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_wg(rocshmem_ctx_t ctx,
                                                       void *dest,
                                                       const void *source,
                                                       size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_wg(void *dest,
                                                   const void *source,
                                                   size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However,  all threads in the wave must participate in the
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
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_get_wave(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_get_wave(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_get_wave(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_get_wave(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_get_wave(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_get_wave(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_get_wave(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_get_wave(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_get_wave(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_get_wave(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_get_wave(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_get_wave(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_get_wave(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_get_wave(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_get_wave(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_get_wave(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_get_wave(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_get_wave(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_get_wave(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_get_wave(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_get_wave(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_get_wave(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_get_wave(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_get_wave(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_get_wave(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_get_wave(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However,  all threads in the workgroup must participate in
 * the call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_get_wg(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_get_wg(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_get_wg(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_get_wg(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_get_wg(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_get_wg(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_get_wg(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_get_wg(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_get_wg(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_get_wg(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_get_wg(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_get_wg(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_get_wg(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_get_wg(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_get_wg(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_get_wg(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_get_wg(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_get_wg(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_get_wg(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_get_wg(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_get_wg(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_get_wg(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_get_wg(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_get_wg(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_get_wg(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_get_wg(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a the wave must participate in the
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
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_getmem_wave(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_getmem_wave(void *dest,
                                                     const void *source,
                                                     size_t nelems, int pe);

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-workgroup
 * (WG) granularity. However, all threads in the workgroup must participate
 * in the call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_getmem_wg(rocshmem_ctx_t ctx,
                                                       void *dest,
                                                       const void *source,
                                                       size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_getmem_wg(void *dest,
                                                   const void *source,
                                                   size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_nbi_wave(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_nbi_wave(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_nbi_wave(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_nbi_wave(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_nbi_wave(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_nbi_wave(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_nbi_wave(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_nbi_wave(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_nbi_wave(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_nbi_wave(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_nbi_wave(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_nbi_wave(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_nbi_wave(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_nbi_wave(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_nbi_wave(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_nbi_wave(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_nbi_wave(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_nbi_wave(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_nbi_wave(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_nbi_wave(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_nbi_wave(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_nbi_wave(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_nbi_wave(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_nbi_wave(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_nbi_wave(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_nbi_wave(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the sameo
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_nbi_wg(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_nbi_wg(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_nbi_wg(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_nbi_wg(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_nbi_wg(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_nbi_wg(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_nbi_wg(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_nbi_wg(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_nbi_wg(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_nbi_wg(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_nbi_wg(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_nbi_wg(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_nbi_wg(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_nbi_wg(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_nbi_wg(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_nbi_wg(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_nbi_wg(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_nbi_wg(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_nbi_wg(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_nbi_wg(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_nbi_wg(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_nbi_wg(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_nbi_wg(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_nbi_wg(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_nbi_wg(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_nbi_wg(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must call in with the same
 * parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_nbi_wave(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_nbi_wave(void *dest,
                                                         const void *source,
                                                         size_t nelems,
                                                         int pe);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in a WG must call in with the same
 * parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_nbi_wg(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_nbi_wg(void *dest,
                                                       const void *source,
                                                       size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_get_nbi_wave(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_get_nbi_wave(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_get_nbi_wave(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_get_nbi_wave(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_get_nbi_wave(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_get_nbi_wave(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_get_nbi_wave(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_get_nbi_wave(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_get_nbi_wave(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_get_nbi_wave(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_get_nbi_wave(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_get_nbi_wave(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_get_nbi_wave(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_get_nbi_wave(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_get_nbi_wave(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_get_nbi_wave(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_get_nbi_wave(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_get_nbi_wave(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_get_nbi_wave(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_get_nbi_wave(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_get_nbi_wave(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_get_nbi_wave(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_get_nbi_wave(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_get_nbi_wave(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_get_nbi_wave(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_get_nbi_wave(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_get_nbi_wg(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_get_nbi_wg(
    float *dest, const float *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_get_nbi_wg(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_get_nbi_wg(
    double *dest, const double *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_get_nbi_wg(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_get_nbi_wg(
    char *dest, const char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_get_nbi_wg(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_get_nbi_wg(
    signed char *dest, const signed char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_get_nbi_wg(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_get_nbi_wg(
    short *dest, const short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_get_nbi_wg(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_get_nbi_wg(
    int *dest, const int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_get_nbi_wg(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_get_nbi_wg(
    long *dest, const long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_get_nbi_wg(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_get_nbi_wg(
    long long *dest, const long long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_get_nbi_wg(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_get_nbi_wg(
    unsigned char *dest, const unsigned char *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_get_nbi_wg(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_get_nbi_wg(
    unsigned short *dest, const unsigned short *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_get_nbi_wg(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_get_nbi_wg(
    unsigned int *dest, const unsigned int *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_get_nbi_wg(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_get_nbi_wg(
    unsigned long *dest, const unsigned long *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_get_nbi_wg(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    size_t nelems, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_get_nbi_wg(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, int pe);


/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_getmem_nbi_wave(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_getmem_nbi_wave(void *dest,
                                                         const void *source,
                                                         size_t nelems,
                                                         int pe);

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_getmem_nbi_wg(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe);

__device__ ATTR_NO_INLINE void rocshmem_getmem_nbi_wg(void *dest,
                                                       const void *source,
                                                       size_t nelems, int pe);


}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_RMA_X_HPP
