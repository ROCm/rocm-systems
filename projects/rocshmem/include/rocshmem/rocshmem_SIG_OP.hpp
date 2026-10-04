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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_SIG_OP_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_SIG_OP_HPP

namespace rocshmem {
__device__ ATTR_NO_INLINE void rocshmem_putmem_signal(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_signal(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_signal(
    rocshmem_ctx_t ctx, float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_signal(
    float *dest, const float *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_signal(
    rocshmem_ctx_t ctx, double *dest, const double *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_signal(
    double *dest, const double *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_signal(
    rocshmem_ctx_t ctx, char *dest, const char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_signal(
    char *dest, const char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_signal(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_signal(
    signed char *dest, const signed char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_signal(
    rocshmem_ctx_t ctx, short *dest, const short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_signal(
    short *dest, const short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_signal(
    rocshmem_ctx_t ctx, int *dest, const int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_signal(
    int *dest, const int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_signal(
    rocshmem_ctx_t ctx, long *dest, const long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_signal(
    long *dest, const long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_signal(
    rocshmem_ctx_t ctx, long long *dest, const long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_signal(
    long long *dest, const long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_signal(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_signal(
    unsigned char *dest, const unsigned char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_signal(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_signal(
    unsigned short *dest, const unsigned short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_signal(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_signal(
    unsigned int *dest, const unsigned int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_signal(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_signal(
    unsigned long *dest, const unsigned long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_signal(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_signal(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_half_put_signal(
    rocshmem_ctx_t ctx, __half *dest, const __half *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_half_put_signal(
    __half *dest, const __half *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_bfloat16_put_signal(
    rocshmem_ctx_t ctx, __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_bfloat16_put_signal(
    __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int8_put_signal(
    rocshmem_ctx_t ctx, int8_t *dest, const int8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int8_put_signal(
    int8_t *dest, const int8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int16_put_signal(
    rocshmem_ctx_t ctx, int16_t *dest, const int16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int16_put_signal(
    int16_t *dest, const int16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_put_signal(
    rocshmem_ctx_t ctx, int32_t *dest, const int32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_put_signal(
    int32_t *dest, const int32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_put_signal(
    rocshmem_ctx_t ctx, int64_t *dest, const int64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_put_signal(
    int64_t *dest, const int64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint8_put_signal(
    rocshmem_ctx_t ctx, uint8_t *dest, const uint8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint8_put_signal(
    uint8_t *dest, const uint8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint16_put_signal(
    rocshmem_ctx_t ctx, uint16_t *dest, const uint16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint16_put_signal(
    uint16_t *dest, const uint16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_put_signal(
    rocshmem_ctx_t ctx, uint32_t *dest, const uint32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_put_signal(
    uint32_t *dest, const uint32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_put_signal(
    rocshmem_ctx_t ctx, uint64_t *dest, const uint64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_put_signal(
    uint64_t *dest, const uint64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_put_signal(
    rocshmem_ctx_t ctx, size_t *dest, const size_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_put_signal(
    size_t *dest, const size_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_put_signal(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_put_signal(
    ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_signal_wg(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_signal_wg(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_signal_wg(
    rocshmem_ctx_t ctx, float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_signal_wg(
    float *dest, const float *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_signal_wg(
    rocshmem_ctx_t ctx, double *dest, const double *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_signal_wg(
    double *dest, const double *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_signal_wg(
    rocshmem_ctx_t ctx, char *dest, const char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_signal_wg(
    char *dest, const char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_signal_wg(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_signal_wg(
    signed char *dest, const signed char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_signal_wg(
    rocshmem_ctx_t ctx, short *dest, const short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_signal_wg(
    short *dest, const short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_signal_wg(
    rocshmem_ctx_t ctx, int *dest, const int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_signal_wg(
    int *dest, const int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_signal_wg(
    rocshmem_ctx_t ctx, long *dest, const long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_signal_wg(
    long *dest, const long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_signal_wg(
    rocshmem_ctx_t ctx, long long *dest, const long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_signal_wg(
    long long *dest, const long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_signal_wg(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_signal_wg(
    unsigned char *dest, const unsigned char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_signal_wg(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_signal_wg(
    unsigned short *dest, const unsigned short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_signal_wg(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_signal_wg(
    unsigned int *dest, const unsigned int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_signal_wg(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_signal_wg(
    unsigned long *dest, const unsigned long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_signal_wg(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_signal_wg(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_half_put_signal_wg(
    rocshmem_ctx_t ctx, __half *dest, const __half *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_half_put_signal_wg(
    __half *dest, const __half *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_bfloat16_put_signal_wg(
    rocshmem_ctx_t ctx, __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_bfloat16_put_signal_wg(
    __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int8_put_signal_wg(
    rocshmem_ctx_t ctx, int8_t *dest, const int8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int8_put_signal_wg(
    int8_t *dest, const int8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int16_put_signal_wg(
    rocshmem_ctx_t ctx, int16_t *dest, const int16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int16_put_signal_wg(
    int16_t *dest, const int16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_put_signal_wg(
    rocshmem_ctx_t ctx, int32_t *dest, const int32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_put_signal_wg(
    int32_t *dest, const int32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_put_signal_wg(
    rocshmem_ctx_t ctx, int64_t *dest, const int64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_put_signal_wg(
    int64_t *dest, const int64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint8_put_signal_wg(
    rocshmem_ctx_t ctx, uint8_t *dest, const uint8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint8_put_signal_wg(
    uint8_t *dest, const uint8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint16_put_signal_wg(
    rocshmem_ctx_t ctx, uint16_t *dest, const uint16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint16_put_signal_wg(
    uint16_t *dest, const uint16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_put_signal_wg(
    rocshmem_ctx_t ctx, uint32_t *dest, const uint32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_put_signal_wg(
    uint32_t *dest, const uint32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_put_signal_wg(
    rocshmem_ctx_t ctx, uint64_t *dest, const uint64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_put_signal_wg(
    uint64_t *dest, const uint64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_put_signal_wg(
    rocshmem_ctx_t ctx, size_t *dest, const size_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_put_signal_wg(
    size_t *dest, const size_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_put_signal_wg(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_put_signal_wg(
    ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_signal_wave(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_signal_wave(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_signal_wave(
    rocshmem_ctx_t ctx, float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_signal_wave(
    float *dest, const float *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_signal_wave(
    rocshmem_ctx_t ctx, double *dest, const double *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_signal_wave(
    double *dest, const double *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_signal_wave(
    rocshmem_ctx_t ctx, char *dest, const char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_signal_wave(
    char *dest, const char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_signal_wave(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_signal_wave(
    signed char *dest, const signed char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_signal_wave(
    rocshmem_ctx_t ctx, short *dest, const short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_signal_wave(
    short *dest, const short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_signal_wave(
    rocshmem_ctx_t ctx, int *dest, const int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_signal_wave(
    int *dest, const int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_signal_wave(
    rocshmem_ctx_t ctx, long *dest, const long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_signal_wave(
    long *dest, const long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_signal_wave(
    rocshmem_ctx_t ctx, long long *dest, const long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_signal_wave(
    long long *dest, const long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_signal_wave(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_signal_wave(
    unsigned char *dest, const unsigned char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_signal_wave(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_signal_wave(
    unsigned short *dest, const unsigned short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_signal_wave(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_signal_wave(
    unsigned int *dest, const unsigned int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_signal_wave(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_signal_wave(
    unsigned long *dest, const unsigned long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_signal_wave(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_signal_wave(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_half_put_signal_wave(
    rocshmem_ctx_t ctx, __half *dest, const __half *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_half_put_signal_wave(
    __half *dest, const __half *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_bfloat16_put_signal_wave(
    rocshmem_ctx_t ctx, __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_bfloat16_put_signal_wave(
    __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int8_put_signal_wave(
    rocshmem_ctx_t ctx, int8_t *dest, const int8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int8_put_signal_wave(
    int8_t *dest, const int8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int16_put_signal_wave(
    rocshmem_ctx_t ctx, int16_t *dest, const int16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int16_put_signal_wave(
    int16_t *dest, const int16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_put_signal_wave(
    rocshmem_ctx_t ctx, int32_t *dest, const int32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_put_signal_wave(
    int32_t *dest, const int32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_put_signal_wave(
    rocshmem_ctx_t ctx, int64_t *dest, const int64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_put_signal_wave(
    int64_t *dest, const int64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint8_put_signal_wave(
    rocshmem_ctx_t ctx, uint8_t *dest, const uint8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint8_put_signal_wave(
    uint8_t *dest, const uint8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint16_put_signal_wave(
    rocshmem_ctx_t ctx, uint16_t *dest, const uint16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint16_put_signal_wave(
    uint16_t *dest, const uint16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_put_signal_wave(
    rocshmem_ctx_t ctx, uint32_t *dest, const uint32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_put_signal_wave(
    uint32_t *dest, const uint32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_put_signal_wave(
    rocshmem_ctx_t ctx, uint64_t *dest, const uint64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_put_signal_wave(
    uint64_t *dest, const uint64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_put_signal_wave(
    rocshmem_ctx_t ctx, size_t *dest, const size_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_put_signal_wave(
    size_t *dest, const size_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_put_signal_wave(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_put_signal_wave(
    ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_signal_nbi(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_signal_nbi(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_signal_nbi(
    rocshmem_ctx_t ctx, float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_signal_nbi(
    float *dest, const float *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_signal_nbi(
    rocshmem_ctx_t ctx, double *dest, const double *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_signal_nbi(
    double *dest, const double *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_signal_nbi(
    rocshmem_ctx_t ctx, char *dest, const char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_signal_nbi(
    char *dest, const char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_signal_nbi(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_signal_nbi(
    signed char *dest, const signed char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_signal_nbi(
    rocshmem_ctx_t ctx, short *dest, const short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_signal_nbi(
    short *dest, const short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_signal_nbi(
    rocshmem_ctx_t ctx, int *dest, const int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_signal_nbi(
    int *dest, const int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_signal_nbi(
    rocshmem_ctx_t ctx, long *dest, const long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_signal_nbi(
    long *dest, const long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_signal_nbi(
    rocshmem_ctx_t ctx, long long *dest, const long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_signal_nbi(
    long long *dest, const long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_signal_nbi(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_signal_nbi(
    unsigned char *dest, const unsigned char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_signal_nbi(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_signal_nbi(
    unsigned short *dest, const unsigned short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_signal_nbi(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_signal_nbi(
    unsigned int *dest, const unsigned int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_signal_nbi(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_signal_nbi(
    unsigned long *dest, const unsigned long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_signal_nbi(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_signal_nbi(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_half_put_signal_nbi(
    rocshmem_ctx_t ctx, __half *dest, const __half *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_half_put_signal_nbi(
    __half *dest, const __half *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_bfloat16_put_signal_nbi(
    rocshmem_ctx_t ctx, __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_bfloat16_put_signal_nbi(
    __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int8_put_signal_nbi(
    rocshmem_ctx_t ctx, int8_t *dest, const int8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int8_put_signal_nbi(
    int8_t *dest, const int8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int16_put_signal_nbi(
    rocshmem_ctx_t ctx, int16_t *dest, const int16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int16_put_signal_nbi(
    int16_t *dest, const int16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_put_signal_nbi(
    rocshmem_ctx_t ctx, int32_t *dest, const int32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_put_signal_nbi(
    int32_t *dest, const int32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_put_signal_nbi(
    rocshmem_ctx_t ctx, int64_t *dest, const int64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_put_signal_nbi(
    int64_t *dest, const int64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint8_put_signal_nbi(
    rocshmem_ctx_t ctx, uint8_t *dest, const uint8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint8_put_signal_nbi(
    uint8_t *dest, const uint8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint16_put_signal_nbi(
    rocshmem_ctx_t ctx, uint16_t *dest, const uint16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint16_put_signal_nbi(
    uint16_t *dest, const uint16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_put_signal_nbi(
    rocshmem_ctx_t ctx, uint32_t *dest, const uint32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_put_signal_nbi(
    uint32_t *dest, const uint32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_put_signal_nbi(
    rocshmem_ctx_t ctx, uint64_t *dest, const uint64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_put_signal_nbi(
    uint64_t *dest, const uint64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_put_signal_nbi(
    rocshmem_ctx_t ctx, size_t *dest, const size_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_put_signal_nbi(
    size_t *dest, const size_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_put_signal_nbi(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_put_signal_nbi(
    ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_signal_nbi_wg(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_signal_nbi_wg(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_signal_nbi_wg(
    float *dest, const float *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, double *dest, const double *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_signal_nbi_wg(
    double *dest, const double *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, char *dest, const char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_signal_nbi_wg(
    char *dest, const char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_signal_nbi_wg(
    signed char *dest, const signed char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, short *dest, const short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_signal_nbi_wg(
    short *dest, const short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, int *dest, const int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_signal_nbi_wg(
    int *dest, const int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, long *dest, const long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_signal_nbi_wg(
    long *dest, const long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, long long *dest, const long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_signal_nbi_wg(
    long long *dest, const long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_signal_nbi_wg(
    unsigned char *dest, const unsigned char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_signal_nbi_wg(
    unsigned short *dest, const unsigned short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_signal_nbi_wg(
    unsigned int *dest, const unsigned int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_signal_nbi_wg(
    unsigned long *dest, const unsigned long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_signal_nbi_wg(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_half_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, __half *dest, const __half *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_half_put_signal_nbi_wg(
    __half *dest, const __half *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_bfloat16_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_bfloat16_put_signal_nbi_wg(
    __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int8_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, int8_t *dest, const int8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int8_put_signal_nbi_wg(
    int8_t *dest, const int8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int16_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, int16_t *dest, const int16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int16_put_signal_nbi_wg(
    int16_t *dest, const int16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, int32_t *dest, const int32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_put_signal_nbi_wg(
    int32_t *dest, const int32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, int64_t *dest, const int64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_put_signal_nbi_wg(
    int64_t *dest, const int64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint8_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, uint8_t *dest, const uint8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint8_put_signal_nbi_wg(
    uint8_t *dest, const uint8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint16_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, uint16_t *dest, const uint16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint16_put_signal_nbi_wg(
    uint16_t *dest, const uint16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, uint32_t *dest, const uint32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_put_signal_nbi_wg(
    uint32_t *dest, const uint32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, uint64_t *dest, const uint64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_put_signal_nbi_wg(
    uint64_t *dest, const uint64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, size_t *dest, const size_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_put_signal_nbi_wg(
    size_t *dest, const size_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_put_signal_nbi_wg(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_put_signal_nbi_wg(
    ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_putmem_signal_nbi_wave(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ctx_putmem_signal_nbi_wave(
    rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_float_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, float *dest, const float *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_float_put_signal_nbi_wave(
    float *dest, const float *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, double *dest, const double *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_double_put_signal_nbi_wave(
    double *dest, const double *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, char *dest, const char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_char_put_signal_nbi_wave(
    char *dest, const char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_schar_put_signal_nbi_wave(
    signed char *dest, const signed char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, short *dest, const short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_short_put_signal_nbi_wave(
    short *dest, const short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, int *dest, const int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int_put_signal_nbi_wave(
    int *dest, const int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, long *dest, const long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_long_put_signal_nbi_wave(
    long *dest, const long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, long long *dest, const long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_longlong_put_signal_nbi_wave(
    long long *dest, const long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uchar_put_signal_nbi_wave(
    unsigned char *dest, const unsigned char *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ushort_put_signal_nbi_wave(
    unsigned short *dest, const unsigned short *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint_put_signal_nbi_wave(
    unsigned int *dest, const unsigned int *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulong_put_signal_nbi_wave(
    unsigned long *dest, const unsigned long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ulonglong_put_signal_nbi_wave(
    unsigned long long *dest, const unsigned long long *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_half_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, __half *dest, const __half *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_half_put_signal_nbi_wave(
    __half *dest, const __half *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_bfloat16_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_bfloat16_put_signal_nbi_wave(
    __hip_bfloat16 *dest, const __hip_bfloat16 *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int8_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, int8_t *dest, const int8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int8_put_signal_nbi_wave(
    int8_t *dest, const int8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int16_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, int16_t *dest, const int16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int16_put_signal_nbi_wave(
    int16_t *dest, const int16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int32_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, int32_t *dest, const int32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int32_put_signal_nbi_wave(
    int32_t *dest, const int32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int64_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, int64_t *dest, const int64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_int64_put_signal_nbi_wave(
    int64_t *dest, const int64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint8_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, uint8_t *dest, const uint8_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint8_put_signal_nbi_wave(
    uint8_t *dest, const uint8_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint16_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, uint16_t *dest, const uint16_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint16_put_signal_nbi_wave(
    uint16_t *dest, const uint16_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint32_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, uint32_t *dest, const uint32_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint32_put_signal_nbi_wave(
    uint32_t *dest, const uint32_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint64_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, uint64_t *dest, const uint64_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_uint64_put_signal_nbi_wave(
    uint64_t *dest, const uint64_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_size_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, size_t *dest, const size_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_size_put_signal_nbi_wave(
    size_t *dest, const size_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ptrdiff_put_signal_nbi_wave(
    rocshmem_ctx_t ctx, ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems,
    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ ATTR_NO_INLINE void rocshmem_ptrdiff_put_signal_nbi_wave(
    ptrdiff_t *dest, const ptrdiff_t *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);


__device__ ATTR_NO_INLINE uint64_t rocshmem_signal_fetch(const uint64_t *sig_addr);
__device__ ATTR_NO_INLINE uint64_t rocshmem_signal_fetch_wg(const uint64_t *sig_addr);
__device__ ATTR_NO_INLINE uint64_t rocshmem_signal_fetch_wave(const uint64_t *sig_addr);

/**
 * @brief Kernel wrapper for putmem_signal operation on stream
 *
 * @param[in] dest      Destination address on remote PE
 * @param[in] source    Source address on local PE
 * @param[in] nelems    Size of the transfer in bytes
 * @param[in] sig_addr  Address of signal variable on remote PE
 * @param[in] signal    Signal value to write
 * @param[in] sig_op    Signal operation (ROCSHMEM_SIGNAL_SET or
 * ROCSHMEM_SIGNAL_ADD)
 * @param[in] pe        PE of the remote process
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_putmem_signal_kernel(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe);

/**
 * @brief Kernel wrapper for signal_wait_until operation on stream
 *
 * @param[in] sig_addr  Address of signal variable on the symmetric heap
 * @param[in] cmp       Comparison operator
 * @param[in] cmp_value Value to compare against
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_signal_wait_until_kernel(
    uint64_t *sig_addr, int cmp, uint64_t cmp_value);

}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_SIG_OP_HPP
