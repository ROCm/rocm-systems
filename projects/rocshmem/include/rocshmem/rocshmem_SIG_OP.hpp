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


__device__ ATTR_NO_INLINE uint64_t rocshmem_signal_fetch(const uint64_t *sig_addr);
__device__ ATTR_NO_INLINE uint64_t rocshmem_signal_fetch_wg(const uint64_t *sig_addr);
__device__ ATTR_NO_INLINE uint64_t rocshmem_signal_fetch_wave(const uint64_t *sig_addr);


}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_SIG_OP_HPP
