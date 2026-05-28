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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_P2P_SYNC_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_P2P_SYNC_HPP

namespace rocshmem {

/**
 * @name SHMEM_WAIT_UNTIL
 * @brief Block the caller until the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ivars Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return void
 */
__device__ void rocshmem_float_wait_until(
    float *ivars, int cmp, float val);
__device__ size_t rocshmem_float_wait_until_any(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__device__ void rocshmem_float_wait_until_all(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__device__ size_t rocshmem_float_wait_until_some(
    float *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, float val);
__device__ size_t rocshmem_float_wait_until_any_vector(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__device__ void rocshmem_float_wait_until_all_vector(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__device__ size_t rocshmem_float_wait_until_some_vector(
    float *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, float val);
__host__ void rocshmem_float_wait_until(
    float *ivars, int cmp, float val);
__host__ size_t rocshmem_float_wait_until_any(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__host__ void rocshmem_float_wait_until_all(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__host__ size_t rocshmem_float_wait_until_some(
    float *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, float val);
__host__ size_t rocshmem_float_wait_until_any_vector(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__host__ void rocshmem_float_wait_until_all_vector(
    float *ivars, size_t nelems, const int* status,
    int cmp, float val);
__host__ size_t rocshmem_float_wait_until_some_vector(
    float *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, float val);

__device__ void rocshmem_double_wait_until(
    double *ivars, int cmp, double val);
__device__ size_t rocshmem_double_wait_until_any(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__device__ void rocshmem_double_wait_until_all(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__device__ size_t rocshmem_double_wait_until_some(
    double *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, double val);
__device__ size_t rocshmem_double_wait_until_any_vector(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__device__ void rocshmem_double_wait_until_all_vector(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__device__ size_t rocshmem_double_wait_until_some_vector(
    double *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, double val);
__host__ void rocshmem_double_wait_until(
    double *ivars, int cmp, double val);
__host__ size_t rocshmem_double_wait_until_any(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__host__ void rocshmem_double_wait_until_all(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__host__ size_t rocshmem_double_wait_until_some(
    double *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, double val);
__host__ size_t rocshmem_double_wait_until_any_vector(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__host__ void rocshmem_double_wait_until_all_vector(
    double *ivars, size_t nelems, const int* status,
    int cmp, double val);
__host__ size_t rocshmem_double_wait_until_some_vector(
    double *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, double val);

__device__ void rocshmem_char_wait_until(
    char *ivars, int cmp, char val);
__device__ size_t rocshmem_char_wait_until_any(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__device__ void rocshmem_char_wait_until_all(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__device__ size_t rocshmem_char_wait_until_some(
    char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, char val);
__device__ size_t rocshmem_char_wait_until_any_vector(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__device__ void rocshmem_char_wait_until_all_vector(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__device__ size_t rocshmem_char_wait_until_some_vector(
    char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, char val);
__host__ void rocshmem_char_wait_until(
    char *ivars, int cmp, char val);
__host__ size_t rocshmem_char_wait_until_any(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__host__ void rocshmem_char_wait_until_all(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__host__ size_t rocshmem_char_wait_until_some(
    char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, char val);
__host__ size_t rocshmem_char_wait_until_any_vector(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__host__ void rocshmem_char_wait_until_all_vector(
    char *ivars, size_t nelems, const int* status,
    int cmp, char val);
__host__ size_t rocshmem_char_wait_until_some_vector(
    char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, char val);

__device__ void rocshmem_schar_wait_until(
    signed char *ivars, int cmp, signed char val);
__device__ size_t rocshmem_schar_wait_until_any(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__device__ void rocshmem_schar_wait_until_all(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__device__ size_t rocshmem_schar_wait_until_some(
    signed char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, signed char val);
__device__ size_t rocshmem_schar_wait_until_any_vector(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__device__ void rocshmem_schar_wait_until_all_vector(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__device__ size_t rocshmem_schar_wait_until_some_vector(
    signed char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, signed char val);
__host__ void rocshmem_schar_wait_until(
    signed char *ivars, int cmp, signed char val);
__host__ size_t rocshmem_schar_wait_until_any(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__host__ void rocshmem_schar_wait_until_all(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__host__ size_t rocshmem_schar_wait_until_some(
    signed char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, signed char val);
__host__ size_t rocshmem_schar_wait_until_any_vector(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__host__ void rocshmem_schar_wait_until_all_vector(
    signed char *ivars, size_t nelems, const int* status,
    int cmp, signed char val);
__host__ size_t rocshmem_schar_wait_until_some_vector(
    signed char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, signed char val);

__device__ void rocshmem_short_wait_until(
    short *ivars, int cmp, short val);
__device__ size_t rocshmem_short_wait_until_any(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__device__ void rocshmem_short_wait_until_all(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__device__ size_t rocshmem_short_wait_until_some(
    short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, short val);
__device__ size_t rocshmem_short_wait_until_any_vector(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__device__ void rocshmem_short_wait_until_all_vector(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__device__ size_t rocshmem_short_wait_until_some_vector(
    short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, short val);
__host__ void rocshmem_short_wait_until(
    short *ivars, int cmp, short val);
__host__ size_t rocshmem_short_wait_until_any(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__host__ void rocshmem_short_wait_until_all(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__host__ size_t rocshmem_short_wait_until_some(
    short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, short val);
__host__ size_t rocshmem_short_wait_until_any_vector(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__host__ void rocshmem_short_wait_until_all_vector(
    short *ivars, size_t nelems, const int* status,
    int cmp, short val);
__host__ size_t rocshmem_short_wait_until_some_vector(
    short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, short val);

__device__ void rocshmem_int_wait_until(
    int *ivars, int cmp, int val);
__device__ size_t rocshmem_int_wait_until_any(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__device__ void rocshmem_int_wait_until_all(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__device__ size_t rocshmem_int_wait_until_some(
    int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, int val);
__device__ size_t rocshmem_int_wait_until_any_vector(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__device__ void rocshmem_int_wait_until_all_vector(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__device__ size_t rocshmem_int_wait_until_some_vector(
    int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, int val);
__host__ void rocshmem_int_wait_until(
    int *ivars, int cmp, int val);
__host__ size_t rocshmem_int_wait_until_any(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__host__ void rocshmem_int_wait_until_all(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__host__ size_t rocshmem_int_wait_until_some(
    int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, int val);
__host__ size_t rocshmem_int_wait_until_any_vector(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__host__ void rocshmem_int_wait_until_all_vector(
    int *ivars, size_t nelems, const int* status,
    int cmp, int val);
__host__ size_t rocshmem_int_wait_until_some_vector(
    int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, int val);

__device__ void rocshmem_long_wait_until(
    long *ivars, int cmp, long val);
__device__ size_t rocshmem_long_wait_until_any(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__device__ void rocshmem_long_wait_until_all(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__device__ size_t rocshmem_long_wait_until_some(
    long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long val);
__device__ size_t rocshmem_long_wait_until_any_vector(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__device__ void rocshmem_long_wait_until_all_vector(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__device__ size_t rocshmem_long_wait_until_some_vector(
    long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long val);
__host__ void rocshmem_long_wait_until(
    long *ivars, int cmp, long val);
__host__ size_t rocshmem_long_wait_until_any(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__host__ void rocshmem_long_wait_until_all(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__host__ size_t rocshmem_long_wait_until_some(
    long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long val);
__host__ size_t rocshmem_long_wait_until_any_vector(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__host__ void rocshmem_long_wait_until_all_vector(
    long *ivars, size_t nelems, const int* status,
    int cmp, long val);
__host__ size_t rocshmem_long_wait_until_some_vector(
    long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long val);

__device__ void rocshmem_longlong_wait_until(
    long long *ivars, int cmp, long long val);
__device__ size_t rocshmem_longlong_wait_until_any(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__device__ void rocshmem_longlong_wait_until_all(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__device__ size_t rocshmem_longlong_wait_until_some(
    long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long long val);
__device__ size_t rocshmem_longlong_wait_until_any_vector(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__device__ void rocshmem_longlong_wait_until_all_vector(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__device__ size_t rocshmem_longlong_wait_until_some_vector(
    long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long long val);
__host__ void rocshmem_longlong_wait_until(
    long long *ivars, int cmp, long long val);
__host__ size_t rocshmem_longlong_wait_until_any(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__host__ void rocshmem_longlong_wait_until_all(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__host__ size_t rocshmem_longlong_wait_until_some(
    long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long long val);
__host__ size_t rocshmem_longlong_wait_until_any_vector(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__host__ void rocshmem_longlong_wait_until_all_vector(
    long long *ivars, size_t nelems, const int* status,
    int cmp, long long val);
__host__ size_t rocshmem_longlong_wait_until_some_vector(
    long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, long long val);

__device__ void rocshmem_uchar_wait_until(
    unsigned char *ivars, int cmp, unsigned char val);
__device__ size_t rocshmem_uchar_wait_until_any(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__device__ void rocshmem_uchar_wait_until_all(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__device__ size_t rocshmem_uchar_wait_until_some(
    unsigned char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned char val);
__device__ size_t rocshmem_uchar_wait_until_any_vector(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__device__ void rocshmem_uchar_wait_until_all_vector(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__device__ size_t rocshmem_uchar_wait_until_some_vector(
    unsigned char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned char val);
__host__ void rocshmem_uchar_wait_until(
    unsigned char *ivars, int cmp, unsigned char val);
__host__ size_t rocshmem_uchar_wait_until_any(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__host__ void rocshmem_uchar_wait_until_all(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__host__ size_t rocshmem_uchar_wait_until_some(
    unsigned char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned char val);
__host__ size_t rocshmem_uchar_wait_until_any_vector(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__host__ void rocshmem_uchar_wait_until_all_vector(
    unsigned char *ivars, size_t nelems, const int* status,
    int cmp, unsigned char val);
__host__ size_t rocshmem_uchar_wait_until_some_vector(
    unsigned char *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned char val);

__device__ void rocshmem_ushort_wait_until(
    unsigned short *ivars, int cmp, unsigned short val);
__device__ size_t rocshmem_ushort_wait_until_any(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__device__ void rocshmem_ushort_wait_until_all(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__device__ size_t rocshmem_ushort_wait_until_some(
    unsigned short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned short val);
__device__ size_t rocshmem_ushort_wait_until_any_vector(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__device__ void rocshmem_ushort_wait_until_all_vector(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__device__ size_t rocshmem_ushort_wait_until_some_vector(
    unsigned short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned short val);
__host__ void rocshmem_ushort_wait_until(
    unsigned short *ivars, int cmp, unsigned short val);
__host__ size_t rocshmem_ushort_wait_until_any(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__host__ void rocshmem_ushort_wait_until_all(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__host__ size_t rocshmem_ushort_wait_until_some(
    unsigned short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned short val);
__host__ size_t rocshmem_ushort_wait_until_any_vector(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__host__ void rocshmem_ushort_wait_until_all_vector(
    unsigned short *ivars, size_t nelems, const int* status,
    int cmp, unsigned short val);
__host__ size_t rocshmem_ushort_wait_until_some_vector(
    unsigned short *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned short val);

__device__ void rocshmem_uint_wait_until(
    unsigned int *ivars, int cmp, unsigned int val);
__device__ size_t rocshmem_uint_wait_until_any(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__device__ void rocshmem_uint_wait_until_all(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__device__ size_t rocshmem_uint_wait_until_some(
    unsigned int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned int val);
__device__ size_t rocshmem_uint_wait_until_any_vector(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__device__ void rocshmem_uint_wait_until_all_vector(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__device__ size_t rocshmem_uint_wait_until_some_vector(
    unsigned int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned int val);
__host__ void rocshmem_uint_wait_until(
    unsigned int *ivars, int cmp, unsigned int val);
__host__ size_t rocshmem_uint_wait_until_any(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__host__ void rocshmem_uint_wait_until_all(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__host__ size_t rocshmem_uint_wait_until_some(
    unsigned int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned int val);
__host__ size_t rocshmem_uint_wait_until_any_vector(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__host__ void rocshmem_uint_wait_until_all_vector(
    unsigned int *ivars, size_t nelems, const int* status,
    int cmp, unsigned int val);
__host__ size_t rocshmem_uint_wait_until_some_vector(
    unsigned int *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned int val);

__device__ void rocshmem_ulong_wait_until(
    unsigned long *ivars, int cmp, unsigned long val);
__device__ size_t rocshmem_ulong_wait_until_any(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__device__ void rocshmem_ulong_wait_until_all(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__device__ size_t rocshmem_ulong_wait_until_some(
    unsigned long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long val);
__device__ size_t rocshmem_ulong_wait_until_any_vector(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__device__ void rocshmem_ulong_wait_until_all_vector(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__device__ size_t rocshmem_ulong_wait_until_some_vector(
    unsigned long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long val);
__host__ void rocshmem_ulong_wait_until(
    unsigned long *ivars, int cmp, unsigned long val);
__host__ size_t rocshmem_ulong_wait_until_any(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__host__ void rocshmem_ulong_wait_until_all(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__host__ size_t rocshmem_ulong_wait_until_some(
    unsigned long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long val);
__host__ size_t rocshmem_ulong_wait_until_any_vector(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__host__ void rocshmem_ulong_wait_until_all_vector(
    unsigned long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long val);
__host__ size_t rocshmem_ulong_wait_until_some_vector(
    unsigned long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long val);

__device__ void rocshmem_ulonglong_wait_until(
    unsigned long long *ivars, int cmp, unsigned long long val);
__device__ size_t rocshmem_ulonglong_wait_until_any(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__device__ void rocshmem_ulonglong_wait_until_all(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__device__ size_t rocshmem_ulonglong_wait_until_some(
    unsigned long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long long val);
__device__ size_t rocshmem_ulonglong_wait_until_any_vector(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__device__ void rocshmem_ulonglong_wait_until_all_vector(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__device__ size_t rocshmem_ulonglong_wait_until_some_vector(
    unsigned long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long long val);
__host__ void rocshmem_ulonglong_wait_until(
    unsigned long long *ivars, int cmp, unsigned long long val);
__host__ size_t rocshmem_ulonglong_wait_until_any(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__host__ void rocshmem_ulonglong_wait_until_all(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__host__ size_t rocshmem_ulonglong_wait_until_some(
    unsigned long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long long val);
__host__ size_t rocshmem_ulonglong_wait_until_any_vector(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__host__ void rocshmem_ulonglong_wait_until_all_vector(
    unsigned long long *ivars, size_t nelems, const int* status,
    int cmp, unsigned long long val);
__host__ size_t rocshmem_ulonglong_wait_until_some_vector(
    unsigned long long *ivars, size_t nelems, size_t* indices, const int* status,
    int cmp, unsigned long long val);


/**
 * @name SHMEM_TEST
 * @brief test if the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROCSHMEM function.
 *
 * @param[in] ivars Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return 1 if the evaluation is true else 0
 */
__device__ int rocshmem_float_test(
    float *ivars, int cmp, float val);
__host__ int rocshmem_float_test(
    float *ivars, int cmp, float val);

__device__ int rocshmem_double_test(
    double *ivars, int cmp, double val);
__host__ int rocshmem_double_test(
    double *ivars, int cmp, double val);

__device__ int rocshmem_char_test(
    char *ivars, int cmp, char val);
__host__ int rocshmem_char_test(
    char *ivars, int cmp, char val);

__device__ int rocshmem_schar_test(
    signed char *ivars, int cmp, signed char val);
__host__ int rocshmem_schar_test(
    signed char *ivars, int cmp, signed char val);

__device__ int rocshmem_short_test(
    short *ivars, int cmp, short val);
__host__ int rocshmem_short_test(
    short *ivars, int cmp, short val);

__device__ int rocshmem_int_test(
    int *ivars, int cmp, int val);
__host__ int rocshmem_int_test(
    int *ivars, int cmp, int val);

__device__ int rocshmem_long_test(
    long *ivars, int cmp, long val);
__host__ int rocshmem_long_test(
    long *ivars, int cmp, long val);

__device__ int rocshmem_longlong_test(
    long long *ivars, int cmp, long long val);
__host__ int rocshmem_longlong_test(
    long long *ivars, int cmp, long long val);

__device__ int rocshmem_uchar_test(
    unsigned char *ivars, int cmp, unsigned char val);
__host__ int rocshmem_uchar_test(
    unsigned char *ivars, int cmp, unsigned char val);

__device__ int rocshmem_ushort_test(
    unsigned short *ivars, int cmp, unsigned short val);
__host__ int rocshmem_ushort_test(
    unsigned short *ivars, int cmp, unsigned short val);

__device__ int rocshmem_uint_test(
    unsigned int *ivars, int cmp, unsigned int val);
__host__ int rocshmem_uint_test(
    unsigned int *ivars, int cmp, unsigned int val);

__device__ int rocshmem_ulong_test(
    unsigned long *ivars, int cmp, unsigned long val);
__host__ int rocshmem_ulong_test(
    unsigned long *ivars, int cmp, unsigned long val);

__device__ int rocshmem_ulonglong_test(
    unsigned long long *ivars, int cmp, unsigned long long val);
__host__ int rocshmem_ulonglong_test(
    unsigned long long *ivars, int cmp, unsigned long long val);


}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_P2P_SYNC_HPP
