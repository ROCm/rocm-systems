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

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>

#define DECL_RMA(T, TNAME)                                              \
  __device__ void rocshmem_##TNAME##_put(T*, const T*, size_t, int);    \
  __device__ void rocshmem_##TNAME##_get(T*, const T*, size_t, int);

#define DECL_RMA_WAVE(T, TNAME)                                         \
  __device__ void rocshmem_##TNAME##_put_wave(T*, const T*, size_t, int);\
  __device__ void rocshmem_##TNAME##_get_wave(T*, const T*, size_t, int);

extern "C" {
  __device__ int rocshmem_my_pe();
  __device__ int rocshmem_n_pes();
  __device__ void rocshmem_barrier_all();
  __device__ void rocshmem_barrier_all_wave();
  __device__ void rocshmem_quiet();

  __device__ void rocshmem_putmem(void*, const void*, size_t, int);
  __device__ void rocshmem_getmem(void*, const void*, size_t, int);

  __device__ void rocshmem_int_p(int*, int, int);
  __device__ void rocshmem_int64_p(int64_t*, int64_t, int);

  DECL_RMA(int, int)
  DECL_RMA(float, float)
  DECL_RMA_WAVE(int, int)

  __device__ int64_t rocshmem_int64_atomic_fetch_add(int64_t*, int64_t, int);
  __device__ int64_t rocshmem_int64_atomic_compare_swap(int64_t*, int64_t, int64_t, int);
  __device__ int64_t rocshmem_int64_atomic_fetch(int64_t*, int);
  __device__ void rocshmem_int64_atomic_set(int64_t*, int64_t, int);
  __device__ int64_t rocshmem_int64_atomic_swap(int64_t*, int64_t, int);
  __device__ uint64_t rocshmem_uint64_atomic_fetch_and(uint64_t*, uint64_t, int);
  __device__ void rocshmem_uint64_atomic_and(uint64_t*, uint64_t, int);

  __device__ void rocshmem_int_put_signal(int*, const int*, size_t,
                                          uint64_t*, uint64_t, int, int);
  __device__ void rocshmem_int_wait_until(int*, int, int);
  __device__ size_t rocshmem_int_wait_until_any_vector(int*, size_t, const int*, int, int*);
  __device__ void rocshmem_int_wait_until_all_vector(int*, size_t, const int*, int, int*);
}

extern "C" __global__ void test_pe_info(int *out_my_pe, int *out_n_pes) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out_my_pe = rocshmem_my_pe();
    *out_n_pes = rocshmem_n_pes();
  }
}

extern "C" __global__ void test_put(int *symmetric_buf, int my_pe, int n_pes) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    rocshmem_int_p(symmetric_buf, my_pe * 100 + 42, next_pe);
    rocshmem_barrier_all();
  }
}

extern "C" __global__ void test_putmem_getmem(
    int *sym_src, int *sym_dst, int *local_result,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    rocshmem_putmem(sym_dst, sym_src,
                    count * sizeof(int), next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();

    rocshmem_getmem(local_result, sym_src,
                    count * sizeof(int), prev_pe);
    rocshmem_quiet();
  }
}

extern "C" __global__ void test_typed_int_put_get(
    int *sym_src, int *sym_dst, int *local_result,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    rocshmem_int_put(sym_dst, sym_src, count, next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();

    rocshmem_int_get(local_result, sym_src, count, prev_pe);
    rocshmem_quiet();
  }
}

extern "C" __global__ void test_typed_float_put_get(
    float *sym_src, float *sym_dst, float *local_result,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    rocshmem_float_put(sym_dst, sym_src, count, next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();

    rocshmem_float_get(local_result, sym_src, count, prev_pe);
    rocshmem_quiet();
  }
}

extern "C" __global__ void test_typed_atomic(
    int64_t *sym_counter, int64_t *out_old, int my_pe, int n_pes)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int64_t old = rocshmem_int64_atomic_fetch_add(sym_counter, 10, my_pe);
    out_old[0] = old;

    rocshmem_barrier_all();

    int64_t swapped = rocshmem_int64_atomic_compare_swap(
        sym_counter, 10, 999, my_pe);
    out_old[1] = swapped;
  }
}

extern "C" __global__ void test_typed_put_signal(
    int *sym_src, int *sym_dst, uint64_t *sym_sig,
    int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;

    rocshmem_int_put_signal(
        sym_dst, sym_src, count,
        sym_sig, 1, 0 /* ROCSHMEM_SIGNAL_SET */, next_pe);
    rocshmem_quiet();
    rocshmem_barrier_all();
  }
}

extern "C" __global__ void test_typed_wait_until(
    int *sym_flag, int my_pe, int n_pes)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;

    rocshmem_int_p(sym_flag, my_pe + 1, next_pe);
    rocshmem_int_wait_until(sym_flag, 1 /* ROCSHMEM_CMP_NE */, 0);
  }
}

extern "C" __global__ void test_typed_int_put_get_wave(
    int *sym_src, int *sym_dst, int *local_result,
    int my_pe, int n_pes, int count)
{
  if (blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    int prev_pe = (my_pe - 1 + n_pes) % n_pes;

    rocshmem_int_put_wave(sym_dst, sym_src, count, next_pe);
    rocshmem_barrier_all_wave();

    rocshmem_int_get_wave(local_result, sym_src, count, prev_pe);
  }
}

extern "C" __global__ void test_typed_amo_extended(
    int64_t *sym_val, int64_t *out_results, int my_pe)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem_int64_atomic_set(sym_val, 42, my_pe);
    int64_t fetched = rocshmem_int64_atomic_fetch(sym_val, my_pe);
    out_results[0] = fetched;

    int64_t old = rocshmem_int64_atomic_swap(sym_val, 99, my_pe);
    out_results[1] = old;

    int64_t final_val = rocshmem_int64_atomic_fetch(sym_val, my_pe);
    out_results[2] = final_val;
  }
}

extern "C" __global__ void test_typed_amo_bitwise(
    uint64_t *sym_val, uint64_t *out_results, int my_pe)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem_int64_atomic_set(reinterpret_cast<int64_t*>(sym_val), 0xFF, my_pe);
    uint64_t old = rocshmem_uint64_atomic_fetch_and(sym_val, 0x0F, my_pe);
    out_results[0] = old;

    uint64_t after_and = rocshmem_int64_atomic_fetch(
        reinterpret_cast<int64_t*>(sym_val), my_pe);
    out_results[1] = after_and;

    rocshmem_uint64_atomic_and(sym_val, 0x03, my_pe);
    uint64_t final_val = rocshmem_int64_atomic_fetch(
        reinterpret_cast<int64_t*>(sym_val), my_pe);
    out_results[2] = final_val;
  }
}

extern "C" __global__ void test_typed_int64_p(
    int64_t *sym_buf, int my_pe, int n_pes)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;
    rocshmem_int64_p(sym_buf, (int64_t)my_pe * 1000000LL + 12345LL, next_pe);
    rocshmem_barrier_all();
  }
}

extern "C" __global__ void test_typed_wait_vector(
    int *sym_arr, int *out_idx, int my_pe, int n_pes, int count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int next_pe = (my_pe + 1) % n_pes;

    for (int i = 0; i < count; i++)
      rocshmem_int_p(sym_arr + i, (my_pe + 1) * 10 + i, next_pe);
    rocshmem_quiet();

    int vals[4] = {0, 0, 0, 0};
    rocshmem_int_wait_until_any_vector(
        sym_arr, count, nullptr, 1 /* CMP_NE */, vals);

    rocshmem_int_wait_until_all_vector(
        sym_arr, count, nullptr, 1 /* CMP_NE */, vals);

    *out_idx = 1;
  }
}
