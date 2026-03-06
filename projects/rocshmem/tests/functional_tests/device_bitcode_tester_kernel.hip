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

extern "C" {
  __device__ int rocshmem_my_pe();
  __device__ int rocshmem_n_pes();
  __device__ void rocshmem_int_p(int *dest, int value, int pe);
  __device__ void rocshmem_putmem(void *dest, const void *source,
                                  size_t nbytes, int pe);
  __device__ void rocshmem_getmem(void *dest, const void *source,
                                  size_t nbytes, int pe);
  __device__ void rocshmem_barrier_all();
  __device__ void rocshmem_fence();
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

    rocshmem_putmem(sym_dst, sym_src,
                    count * sizeof(int), next_pe);
    rocshmem_barrier_all();

    for (int i = 0; i < count; i++)
      local_result[i] = sym_dst[i];
  }
}
