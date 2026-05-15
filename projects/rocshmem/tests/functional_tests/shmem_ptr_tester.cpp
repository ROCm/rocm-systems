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

#include "shmem_ptr_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void ShmemPtrTest(int loop, int skip, long long int *start_time,
                             long long int *end_time, char *dest, int wf_size,
                             ShmemContextType ctx_type, int *available) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int t_id  = get_flat_block_id();
  int wf_id = t_id / wf_size;

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  /**
   * Shared array to capture the start time for each wavefront
   * Max threads per block = 1024, wavefront size = 64 or 32 depending
   * on the GPUs. Using 32 since its safer for the dimensioning of the array,
   * the last 16 elements will not be used on GPUs with a wf size of 64.
   * Maximum array size required = 1024/32 = 32
   */
  __shared__ long long int wf_start_time[32];


  /**
   * Calculate start index for each thread within the grid
   */
  dest += get_flat_id();

  char *local_addr = dest;
  void *remote_addr = rocshmem_ptr((void *)local_addr, 1);
  if (remote_addr != NULL) {
    *available = 1;
  }

  if(*available) {
    for (int i = 0; i < loop + skip; i++) {
      if (i == skip) {
        __syncthreads();
        // Ensures all RMA calls from the skip loops are completed
        if(is_thread_zero_in_block()) {
          rocshmem_ctx_quiet(ctx);
        }
        __syncthreads();
        // Capture the start time of each wavefront to identify the earliest one
        wf_start_time[wf_id] = wall_clock64();
      }

      ((char *)remote_addr)[0] = '1';
    }
  }

  __syncthreads();
  if(is_thread_zero_in_block()) {
    rocshmem_ctx_quiet(ctx);
  }

  /**
   * End time of the last wavefront is recorded by overwriting
   * the value previously set by earlier wavefronts.
   */
  end_time[wg_id] = wall_clock64();

  // Find the earliest start time
  int num_wfs = (get_flat_block_size() - 1 ) / wf_size + 1;
  for (int i = num_wfs / 2; i > 0; i >>= 1 ) {
    if(t_id < i) {
      wf_start_time[t_id] = min(wf_start_time[t_id], wf_start_time[t_id + i]);
    }
  }

  // For data validation in remote PE
  if( get_flat_id() == 0 ) {
    int *store_avail = (int*)(dest + get_flat_grid_size());
    *store_avail = *available;
    rocshmem_ctx_int_put(ctx, store_avail, store_avail, 1, 1);
  }

  __syncthreads();

  if (t_id == 0) {
    start_time[wg_id] = wf_start_time[0];
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
ShmemPtrTester::ShmemPtrTester(TesterArguments args) : Tester(args) {
  size_t buff_size = args.wg_size * args.num_wgs + sizeof(int);
  CHECK_HIP(hipMalloc((void **)&_available, sizeof(int)));
  dest = (char *)rocshmem_malloc(buff_size);

  if (dest == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "dest: " << dest << std::endl;

    rocshmem_global_exit(1);
  }
}

ShmemPtrTester::~ShmemPtrTester() {
  CHECK_HIP(hipFree(_available));
  rocshmem_free(dest);
}

void ShmemPtrTester::resetBuffers(size_t size) {
  size_t buff_size = args.wg_size * args.num_wgs + sizeof(int);
  memset(dest, '0', buff_size);
  memset(_available, 0, sizeof(int));
}

void ShmemPtrTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(ShmemPtrTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     dest, wf_size, _type, _available);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void ShmemPtrTester::verifyResults(size_t size) {
  if (args.myid == 0) {
    if (*_available == 0) {
      _print_results = false;
      std::cout << "rocshmem ptr not available\n" << std::endl;
    }
  }
  else {
    size_t buff_size = args.wg_size * args.num_wgs;
    int *available = (int*)(dest + buff_size);
    if(*available == 1) {
      for (size_t i = 0; i < buff_size; i++) {
        if (dest[i] != '1') {
          std::cerr << "Data validation error at idx " << i << std::endl;
          std::cerr << " Got " << dest[i] << ", Expected 1 " << std::endl;
          exit(-1);
        }
      }
    }
  }
}
