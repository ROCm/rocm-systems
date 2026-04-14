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

#include "primitive_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void PrimitiveTest(int loop, int skip, long long int *start_time,
                              long long int *end_time, char *source,
                              char *dest, size_t size, TestType type,
                              ShmemContextType ctx_type, int wf_size) {
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
  size_t offset = size * get_flat_id();
  source += offset;
  dest += offset;

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

    switch (type) {
      case GetTestType:
        rocshmem_ctx_getmem(ctx, dest, source, size, 1);
        break;
      case GetNBITestType:
        rocshmem_ctx_getmem_nbi(ctx, dest, source, size, 1);
        break;
      case PutTestType:
        rocshmem_ctx_putmem(ctx, dest, source, size, 1);
        break;
      case PutNBITestType:
        rocshmem_ctx_putmem_nbi(ctx, dest, source, size, 1);
        break;
      case PTestType:
        {
          /* Assigment required to verify we can send non-symetric memory */
          char val = *source;
          rocshmem_ctx_char_p(ctx, dest, val, 1);
        }
        break;
      case GTestType:
        *dest = rocshmem_ctx_char_g(ctx, source, 1);
        break;
      default:
        break;
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
  __syncthreads();

  if (t_id == 0) {
    start_time[wg_id] = wf_start_time[0];
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PrimitiveTester::PrimitiveTester(TesterArguments args) : Tester(args) {
  size_t buff_size = max_msg_size * args.wg_size * args.num_wgs;
  source = (char *)rocshmem_malloc(buff_size);
  dest = (char *)rocshmem_malloc(buff_size);

  if (source == nullptr || dest == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "source: " << source << ", dest: " << dest << std::endl;
    if (source) {
      rocshmem_free(source);
    }
    if (dest) {
      rocshmem_free(dest);
    }
    rocshmem_global_exit(1);
  }

  for(size_t i = 0; i < buff_size; i++) {
    source[i] = static_cast<char>('a' + i % 26);
  }
}

PrimitiveTester::~PrimitiveTester() {
  rocshmem_free(source);
  rocshmem_free(dest);
}

void PrimitiveTester::resetBuffers(size_t size) {
  size_t buff_size = size * args.wg_size * args.num_wgs;
  memset(dest, '1', buff_size);
}

void PrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                   size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(PrimitiveTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, source, dest,
                     size, _type, _shmem_context, wf_size);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void PrimitiveTester::verifyResults(size_t size) {
  int check_id =
      (_type == GetTestType || _type == GetNBITestType || _type == GTestType)
          ? 0
          : 1;

  if (args.myid == check_id) {
    size_t buff_size = size * args.wg_size * args.num_wgs;
    size_t verify_wg_size = std::min((size_t) 1024, buff_size);
    size_t verify_num_wgs = buff_size / verify_wg_size;

    hipLaunchKernelGGL(verify_results_kernel_char, verify_num_wgs, verify_wg_size, 0, stream,
                       source, dest, buff_size, verification_error);
    CHECK_HIP(hipStreamSynchronize(stream));

    if (*verification_error) {
      for (uint64_t i = 0; i < buff_size; i++) {
        if (dest[i] != source[i]) {
          std::cerr << "Data validation error at idx " << i << std::endl;
          std::cerr << " Got " << dest[i] << ", Expected "
                    << source[i] << std::endl;
          exit(-1);
        }
      }
      *verification_error = false;
    }
  }
}
