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

#include "workgroup_primitives.hpp"
#include "shmem_api_adapter.hpp"

#include <numeric>

using namespace shmem_adapter;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void WorkGroupPrimitiveTest(int loop, int skip,
                                      long long int *start_time,
                                      long long int *end_time, char *source,
                                      char *dest, size_t size, TestType type,
                                      ShmemContextType ctx_type) {
  __shared__ shmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  shmem_wg_ctx_create(ctx_type, &ctx);

  // Calculate start index for each work group
  size_t offset = size * wg_id;
  source += offset;
  dest += offset;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      // Ensures all RMA calls from the skip loops are completed
      if (is_thread_zero_in_block()) {
        shmem_quiet(ctx);
      }
      __syncthreads();
      start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case WGGetTestType:
        shmem_getmem_wg(ctx, dest, source, size, 1);
        break;
      case WGGetNBITestType:
        shmem_getmem_nbi_wg(ctx, dest, source, size, 1);
        break;
      case WGPutTestType:
        shmem_putmem_wg(ctx, dest, source, size, 1);
        break;
      case WGPutNBITestType:
        shmem_putmem_nbi_wg(ctx, dest, source, size, 1);
        break;
      default:
        break;
    }
  }

  if (is_thread_zero_in_block()) {
    shmem_quiet(ctx);
    end_time[wg_id] = wall_clock64();
  }

  shmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
WorkGroupPrimitiveTester::WorkGroupPrimitiveTester(TesterArguments args)
    : Tester(args) {
  size_t buff_size = max_msg_size * args.num_wgs;
  source = (char *)shmem_malloc(buff_size);
  dest = (char *)shmem_malloc(buff_size);

  if (source == nullptr || dest == nullptr) {
    std::cerr << "Error allocating memory from symmetric heap" << std::endl;
    std::cerr << "source: " << source << ", dest: " << dest << std::endl;
    if (source) {
      shmem_free(source);
    }
    if (dest) {
      shmem_free(dest);
    }
    shmem_global_exit(1);
  }

  for(size_t i = 0; i < buff_size; i++) {
    source[i] = static_cast<char>('a' + i % 26);
  }
}

WorkGroupPrimitiveTester::~WorkGroupPrimitiveTester() {
  shmem_free(source);
  shmem_free(dest);
}

void WorkGroupPrimitiveTester::resetBuffers(size_t size) {
  size_t buff_size = size * args.num_wgs;
  memset(dest, '1', buff_size);
}

void WorkGroupPrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(WorkGroupPrimitiveTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     source, dest, size, _type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void WorkGroupPrimitiveTester::verifyResults(size_t size) {
  int check_id = (_type == WGGetTestType || _type == WGGetNBITestType)
                     ? 0
                     : 1;

  if (args.myid == check_id) {
    size_t buff_size = size * args.num_wgs;
    size_t verify_wg_size = std::min((size_t) 1024, buff_size);
    size_t verify_num_wgs = buff_size / verify_wg_size;

    hipLaunchKernelGGL(rocshmem::verify_results_kernel_char, verify_num_wgs, verify_wg_size, 0, stream,
                       source, dest, buff_size, verification_error);
    CHECK_HIP(hipStreamSynchronize(stream));

    if (*verification_error) {
      for (size_t i = 0; i < buff_size; i++) {
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
