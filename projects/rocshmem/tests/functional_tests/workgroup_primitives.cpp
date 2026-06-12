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

#include <rocshmem/rocshmem.hpp>

#include <numeric>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void WorkGroupPrimitiveTest(int loop, int skip,
                                      long long int *start_time,
                                      long long int *end_time, char *source,
                                      char *dest, size_t size, TestType type,
                                      ShmemContextType ctx_type, int batch,
                                      int *grid_psync) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  // Calculate start index for each work group
  // Each workgroup owns `batch` contiguous slots of `size` bytes.
  source += size * batch * wg_id;
  dest += size * batch * wg_id;

  // Choose start_slot so that after `skip` iterations slot wraps to 0.
  int start_slot = (batch - (skip % batch)) % batch;

  for (int i = 0; i < loop + skip; i++) {
    size_t offset = ((start_slot + i) % batch) * size;

    // Quiet at batch boundaries to allow safe buffer reuse
    if (offset == 0) {
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
      __syncthreads();
      if (i == skip) {
        // Global barrier ensures all WGs have finished their skip-region
        // puts before any WG starts timing, preventing skip traffic from
        // contaminating the timed window.
        grid_barrier(grid_psync, gridDim.x);
        start_time[wg_id] = wall_clock64();
      }
    }

    switch (type) {
      case WGGetTestType:
        rocshmem_ctx_getmem_wg(ctx, dest + offset, source + offset, size, 1);
        break;
      case WGGetNBITestType:
        rocshmem_ctx_getmem_nbi_wg(ctx, dest + offset, source + offset, size, 1);
        break;
      case WGPutTestType:
        rocshmem_ctx_putmem_wg(ctx, dest + offset, source + offset, size, 1);
        break;
      case WGPutNBITestType:
        rocshmem_ctx_putmem_nbi_wg(ctx, dest + offset, source + offset, size, 1);
        break;
      default:
        break;
    }
  }

  __syncthreads();
  if (is_thread_zero_in_block()) {
    rocshmem_ctx_quiet(ctx);
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
WorkGroupPrimitiveTester::WorkGroupPrimitiveTester(TesterArguments args)
    : Tester(args) {
  size_t buff_size = max_msg_size * batch_size * args.num_wgs;
  char *local = (char *) alloc_test_buffer(buff_size, args.local_buf_type);
  char *remote = (char *) alloc_test_buffer(buff_size);
  CHECK_HIP(hipMalloc(&grid_psync, sizeof(int)));

  int max_co_resident_wgs_per_cu = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_co_resident_wgs_per_cu, WorkGroupPrimitiveTest, args.wg_size, 0));
  const int max_sustainable_wgs =
      max_co_resident_wgs_per_cu * deviceProps.multiProcessorCount;
  if (args.num_wgs > static_cast<unsigned>(max_sustainable_wgs)) {
    std::cout << "Warning: Requested work-groups (" << args.num_wgs
              << ") exceeds max co-resident work-groups (" << max_sustainable_wgs
              << "). Capping to " << max_sustainable_wgs
              << " to avoid grid_barrier deadlock." << std::endl;
    args.num_wgs = max_sustainable_wgs;
  }

  switch (_type) {
    case WGPutTestType:
    case WGPutNBITestType:
      source = local;
      dest = remote;
      break;
    case WGGetTestType:
    case WGGetNBITestType:
    default:
      dest = local;
      source = remote;
      break;
  }

  CHECK_HIP(hipMemset(source, 'a', buff_size));
}

WorkGroupPrimitiveTester::~WorkGroupPrimitiveTester() {
  char *local = nullptr;
  char *remote = nullptr;

  switch (_type) {
    case WGPutTestType:
    case WGPutNBITestType:
      local = source;
      remote = dest;
      break;
    case WGGetTestType:
    case WGGetNBITestType:
    default:
      local = dest;
      remote = source;
      break;
  }

  free_test_buffer(local, args.local_buf_type);
  free_test_buffer(remote);
  CHECK_HIP(hipFree(grid_psync));
}

void WorkGroupPrimitiveTester::resetBuffers(size_t size) {
  size_t buff_size = size * batch_size * args.num_wgs;
  CHECK_HIP(hipMemsetAsync(dest, '1', buff_size, stream));
  CHECK_HIP(hipMemsetAsync(grid_psync, 0, sizeof(int), stream));
}

void WorkGroupPrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(WorkGroupPrimitiveTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     source, dest, size, _type, _shmem_context,
                     batch_size, grid_psync);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void WorkGroupPrimitiveTester::verifyResults(size_t size) {
  int check_id = (_type == WGGetTestType || _type == WGGetNBITestType)
                     ? 0
                     : 1;

  if (args.myid == check_id) {
    int start_slot = (batch_size - (args.skip % batch_size)) % batch_size;
    int verify_iters = std::min(batch_size, num_loops + args.skip);
    size_t buf_bytes = size * batch_size;
    size_t concurrency = args.num_wgs;
    size_t total = size * verify_iters * concurrency;
    size_t verify_wg_size = std::min((size_t) 1024, total);
    size_t verify_num_wgs = (total + verify_wg_size - 1) / verify_wg_size;

    hipLaunchKernelGGL(verify_results_kernel_char, verify_num_wgs, verify_wg_size, 0, stream,
                       dest, size, buf_bytes, concurrency,
                       num_loops, args.skip, batch_size, verification_error);
    CHECK_HIP(hipStreamSynchronize(stream));

    if (*verification_error) {
      for (size_t b = 0; b < concurrency; b++) {
        for (int iter = 0; iter < verify_iters; iter++) {
          int slot = (start_slot + iter) % batch_size;
          for (size_t i = 0; i < size; i++) {
            if (dest[b * buf_bytes + slot * size + i] != 'a') {
              std::cerr << "Data validation error at buffer " << b
                        << " slot " << slot << " idx " << i << std::endl;
              std::cerr << " Got " << (int)(unsigned char)dest[b * buf_bytes + slot * size + i]
                        << ", Expected " << (int)(unsigned char)'a'
                        << std::endl;
              exit(-1);
            }
          }
        }
      }
      *verification_error = false;
    }
  }
}
