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
                              ShmemContextType ctx_type, int wf_size,
                              int batch, int *grid_psync) {
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
  // Each thread owns `batch` contiguous slots of `size` bytes.
  source += size * batch * get_flat_id();
  dest += size * batch * get_flat_id();

  // Choose start_slot so that after `skip` iterations slot wraps to 0.
  int start_slot = (batch - (skip % batch)) % batch;

  for (int i = 0; i < loop + skip; i++) {
    size_t offset = ((start_slot + i) % batch) * size;

    // Quiet at batch boundaries to allow safe buffer reuse
    if (offset == 0) {
      __syncthreads();
      if(is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
      __syncthreads();
      if (i == skip) {
        // Global barrier ensures all WGs have finished their skip-region
        // puts before any WG starts timing, preventing skip traffic from
        // contaminating the timed window.
        grid_barrier(grid_psync, gridDim.x);
        // Capture the start time of each wavefront to identify the earliest one
        wf_start_time[wf_id] = wall_clock64();
      }
    }

    switch (type) {
      case GetTestType:
        rocshmem_ctx_getmem(ctx, dest + offset, source + offset, size, 1);
        break;
      case GetNBITestType:
        rocshmem_ctx_getmem_nbi(ctx, dest + offset, source + offset, size, 1);
        break;
      case PutTestType:
        rocshmem_ctx_putmem(ctx, dest + offset, source + offset, size, 1);
        break;
      case PutNBITestType:
        rocshmem_ctx_putmem_nbi(ctx, dest + offset, source + offset, size, 1);
        break;
      case PTestType:
        {
          /* Assignment required to verify we can send non-symetric memory */
          char val = source[offset];
          rocshmem_ctx_char_p(ctx, dest + offset, val, 1);
        }
        break;
      case GTestType:
        dest[offset] = rocshmem_ctx_char_g(ctx, source + offset, 1);
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
  size_t buff_size = max_msg_size * batch_size * args.wg_size * args.num_wgs;
  char *local = (char *) alloc_test_buffer(buff_size, args.local_buf_type);
  char *remote = (char *) alloc_test_buffer(buff_size);
  CHECK_HIP(hipMalloc(&grid_psync, sizeof(int)));

  int max_co_resident_wgs_per_cu = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_co_resident_wgs_per_cu, PrimitiveTest, args.wg_size, 0));
  const int max_sustainable_wgs =
      max_co_resident_wgs_per_cu * deviceProps.multiProcessorCount;
  if (args.num_wgs > static_cast<unsigned>(max_sustainable_wgs)) {
    std::cerr << "Error: Requested work-groups (" << args.num_wgs
              << ") exceeds max co-resident work-groups (" << max_sustainable_wgs
              << "). Reduce -w to avoid grid_barrier deadlock." << std::endl;
    exit(-1);
  }

  switch (_type) {
    case PutTestType:
    case PutNBITestType:
    case PTestType:
      source = local;
      dest = remote;
      break;
    case GetTestType:
    case GetNBITestType:
    case GTestType:
    default:
      dest = local;
      source = remote;
      break;
  }

  CHECK_HIP(hipMemsetAsync(source, 'a', buff_size, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
}

PrimitiveTester::~PrimitiveTester() {
  char *local = nullptr;
  char *remote = nullptr;

  switch (_type) {
    case PutTestType:
    case PutNBITestType:
    case PTestType:
      local = source;
      remote = dest;
      break;
    case GetTestType:
    case GetNBITestType:
    case GTestType:
    default:
      local = dest;
      remote = source;
      break;
  }

  free_test_buffer(local, args.local_buf_type);
  free_test_buffer(remote);
  CHECK_HIP(hipFree(grid_psync));
}

void PrimitiveTester::resetBuffers(size_t size) {
  size_t buff_size = size * batch_size * args.wg_size * args.num_wgs;
  CHECK_HIP(hipMemsetAsync(dest, '1', buff_size, stream));
  CHECK_HIP(hipMemsetAsync(grid_psync, 0, sizeof(int), stream));
  CHECK_HIP(hipStreamSynchronize(stream));
}

void PrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                   size_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(PrimitiveTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, source, dest,
                     size, _type, _shmem_context, wf_size,
                     batch_size, grid_psync);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void PrimitiveTester::verifyResults(size_t size) {
  int check_id =
      (_type == GetTestType || _type == GetNBITestType || _type == GTestType)
          ? 0
          : 1;

  if (args.myid == check_id) {
    int start_slot = (batch_size - (args.skip % batch_size)) % batch_size;
    int verify_iters = std::min(batch_size, num_loops + args.skip);
    size_t buf_bytes = size * batch_size;
    size_t concurrency = args.wg_size * args.num_wgs;
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
