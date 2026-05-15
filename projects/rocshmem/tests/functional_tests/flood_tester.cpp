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

#include "flood_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void FloodTest(int loop, int skip, long long int *start_time,
                           long long int *end_time, uint64_t *r_buf, uint64_t *s_buf,
                           TestType type, ShmemContextType ctx_type, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;

  /**
   * Shared array to capture the start time for each wavefront
   * Max threads per block = 1024, wavefront size = 64 or 32 depending
   * on the GPUs. Using 32 since its safer for the dimensioning of the array,
   * the last 16 elements will not be used on GPUs with a wf size of 64.
   * Maximum array size required = 1024/32 = 32
   */
  __shared__ long long int wf_start_time[32];

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  int num_pe {rocshmem_ctx_n_pes(ctx)};
  int num_wg {get_grid_num_blocks()};
  int num_th {get_flat_block_size()};
  int my_pe {rocshmem_ctx_my_pe(ctx)};
  int wg_id {get_flat_grid_id()};
  int t_id {get_flat_block_id()};
  int wf_id {t_id / wf_size};

  auto t_offset {wg_id * num_th + t_id};
  auto tgt_offset {my_pe * num_wg * num_th + t_offset};
  auto dst_offset {0};

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      // Capture the start time of each wavefront to identify the earliest one
      wf_start_time[wf_id] = wall_clock64();
    }

    for (int j{0}; j < num_pe; j++) {
      // shuffle ordering so that threads in the wave put to a
      // different pe 'simultaneously'
      auto pe = (t_id + j) % num_pe;
      switch (type) {
      case FloodPutTestType:
        rocshmem_ctx_putmem(ctx, &r_buf[tgt_offset], &s_buf[t_offset], sizeof(uint64_t), pe);
        break;
      case FloodPutNBITestType:
        rocshmem_ctx_putmem_nbi(ctx, &r_buf[tgt_offset], &s_buf[t_offset], sizeof(uint64_t), pe);
        break;
      case FloodPTestType:
        rocshmem_ctx_ulong_p(ctx, &r_buf[tgt_offset], s_buf[t_offset], pe);
        break;
      case FloodGetTestType:
        dst_offset = pe * num_wg * num_th + t_offset;
        rocshmem_ctx_getmem(ctx, &r_buf[dst_offset], &s_buf[t_offset], sizeof(uint64_t), pe);
        break;
      case FloodGetNBITestType:
        dst_offset = pe * num_wg * num_th + t_offset;
        rocshmem_ctx_getmem_nbi(ctx, &r_buf[dst_offset], &s_buf[t_offset], sizeof(uint64_t), pe);
        break;
      case FloodGTestType:
        dst_offset = pe * num_wg * num_th + t_offset;
        r_buf[dst_offset] = rocshmem_ctx_ulong_g(ctx, &s_buf[t_offset], pe);
        break;
      default:
        break;
      }
      __syncthreads();
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
    }
  }

  __syncthreads();
  if (is_thread_zero_in_wave()) {
    end_time[wg_id] = wall_clock64();
  }
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

static __global__ void verify_results_kernel(uint64_t *dest, size_t buf_size,
                                             bool *verification_error) {
  int num_pe {rocshmem_n_pes()};
  int num_wg {get_grid_num_blocks()};
  int num_th {get_flat_block_size()};
  int my_pe {rocshmem_my_pe()};
  int wg_id {get_flat_grid_id()};
  int t_id {get_flat_block_id()};

  auto t_offset {wg_id * num_th + t_id};

  for (int pe{0}; pe < num_pe; pe++) {
    auto dst_offset {pe * num_wg * num_th + t_offset};
    auto value = dest[dst_offset];
    auto v_th = value & 0x0fff;
    auto v_wg = (value>>12) & 0xffff'ffff;
    auto v_pe = (value>>44);

    if (v_th != t_id || v_wg != wg_id || v_pe != pe) {
      *verification_error = true;
    }
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
FloodTester::FloodTester(TesterArguments args) : Tester(args) {
  int num_pes {rocshmem_n_pes()};
  int my_pe {rocshmem_my_pe()};
  s_buf = (uint64_t*)rocshmem_malloc(sizeof(uint64_t) * args.num_wgs * args.wg_size);
  for(int wg = 0; wg < args.num_wgs; wg++) for(int th = 0; th < args.wg_size; th++) {
    s_buf[wg * args.wg_size + th] = (((uint64_t)my_pe)<<44) + (wg<<12) + th; // set value for verification
  }
  r_buf = (uint64_t*)rocshmem_malloc(sizeof(uint64_t) * args.num_wgs * args.wg_size * num_pes);
}

FloodTester::~FloodTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
}

void FloodTester::resetBuffers(size_t size) {
  int num_pes {rocshmem_n_pes()};
  memset(r_buf, 0, sizeof(uint64_t) * args.num_wgs * args.wg_size * num_pes);
}

void FloodTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                size_t size) {
  size_t shared_bytes = 0;
  int num_pes {rocshmem_n_pes()};

  hipLaunchKernelGGL(FloodTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, r_buf, s_buf,
                     _type, _shmem_context, wf_size);


  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x * num_pes;
  num_timed_msgs = loop * gridSize.x * blockSize.x * num_pes;
}

void FloodTester::verifyResults(size_t size) {
  int num_pes {rocshmem_n_pes()};
  int my_pe {rocshmem_my_pe()};

  if (num_pes > 1<<20 || args.num_wgs > 1<<31 || args.wg_size > 1<<12) {
    // can't check
    return;
  }
  assert(size == sizeof(uint64_t));

  hipLaunchKernelGGL(verify_results_kernel, args.num_wgs, args.wg_size, 0, stream,
                     r_buf, sizeof(uint64_t), verification_error);
  CHECK_HIP(hipStreamSynchronize(stream));

  if (*verification_error) {
    for(auto pe = 0; pe < num_pes; pe++)
      for(auto wg = 0; wg < args.num_wgs; wg++)
        for(auto th = 0; th < args.wg_size; th++) {
      auto t_offset {wg * args.wg_size + th};
      auto dst_offset {pe * args.num_wgs * args.wg_size + t_offset};
      auto value = r_buf[dst_offset];
      auto v_th = value & 0x0fff;
      auto v_wg = (value>>12) & 0xffff'ffff;
      auto v_pe = (value>>44);
      if (v_th != th || v_wg != wg || v_pe != pe) {
        std::cerr << "Data validation error at idx " << dst_offset << std::endl;
        std::cerr << " Got " << v_pe << ":" << v_wg << ":" << v_th
                  << ", Expected " << pe << ":" << wg << ":" << th << std::endl;

        *verification_error = false;
      }
    }
  }
}
