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

#include "flood_amo_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void FloodAmoTest(int loop, int skip, long long int *start_time,
                           long long int *end_time, uint64_t *s_buf,
                           TestType type, ShmemContextType ctx_type, int wf_size,
                           bool *verification_error, int *grid_psync) {
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

  //TODO collapse all threads of the wg to the same tgt_offset? (force atomic competition at target)
  auto tgt_offset {wg_id};

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      // Capture the start time of each wavefront to identify the earliest one
      wf_start_time[wf_id] = wall_clock64();
    }

    for (int j{0}; j < num_pe; j++) {
      // shuffle ordering so that threads in the wave put to a
      // different pe 'simultaneously'
      auto pe = (t_id + j) % num_pe;
      auto ret{0};
      switch (type) {
      case FloodAddTestType:
        rocshmem_ctx_uint64_atomic_add(ctx, &s_buf[tgt_offset], t_id+1, pe);
        break;
      case FloodFAddTestType:
        ret = rocshmem_ctx_uint64_atomic_fetch_add(ctx, &s_buf[tgt_offset], t_id+1, pe);
        //TODO check ret? How?
        break;
#if 0
      case Flood_FCswapTestType:
        dst_offset = pe * num_wg * num_th + t_offset; //TODO not used, use r_buf[dst_offset] for verif?
        auto ret = rocshmem_ctx_uint64_atomic_compare_swap(ctx, &s_buf[t_offset], 0, t_offset, pe);
        assert(ret == 0);
        ret = rocshmem_ctx_uint64_atomic_compare_swap(ctx, &s_buf[t_offset], t_offset, 0, pe);
        break;
#endif
      default:
        break;
      }
      __syncthreads();
      if (is_thread_zero_in_block()) {
        rocshmem_ctx_quiet(ctx);
      }
    }

    // We do verification for each iteration so performance will suffer,
    // thats fine it is a test not a benchmark.
    grid_barrier(grid_psync, num_wg * (2*i+1));
    if (is_block_zero_in_grid() && is_thread_zero_in_block())
      rocshmem_sync_all();
    if (is_thread_zero_in_block()) {
      uint64_t expected = (loop + skip) * num_pe * num_th * (num_th+1) / 2;
      if (expected != s_buf[wg_id] + 1) {
        printf("Data validation error (in kernel, iteration %d)\n"
               "  Expected %zd, got %zd\n",
               i, expected, s_buf[wg_id]);
        *verification_error = true;
      }
      s_buf[wg_id] = 0;
    }
    if (is_block_zero_in_grid() && is_thread_zero_in_block())
      rocshmem_sync_all();
    grid_barrier(grid_psync, num_wg * (2*i+2));
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

#if 0
static __global__ void verify_results_kernel(uint64_t *dest, size_t buf_size,
                                             bool *verification_error) {
  int num_pe {rocshmem_n_pes()};
  int num_wg {get_grid_num_blocks()};
  int num_th {get_flat_block_size()};
  int my_pe {rocshmem_my_pe()};
  int wg_id {get_flat_grid_id()};
  int t_id {get_flat_block_id()};

  auto t_offset {wg_id * num_th + t_id};
  if (*verification_error) {
    printf("VERIFICATION ERROR\n");
  }
#if 0
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
#endif
}
#endif

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
FloodAmoTester::FloodAmoTester(TesterArguments args) : Tester(args) {
  int num_pes {rocshmem_n_pes()};
  int my_pe {rocshmem_my_pe()};
  CHECK_HIP(hipMalloc(&grid_psync, sizeof(int)));
  s_buf = (uint64_t*)rocshmem_malloc(sizeof(uint64_t) * args.num_wgs);
#if 0
  for(int wg = 0; wg < args.num_wgs; wg++) for(int th = 0; th < args.wg_size; th++) {
    s_buf[wg * args.wg_size + th] = (((uint64_t)my_pe)<<44) + (wg<<12) + th; // set value for verification
  }
  r_buf = (uint64_t*)rocshmem_malloc(sizeof(uint64_t) * args.num_wgs * args.wg_size * num_pes);
#endif
}

FloodAmoTester::~FloodAmoTester() {
  rocshmem_free(s_buf);
  CHECK_HIP(hipFree(grid_psync));
#if 0
  rocshmem_free(r_buf);
#endif
}

void FloodAmoTester::resetBuffers(size_t size) {
  int num_pes {rocshmem_n_pes()};
  memset(s_buf, 0, sizeof(uint64_t) * args.num_wgs);
  *grid_psync = 0;
}

void FloodAmoTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                size_t size) {
  size_t shared_bytes = 0;
  int num_pes {rocshmem_n_pes()};

  hipLaunchKernelGGL(FloodAmoTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, s_buf,
                     _type, _shmem_context, wf_size, verification_error, grid_psync);


  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x * num_pes;
  num_timed_msgs = loop * gridSize.x * blockSize.x * num_pes;
}

void FloodAmoTester::verifyResults(size_t size) {
  int num_pes {rocshmem_n_pes()};
  int my_pe {rocshmem_my_pe()};

  // TODO: update: overflow of uint64_t is the can't test case
  if (num_pes > 1<<20 || args.num_wgs > 1<<31 || args.wg_size > 1<<12) {
    // can't check
    return;
  }
  assert(size == sizeof(uint64_t));

#if 0
  hipLaunchKernelGGL(verify_results_kernel, args.num_wgs, args.wg_size, 0, stream,
                     r_buf, sizeof(uint64_t), verification_error);
  CHECK_HIP(hipStreamSynchronize(stream));
#endif

  if (*verification_error) {
    std::cerr << "Data validation error (found in kernel)" << std::endl;
#if 0
    uint64_t expected = (args.loop + args.skip) * num_pes * args.wg_size * (args.wg_size+1) / 2;
    for(auto wg = 0; wg < args.num_wgs; wg++) {
      if (expected != s_buf[wg] + 1) {
        std::cerr << "Data validation error for wg " << wg << std::endl;
        std::cerr << " Got " << s_buf[wg]
                  << ", Expected " << expected << std::endl;
      }
    }
#endif
    *verification_error = false;
  }
}
