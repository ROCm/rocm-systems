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

  auto tgt_offset {(wg_id + 1) % num_wg}; //for npes=1: avoid writting and reading from same wg

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      // Capture the start time of each wavefront to identify the earliest one
      wf_start_time[wf_id] = wall_clock64();
    }

    for (int j{0}; j < num_pe; j++) {
      // shuffle ordering so that threads in the wave put to a
      // different pe 'simultaneously'
      auto pe = (t_id + j) % num_pe;
      [[maybe_unused]] uint64_t ret{0};
      switch (type) {
      case FloodWaitAmoTestType:
      case FloodAddTestType:
        rocshmem_ctx_uint64_atomic_add(ctx, &s_buf[tgt_offset], t_id+1, pe);
        break;
      case FloodFAddTestType:
        ret = rocshmem_ctx_uint64_atomic_fetch_add(ctx, &s_buf[tgt_offset], t_id+1, pe);
        //TODO check ret? How?
        break;
      default:
        break;
      }
      if (type != FloodWaitAmoTestType) {
        __syncthreads();
        if (is_thread_zero_in_block()) {
          rocshmem_ctx_quiet(ctx);
        }
      }
    }

    // We do verification for each iteration so performance will suffer,
    // that's fine it is a test not a benchmark.
    uint64_t expected = static_cast<uint64_t>(i+1) * num_pe * (num_th * (num_th+1)) / 2;
    switch (type) {
    case FloodWaitAmoTestType:
      if (is_thread_zero_in_block()) {
        rocshmem_uint64_wait_until(&s_buf[wg_id], ROCSHMEM_CMP_EQ, expected);
        uint64_t observed = s_buf[wg_id];
        // if test detects an atomicity issue in the library, it may
        //   a) deadlock in wait_until,
        //   b) overshoot and trigger this case
        if (expected != observed) {
          printf("Data validation error (pe %d, wg %d, iteration %d)\n"
                 "  Expected %ld, got %ld\n",
                 my_pe, wg_id, i, expected, observed);
          *verification_error = true;
        }
      }
      // still sync so that iterations remain synchronous.
      grid_barrier(&grid_psync[0], num_wg * (i+1));
      if (is_block_zero_in_grid() && is_thread_zero_in_block())
        rocshmem_sync_all();
      grid_barrier(&grid_psync[1], num_wg * (i+1));
      break;
    case FloodAddTestType:
    case FloodFAddTestType:
      grid_barrier(&grid_psync[0], num_wg * (i+1));
      if (is_block_zero_in_grid() && is_thread_zero_in_block())
        rocshmem_sync_all();
      grid_barrier(&grid_psync[1], num_wg * (i+1));
      if (is_thread_zero_in_block()) {
        //uint64_t observed = __hip_atomic_load(&s_buf[wg_id], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        //uint64_t observed = static_cast<volatile uint64_t>(s_buf[wg_id]);
        uint64_t observed = s_buf[wg_id];
        if (expected != observed) {
          printf("Data validation error (pe %d, wg %d, iteration %d)\n"
                 "  Expected %ld, got %ld\n",
                 my_pe, wg_id, i, expected, observed);
          *verification_error = true;
        }
      }
      grid_barrier(&grid_psync[2], num_wg * (i+1));
      if (is_block_zero_in_grid() && is_thread_zero_in_block())
        rocshmem_sync_all();
      grid_barrier(&grid_psync[3], num_wg * (i+1));
      break;
    default:
      break;
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

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
FloodAmoTester::FloodAmoTester(TesterArguments args) : Tester(args) {
  int num_pes {rocshmem_n_pes()};
  CHECK_HIP(hipMalloc(&grid_psync, 4 * sizeof(int)));
  s_buf = (uint64_t*)rocshmem_malloc(sizeof(uint64_t) * args.num_wgs);
  /**
   * Warn about boundary conditions on num-wgs etc.
   */
  uint64_t limit = std::numeric_limits<uint64_t>::max() / (args.loop + args.skip) / num_pes / args.wg_size / (args.wg_size+1);
  if (0 == limit) {
    std::cout << "Warning: Number of iterations (" << args.loop + args.skip
              << ") will cause overflow during verification." << std::endl
              << "  Reduce -n or -nskip." << std::endl;
  }
  /**
   * Calculate the maximum number of co-resident work-groups per compute unit
   * based on the resource usage of the kernel
   */
  int max_co_resident_wgs_per_cu = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_co_resident_wgs_per_cu,
      FloodAmoTest,
      args.wg_size,
      0));
  // Get the number of compute units
  hipDeviceProp_t device_prop;
  CHECK_HIP(hipGetDeviceProperties(&device_prop, 0));
  const int num_cus = device_prop.multiProcessorCount;
  const int max_sustainable_wgs = max_co_resident_wgs_per_cu * num_cus;

  // Print warning if num_wgs exceeds max co-resident work-groups
  if (static_cast<unsigned int>(args.num_wgs) > static_cast<unsigned int>(max_sustainable_wgs)) {
    std::cout << "Warning: Number of work-groups (" << args.num_wgs
              << ") exceeds max sustainable work-groups ("
              << max_sustainable_wgs << ")." << std::endl;
  }
}

FloodAmoTester::~FloodAmoTester() {
  rocshmem_free(s_buf);
  CHECK_HIP(hipFree(grid_psync));
}

void FloodAmoTester::resetBuffers([[maybe_unused]] size_t size) {
  CHECK_HIP(hipMemset(s_buf, 0, sizeof(uint64_t) * args.num_wgs));
  CHECK_HIP(hipMemset(grid_psync, 0, 4 * sizeof(int)));
}

void FloodAmoTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                [[maybe_unused]] size_t size) {
  size_t shared_bytes = 0;
  int num_pes {rocshmem_n_pes()};

  hipLaunchKernelGGL(FloodAmoTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, s_buf,
                     _type, _shmem_context, wf_size, verification_error, grid_psync);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x * num_pes;
  num_timed_msgs = loop * gridSize.x * blockSize.x * num_pes;
}

void FloodAmoTester::verifyResults([[maybe_unused]] size_t size) {
  int num_pes {rocshmem_n_pes()};

  assert(size == sizeof(uint64_t));

  if (*verification_error) {
    std::cerr << "Data validation error (found by device kernel)" << std::endl;
    uint64_t expected = static_cast<uint64_t>(args.loop + args.skip) * num_pes * (args.wg_size * (args.wg_size+1)) / 2;
    for(unsigned int wg = 0; wg < static_cast<unsigned int>(args.num_wgs); wg++) {
      if (expected != s_buf[wg]) {
        std::cerr << "Data validation error for wg " << wg << std::endl;
        std::cerr << " Got " << s_buf[wg]
                  << ", Expected " << expected << std::endl;
      }
    }
    *verification_error = false;
  }
}
