/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "default_ctx_fadd_tester.hpp"

#include <algorithm>
#include <vector>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void DefaultCTXFAddTest(int loop, int skip, long long int *start_time, long long int *end_time,
                                   uint64_t *counter, uint64_t *fetched) {
  int wg_id = get_flat_grid_id();
  size_t num_threads = static_cast<size_t>(get_grid_num_blocks()) * get_flat_block_size();
  size_t t_id = get_flat_id();

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && is_thread_zero_in_block()) {
      start_time[wg_id] = wall_clock64();
    }
    fetched[i * num_threads + t_id] = rocshmem_uint64_atomic_fetch_add(counter, 1, 1);
  }

  __syncthreads();
  if (is_thread_zero_in_block()) {
    rocshmem_quiet();
    end_time[wg_id] = wall_clock64();
  }
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
DefaultCTXFAddTester::DefaultCTXFAddTester(TesterArguments args) : Tester(args) {
  int max_loop = std::max(args.loop, args.loop_large);
  max_fetches = static_cast<size_t>(max_loop + args.skip) * args.num_wgs * args.wg_size;
  counter = static_cast<uint64_t *>(alloc_test_buffer(sizeof(uint64_t)));
  CHECK_HIP(hipMalloc(&fetched, max_fetches * sizeof(uint64_t)));
}

DefaultCTXFAddTester::~DefaultCTXFAddTester() {
  free_test_buffer(counter);
  CHECK_HIP(hipFree(fetched));
}

void DefaultCTXFAddTester::resetBuffers([[maybe_unused]] uint64_t size) {
  CHECK_HIP(hipMemsetAsync(counter, 0, sizeof(uint64_t), stream));
  CHECK_HIP(hipMemsetAsync(fetched, 0xff, max_fetches * sizeof(uint64_t), stream));
  CHECK_HIP(hipStreamSynchronize(stream));
}

void DefaultCTXFAddTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop, [[maybe_unused]] uint64_t size) {
  size_t shared_bytes = 0;
  hipLaunchKernelGGL(DefaultCTXFAddTest, gridSize, blockSize, shared_bytes, stream, loop, args.skip, start_time,
                     end_time, counter, fetched);

  num_fetches = static_cast<size_t>(loop + args.skip) * gridSize.x * blockSize.x;
  num_msgs = num_fetches;
  num_timed_msgs = static_cast<size_t>(loop) * gridSize.x * blockSize.x;
}

void DefaultCTXFAddTester::verifyResults([[maybe_unused]] uint64_t size) {
  size_t expected = static_cast<size_t>(num_loops + args.skip) * args.num_wgs * args.wg_size;

  // PE 1 owns the counter: every fetch_add must have landed exactly once.
  if (args.myid == 1) {
    uint64_t final_value = 0;
    CHECK_HIP(hipMemcpy(&final_value, counter, sizeof(uint64_t), hipMemcpyDeviceToHost));
    if (final_value != expected) {
      std::cerr << "Data validation error: counter is " << final_value << ", expected " << expected << std::endl;
      exit(-1);
    }
  }

  // PE 0 issued the fetch_adds: the returned values must be exactly 0 .. expected - 1, each once.
  if (args.myid == 0) {
    std::vector<uint64_t> values(expected);
    CHECK_HIP(hipMemcpy(values.data(), fetched, expected * sizeof(uint64_t), hipMemcpyDeviceToHost));
    std::sort(values.begin(), values.end());
    size_t duplicates = 0;
    size_t out_of_range = 0;
    for (size_t i = 0; i < expected; i++) {
      if (values[i] >= expected) {
        out_of_range++;
      }
      if (i > 0 && values[i] == values[i - 1]) {
        duplicates++;
      }
    }
    if (duplicates != 0 || out_of_range != 0) {
      std::cerr << "Data validation error: " << duplicates << " duplicated and " << out_of_range
                << " out-of-range fetch_add returns out of " << expected << std::endl;
      exit(-1);
    }
  }
}
