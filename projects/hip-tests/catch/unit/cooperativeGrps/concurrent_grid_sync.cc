/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Coop concurrency tests: grid.sync() must stay correct across streams and when
// chained on one stream. Overlap (DEBUG_CLR_COOP_CONCURRENT) is INFO, not asserted.

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include <chrono>

namespace cg = cooperative_groups;

static constexpr size_t kBufferLen = 1024 * 1024;

// Grid-wide reduction that calls grid.sync() twice per iteration; the iteration
// loop lengthens the kernel so concurrent overlap (when enabled) is measurable.
__global__ void reduce_grid_sync(const int* buf, size_t n, unsigned long long* partial,
                                 unsigned long long* result, int iters) {
  extern __shared__ unsigned long long sm[];

  cg::thread_block tb = cg::this_thread_block();
  cg::grid_group gg = cg::this_grid();

  const auto tid = gg.thread_rank();
  const auto stride = gg.size();
  const auto local_tid = tb.thread_rank();
  const auto block_size = tb.size();
  const auto grid_blocks = gridDim.x;

  for (int it = 0; it < iters; ++it) {
    unsigned long long sum = 0;
    for (size_t i = tid; i < n; i += stride) {
      sum += buf[i];
    }
    sm[local_tid] = sum;
    tb.sync();

    if (local_tid == 0) {
      unsigned long long block_sum = 0;
      for (unsigned int t = 0; t < block_size; ++t) {
        block_sum += sm[t];
      }
      partial[blockIdx.x] = block_sum;
    }
    gg.sync();

    if (tid == 0) {
      unsigned long long total = 0;
      for (unsigned int b = 0; b < grid_blocks; ++b) {
        total += partial[b];
      }
      *result = total;
    }
    gg.sync();
  }
}

namespace {

struct CoopWork {
  int* buf_d = nullptr;
  unsigned long long* partial_d = nullptr;
  unsigned long long* result_h = nullptr;  // host-visible
  hipStream_t stream = nullptr;
  dim3 grid;
  dim3 block;
  size_t shmem = 0;
};

void SetupWork(CoopWork& w, const int* host_buf, size_t buffer_bytes, dim3 grid, dim3 block) {
  w.grid = grid;
  w.block = block;
  w.shmem = block.x * sizeof(unsigned long long);
  HIP_CHECK(hipMalloc(&w.buf_d, buffer_bytes));
  HIP_CHECK(hipMemcpy(w.buf_d, host_buf, buffer_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&w.partial_d, grid.x * sizeof(unsigned long long)));
  HIP_CHECK(hipHostMalloc(&w.result_h, sizeof(unsigned long long)));
  *w.result_h = 0;
  HIP_CHECK(hipStreamCreate(&w.stream));
}

void LaunchWork(CoopWork& w, int iters) {
  void* params[5];
  params[0] = reinterpret_cast<void*>(&w.buf_d);
  params[1] = const_cast<void*>(reinterpret_cast<const void*>(&kBufferLen));
  params[2] = reinterpret_cast<void*>(&w.partial_d);
  params[3] = reinterpret_cast<void*>(&w.result_h);
  params[4] = reinterpret_cast<void*>(&iters);
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(reduce_grid_sync), w.grid, w.block,
                                       params, w.shmem, w.stream));
}

void TeardownWork(CoopWork& w) {
  HIP_CHECK(hipStreamDestroy(w.stream));
  HIP_CHECK(hipHostFree(w.result_h));
  HIP_CHECK(hipFree(w.partial_d));
  HIP_CHECK(hipFree(w.buf_d));
}

}  // namespace

// Two cooperative grids on two streams, launched back to back before either is
// synchronized. Verifies grid.sync() correctness under concurrent submission.
HIP_TEST_CASE(Unit_ConcurrentCooperativeKernel_GridSync_Correctness) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  if (!props.cooperativeLaunch) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    return;
  }

  const size_t buffer_bytes = kBufferLen * sizeof(int);
  std::vector<int> host_buf(kBufferLen);
  for (size_t i = 0; i < kBufferLen; ++i) {
    host_buf[i] = static_cast<int>(i);
  }
  const unsigned long long expected =
      (static_cast<unsigned long long>(kBufferLen) * (kBufferLen - 1)) / 2;

  const uint32_t block_x = GENERATE(64u, 128u, 256u);
  const dim3 block(block_x);

  int blocks_per_cu = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_cu, reduce_grid_sync, block.x, block.x * sizeof(unsigned long long)));
  REQUIRE(blocks_per_cu > 0);

  // Two DIFFERENT grid sizes (full/4, full/2) that together fit in cooperative
  // occupancy, so each grid.sync() must use its own workgroup count.
  const int full_blocks = props.multiProcessorCount * blocks_per_cu;
  const uint32_t grid0_blocks = static_cast<uint32_t>(std::max(1, full_blocks / 4));
  const uint32_t grid1_blocks = std::min(
      static_cast<uint32_t>(full_blocks),
      std::max(grid0_blocks + 1u, static_cast<uint32_t>(full_blocks / 2)));
  const dim3 grid0(grid0_blocks);
  const dim3 grid1(grid1_blocks);
  REQUIRE((grid0.x != grid1.x || full_blocks <= 1));

  const int iters = 32;

  INFO("block=" << block.x << " grid0=" << grid0.x << " grid1=" << grid1.x
                << " (full coop grid=" << full_blocks << ") iters=" << iters);

  CoopWork w0, w1;
  SetupWork(w0, host_buf.data(), buffer_bytes, grid0, block);
  SetupWork(w1, host_buf.data(), buffer_bytes, grid1, block);

  // Launch both before synchronizing so the runtime is free to overlap them.
  const auto t_start = std::chrono::steady_clock::now();
  LaunchWork(w0, iters);
  LaunchWork(w1, iters);
  HIP_CHECK(hipStreamSynchronize(w0.stream));
  HIP_CHECK(hipStreamSynchronize(w1.stream));
  const auto t_concurrent =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();

  CHECK(*w0.result_h == expected);
  CHECK(*w1.result_h == expected);

  // Informational: serialized baseline for the same total work.
  *w0.result_h = 0;
  *w1.result_h = 0;
  const auto s_start = std::chrono::steady_clock::now();
  LaunchWork(w0, iters);
  HIP_CHECK(hipStreamSynchronize(w0.stream));
  LaunchWork(w1, iters);
  HIP_CHECK(hipStreamSynchronize(w1.stream));
  const auto t_serial =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s_start).count();

  CHECK(*w0.result_h == expected);
  CHECK(*w1.result_h == expected);

  INFO("concurrent=" << t_concurrent << "ms serialized=" << t_serial << "ms speedup="
                     << (t_concurrent > 0.0 ? t_serial / t_concurrent : 0.0) << "x");
  WARN("concurrent=" << t_concurrent << "ms serialized=" << t_serial << "ms speedup="
                     << (t_concurrent > 0.0 ? t_serial / t_concurrent : 0.0)
                     << "x (overlap requires DEBUG_CLR_COOP_CONCURRENT=1 on SW grid.sync ASICs)");

  TeardownWork(w0);
  TeardownWork(w1);
}

// Read-modify-write accumulate with a grid.sync() each iteration. Correct only if
// no other grid touches the same elements concurrently.
__global__ void accumulate_grid_sync(unsigned long long* data, size_t n, int iters) {
  cg::grid_group gg = cg::this_grid();
  const auto tid = gg.thread_rank();
  const auto stride = gg.size();
  for (int it = 0; it < iters; ++it) {
    for (size_t i = tid; i < n; i += stride) {
      data[i] = data[i] + 1ull;  // non-atomic on purpose: races if grids overlap
    }
    gg.sync();
  }
}

// Same-stream chain of coop dispatches must stay ordered (each sees prior writes).
// Relaxed ordering would race the RMW and drop updates, failing the exact-sum check.
HIP_TEST_CASE(Unit_ConcurrentCooperativeKernel_GridSync_SameStreamOrdering) {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  if (!props.cooperativeLaunch) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
    return;
  }

  const size_t n = 256 * 1024;
  const dim3 block(128);
  int blocks_per_cu = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, accumulate_grid_sync,
                                                         block.x, 0));
  REQUIRE(blocks_per_cu > 0);
  const int full_blocks = props.multiProcessorCount * blocks_per_cu;
  const dim3 grid(static_cast<uint32_t>(std::max(1, full_blocks / 2)));

  unsigned long long* data_d = nullptr;
  HIP_CHECK(hipMalloc(&data_d, n * sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(data_d, 0, n * sizeof(unsigned long long)));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  const int kLaunches = 8;
  const int iters = 8;
  for (int l = 0; l < kLaunches; ++l) {
    void* params[3] = {&data_d, const_cast<size_t*>(&n), const_cast<int*>(&iters)};
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(accumulate_grid_sync), grid, block,
                                         params, 0, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<unsigned long long> data_h(n);
  HIP_CHECK(hipMemcpy(data_h.data(), data_d, n * sizeof(unsigned long long), hipMemcpyDeviceToHost));

  const unsigned long long expected =
      static_cast<unsigned long long>(kLaunches) * static_cast<unsigned long long>(iters);
  size_t mismatches = 0;
  for (size_t i = 0; i < n; ++i) {
    if (data_h[i] != expected) ++mismatches;
  }
  INFO("grid=" << grid.x << " launches=" << kLaunches << " iters=" << iters
               << " expected=" << expected << " mismatches=" << mismatches);
  CHECK(mismatches == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(data_d));
}
