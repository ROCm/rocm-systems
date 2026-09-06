/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipPerfDynDataPrefetch hipPerfDynDataPrefetch
 * @{
 * @ingroup PerformanceTestPrefetch
 * Test Description
 * ----------------
 * Measures the performance benefit of CP (Command Processor) dynamic data prefetch
 * for kernel pipeline workloads. Tests both sequential and random tile access patterns.
 *
 * Key scenarios:
 * 1. Sequential tile pipeline - predictable memory pattern (streaming data)
 * 2. Random tile access - unpredictable memory pattern (batch processing)
 *
 * Expected results:
 * - Sequential: ~3-4x speedup with prefetch
 * - Random: ~2-4x speedup with prefetch (depending on matrix size vs L2)
 *
 * Test source
 * -----------
 * performance/scenarios/prefetch/hipPerfDynDataPrefetch.cc
 *
 * Test requirements
 * -----------------
 * - Device with CP prefetch support (gfx1250+)
 * - HIP_VERSION >= 6.3
 */

#include <hip_test_common.hh>
#include <hip/hip_ext.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>

// Kernel that processes one tile with simple computation
__global__ void compute_tile_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int tile_size,
                                   int tile_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < tile_size) {
    float val = input[tile_offset + tid];
    // Light compute to make memory access visible
    #pragma unroll
    for (int i = 0; i < 100; i++) {
      val = val * 1.001f + 0.001f;
    }
    output[tile_offset + tid] = val;
  }
}

// Helper: Check if prefetch is available
static bool isPrefetchAvailable() {
  int maxRegions = 0;
  HIP_CHECK(hipDeviceGetAttribute(&maxRegions,
                                  hipDeviceAttributeMaxDynDataPrefetchRegions, 0));
  return (maxRegions >= 1);
}

// Helper: Setup prefetch configuration for one tile
static void setupPrefetchConfig(hipExtDynDataPrefetchConfig& config,
                               void* address,
                               size_t tile_bytes) {
  config.numRegions = 1;
  config.temporal = hipExtDynDataPrefetchTemporalRegular;
  config.regions[0].address = address;
  config.regions[0].stride = 4096;
  config.regions[0].width = 4096;
  config.regions[0].height = std::min<uint32_t>(tile_bytes / 4096, 65535);
}

// Helper: Run kernel pipeline with optional prefetch
static double runPipeline(float* d_input, float* d_output,
                         int tile_size, int num_kernels,
                         const std::vector<int>& tile_offsets,
                         bool enablePrefetch,
                         hipExtDynDataPrefetchConfig& prefetchConfig,
                         hipLaunchAttribute& prefetchAttr) {
  const int threads = 256;
  const int blocks = (tile_size + threads - 1) / threads;

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));

  for (int k = 0; k < num_kernels; k++) {
    int offset = tile_offsets[k];

    if (enablePrefetch) {
      prefetchConfig.regions[0].address = d_input + offset;
    }

    void* args[] = {&d_input, &d_output, &tile_size, &offset};

    hipLaunchConfig_t config = {};
    config.gridDim = dim3(blocks, 1, 1);
    config.blockDim = dim3(threads, 1, 1);
    config.stream = nullptr;
    config.attrs = enablePrefetch ? &prefetchAttr : nullptr;
    config.numAttrs = enablePrefetch ? 1 : 0;

    HIP_CHECK(hipLaunchKernelExC(&config, (const void*)compute_tile_kernel, args));
  }

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return ms;
}

/**
 * Test Case: Sequential Tile Pipeline
 *
 * Pattern: K0 → K1 → K2 ... (each kernel processes next sequential tile)
 * Memory: Sequential regions (tile[0], tile[1], tile[2], ...)
 * Prefetch: Tile K+1 is prefetched while kernel K runs (shadow prefetch)
 */
TEST_CASE("Performance_hipPerfPrefetch_SequentialPipeline") {
  // Sleep to let cache settle before test
  std::this_thread::sleep_for(std::chrono::seconds(2));

  if (!isPrefetchAvailable()) {
    WARN("CP prefetch not available on this device - skipping test");
    return;
  }

  constexpr int NUM_KERNELS = 20;
  constexpr int TILE_SIZE_KB = 256;
  constexpr int TILE_SIZE = (TILE_SIZE_KB * 1024) / sizeof(float);
  constexpr int TOTAL_SIZE = TILE_SIZE * NUM_KERNELS;

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  CONSOLE_PRINT("=== Sequential Tile Pipeline Prefetch Test ===");
  CONSOLE_PRINT("Device: %s (%s)", props.name, props.gcnArchName);
  CONSOLE_PRINT("Kernels: %d, Tile: %d KB, Total: %.1f MB",
                NUM_KERNELS, TILE_SIZE_KB, TOTAL_SIZE * sizeof(float) / 1e6);

  // Allocate memory
  float *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, TOTAL_SIZE * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, TOTAL_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_input, 0x3F, TOTAL_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_output, 0, TOTAL_SIZE * sizeof(float)));

  // Setup prefetch
  hipExtDynDataPrefetchConfig prefetchConfig = {};
  setupPrefetchConfig(prefetchConfig, d_input, TILE_SIZE * sizeof(float));

  hipLaunchAttribute prefetchAttr = {};
  prefetchAttr.id = hipLaunchAttributeExtDynDataPrefetch;
  prefetchAttr.val.dynDataPrefetch = &prefetchConfig;

  // Sequential tile offsets
  std::vector<int> offsets(NUM_KERNELS);
  for (int i = 0; i < NUM_KERNELS; i++) {
    offsets[i] = i * TILE_SIZE;
  }

  // Test without prefetch
  double ms_noprefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                     offsets, false, prefetchConfig, prefetchAttr);

  HIP_CHECK(hipMemset(d_output, 0, TOTAL_SIZE * sizeof(float)));

  // Test with prefetch
  double ms_prefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                   offsets, true, prefetchConfig, prefetchAttr);

  double speedup = ms_noprefetch / ms_prefetch;

  CONSOLE_PRINT("Results:");
  CONSOLE_PRINT("  No prefetch:   %.3f ms (%.3f ms/kernel)",
                ms_noprefetch, ms_noprefetch / NUM_KERNELS);
  CONSOLE_PRINT("  With prefetch: %.3f ms (%.3f ms/kernel)",
                ms_prefetch, ms_prefetch / NUM_KERNELS);
  CONSOLE_PRINT("  Speedup: %.2fx", speedup);

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  // Validate: expect significant speedup for sequential pattern
  REQUIRE(speedup > 1.5);  // At least 1.5x improvement expected
}

/**
 * Test Case: Random Tile Access
 *
 * Pattern: Random tile selection from single huge matrix
 * Memory: Scattered access (tile[847], tile[1203], tile[42], ...)
 * Prefetch: Must know which random tile to prefetch (address-specific)
 */
TEST_CASE("Performance_hipPerfPrefetch_RandomTile") {
  // Longer sleep to clear cache from previous test
  std::this_thread::sleep_for(std::chrono::seconds(5));

  if (!isPrefetchAvailable()) {
    WARN("CP prefetch not available on this device - skipping test");
    return;
  }

  constexpr int NUM_KERNELS = 20;
  constexpr int TILE_SIZE_KB = 256;
  constexpr int MATRIX_MB = 64;  // Use 64MB for faster test (512MB too large)
  constexpr int TILE_SIZE = (TILE_SIZE_KB * 1024) / sizeof(float);
  constexpr int MATRIX_SIZE = (MATRIX_MB * 1024 * 1024) / sizeof(float);
  constexpr int NUM_TILES = MATRIX_SIZE / TILE_SIZE;

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  CONSOLE_PRINT("\n=== Random Tile Access Prefetch Test ===");
  CONSOLE_PRINT("Device: %s (%s)", props.name, props.gcnArchName);
  CONSOLE_PRINT("Kernels: %d, Tile: %d KB, Matrix: %d MB (%d tiles)",
                NUM_KERNELS, TILE_SIZE_KB, MATRIX_MB, NUM_TILES);
  CONSOLE_PRINT("L2 Cache: %.1f MB", props.l2CacheSize / 1e6);

  REQUIRE(NUM_KERNELS <= NUM_TILES);

  // Allocate single huge matrix
  float *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, MATRIX_SIZE * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, MATRIX_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_input, 0x3F, MATRIX_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_output, 0, MATRIX_SIZE * sizeof(float)));

  // Generate random tile indices (shuffled)
  std::vector<int> tile_indices(NUM_TILES);
  for (int i = 0; i < NUM_TILES; i++) {
    tile_indices[i] = i;
  }
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::shuffle(tile_indices.begin(), tile_indices.end(), rng);

  // Convert to offsets
  std::vector<int> offsets(NUM_KERNELS);
  for (int i = 0; i < NUM_KERNELS; i++) {
    offsets[i] = tile_indices[i] * TILE_SIZE;
  }

  // Setup prefetch
  hipExtDynDataPrefetchConfig prefetchConfig = {};
  setupPrefetchConfig(prefetchConfig, d_input, TILE_SIZE * sizeof(float));

  hipLaunchAttribute prefetchAttr = {};
  prefetchAttr.id = hipLaunchAttributeExtDynDataPrefetch;
  prefetchAttr.val.dynDataPrefetch = &prefetchConfig;

  // Test without prefetch
  double ms_noprefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                     offsets, false, prefetchConfig, prefetchAttr);

  HIP_CHECK(hipMemset(d_output, 0, MATRIX_SIZE * sizeof(float)));

  // Test with prefetch
  double ms_prefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                   offsets, true, prefetchConfig, prefetchAttr);

  double speedup = ms_noprefetch / ms_prefetch;

  CONSOLE_PRINT("Results:");
  CONSOLE_PRINT("  No prefetch:   %.3f ms (%.3f ms/kernel)",
                ms_noprefetch, ms_noprefetch / NUM_KERNELS);
  CONSOLE_PRINT("  With prefetch: %.3f ms (%.3f ms/kernel)",
                ms_prefetch, ms_prefetch / NUM_KERNELS);
  CONSOLE_PRINT("  Speedup: %.2fx", speedup);

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  // Validate: expect speedup for random tiles from large matrix
  REQUIRE(speedup > 1.2);  // At least 1.2x improvement expected
}

/**
 * @}
 */
