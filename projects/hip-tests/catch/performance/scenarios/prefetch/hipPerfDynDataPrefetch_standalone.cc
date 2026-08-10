/*
 * Standalone version (no Catch2 dependency) for testing
 * Compile: hipcc -O3 -std=c++17 hipPerfDynDataPrefetch_standalone.cc -o test
 */

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <unistd.h>  // For sleep()

#define HIP_CHECK(cmd) \
    do { \
        hipError_t _e = (cmd); \
        if (_e != hipSuccess) { \
            fprintf(stderr, "HIP error %s at %s:%d\n", \
                    hipGetErrorName(_e), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

__global__ void compute_tile_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int tile_size,
                                   int tile_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < tile_size) {
    float val = input[tile_offset + tid];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
      val = val * 1.001f + 0.001f;
    }
    output[tile_offset + tid] = val;
  }
}

static bool isPrefetchAvailable() {
  int maxRegions = 0;
  HIP_CHECK(hipDeviceGetAttribute(&maxRegions,
                                  hipDeviceAttributeMaxDynDataPrefetchRegions, 0));
  return (maxRegions >= 1);
}

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

void testSequentialPipeline() {
  printf("\n=== Sequential Tile Pipeline Prefetch Test ===\n");
  printf("Waiting 2s for cache to settle...\n");
  sleep(2);

  constexpr int NUM_KERNELS = 20;
  constexpr int TILE_SIZE_KB = 256;
  constexpr int TILE_SIZE = (TILE_SIZE_KB * 1024) / sizeof(float);
  constexpr int TOTAL_SIZE = TILE_SIZE * NUM_KERNELS;

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  printf("Device: %s (%s)\n", props.name, props.gcnArchName);
  printf("Kernels: %d, Tile: %d KB, Total: %.1f MB\n",
         NUM_KERNELS, TILE_SIZE_KB, TOTAL_SIZE * sizeof(float) / 1e6);

  float *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, TOTAL_SIZE * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, TOTAL_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_input, 0x3F, TOTAL_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_output, 0, TOTAL_SIZE * sizeof(float)));

  hipExtDynDataPrefetchConfig prefetchConfig = {};
  setupPrefetchConfig(prefetchConfig, d_input, TILE_SIZE * sizeof(float));

  hipLaunchAttribute prefetchAttr = {};
  prefetchAttr.id = hipLaunchAttributeExtDynDataPrefetch;
  prefetchAttr.val.dynDataPrefetch = &prefetchConfig;

  std::vector<int> offsets(NUM_KERNELS);
  for (int i = 0; i < NUM_KERNELS; i++) {
    offsets[i] = i * TILE_SIZE;
  }

  double ms_noprefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                     offsets, false, prefetchConfig, prefetchAttr);

  HIP_CHECK(hipMemset(d_output, 0, TOTAL_SIZE * sizeof(float)));

  double ms_prefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                   offsets, true, prefetchConfig, prefetchAttr);

  double speedup = ms_noprefetch / ms_prefetch;

  printf("Results:\n");
  printf("  No prefetch:   %.3f ms (%.3f ms/kernel)\n",
         ms_noprefetch, ms_noprefetch / NUM_KERNELS);
  printf("  With prefetch: %.3f ms (%.3f ms/kernel)\n",
         ms_prefetch, ms_prefetch / NUM_KERNELS);
  printf("  Speedup: %.2fx\n", speedup);

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  if (speedup > 1.5) {
    printf("✓ PASS: Speedup > 1.5x\n");
  } else {
    printf("✗ FAIL: Speedup < 1.5x\n");
  }
}

void testRandomTile() {
  printf("\n=== Random Tile Access Prefetch Test ===\n");
  printf("Waiting 5s for cache to cool from previous test...\n");
  sleep(5);

  constexpr int NUM_KERNELS = 20;
  constexpr int TILE_SIZE_KB = 256;
  constexpr int MATRIX_MB = 64;  // Reduced from 512MB (allocation/initialization overhead)
  constexpr int TILE_SIZE = (TILE_SIZE_KB * 1024) / sizeof(float);
  constexpr int MATRIX_SIZE = (MATRIX_MB * 1024 * 1024) / sizeof(float);
  constexpr int NUM_TILES = MATRIX_SIZE / TILE_SIZE;

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  printf("Device: %s (%s)\n", props.name, props.gcnArchName);
  printf("Kernels: %d, Tile: %d KB, Matrix: %d MB (%d tiles)\n",
         NUM_KERNELS, TILE_SIZE_KB, MATRIX_MB, NUM_TILES);
  printf("L2 Cache: %.1f MB\n", props.l2CacheSize / 1e6);

  float *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, MATRIX_SIZE * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, MATRIX_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_input, 0x3F, MATRIX_SIZE * sizeof(float)));
  HIP_CHECK(hipMemset(d_output, 0, MATRIX_SIZE * sizeof(float)));

  std::vector<int> tile_indices(NUM_TILES);
  for (int i = 0; i < NUM_TILES; i++) {
    tile_indices[i] = i;
  }
  std::mt19937 rng(42);
  std::shuffle(tile_indices.begin(), tile_indices.end(), rng);

  std::vector<int> offsets(NUM_KERNELS);
  for (int i = 0; i < NUM_KERNELS; i++) {
    offsets[i] = tile_indices[i] * TILE_SIZE;
  }

  hipExtDynDataPrefetchConfig prefetchConfig = {};
  setupPrefetchConfig(prefetchConfig, d_input, TILE_SIZE * sizeof(float));

  hipLaunchAttribute prefetchAttr = {};
  prefetchAttr.id = hipLaunchAttributeExtDynDataPrefetch;
  prefetchAttr.val.dynDataPrefetch = &prefetchConfig;

  double ms_noprefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                     offsets, false, prefetchConfig, prefetchAttr);

  HIP_CHECK(hipMemset(d_output, 0, MATRIX_SIZE * sizeof(float)));

  double ms_prefetch = runPipeline(d_input, d_output, TILE_SIZE, NUM_KERNELS,
                                   offsets, true, prefetchConfig, prefetchAttr);

  double speedup = ms_noprefetch / ms_prefetch;

  printf("Results:\n");
  printf("  No prefetch:   %.3f ms (%.3f ms/kernel)\n",
         ms_noprefetch, ms_noprefetch / NUM_KERNELS);
  printf("  With prefetch: %.3f ms (%.3f ms/kernel)\n",
         ms_prefetch, ms_prefetch / NUM_KERNELS);
  printf("  Speedup: %.2fx\n", speedup);

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  if (speedup > 1.2) {
    printf("✓ PASS: Speedup > 1.2x\n");
  } else {
    printf("✗ FAIL: Speedup < 1.2x\n");
  }
}

int main() {
  if (!isPrefetchAvailable()) {
    printf("WARNING: CP prefetch not available - skipping tests\n");
    return 0;
  }

  testSequentialPipeline();
  testRandomTile();

  printf("\n=== All Tests Complete ===\n");
  return 0;
}
