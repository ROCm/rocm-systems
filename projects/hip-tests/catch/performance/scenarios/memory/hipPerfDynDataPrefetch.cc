/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipDynDataPrefetch hipDynDataPrefetch
 * @{
 * @ingroup PerformanceTestMemory
 * Performance test for dynamic data prefetch feature with hipLaunchAttributeExtDynDataPrefetch.
 * Tests GEMM kernel performance with and without prefetch hints to measure memory optimization impact.
 *
 * CONFIGURATION:
 * To configure this test, create a file named "dyndata_prefetch_config.txt" in the same directory
 * as the test executable with the following format:
 *   Line 1: Path to code object file (.co)
 *   Line 2: Kernel name
 *
 * Example dyndata_prefetch_config.txt:
 *   TensileLibrary_gfx1250.co
 *   Cijk_Alik_Bljk_F8F4F8S_MXAE8B32_MXBE8B32_BH_UserArgs_MT64x512x256_...
 *
 * If the config file is not found, the test will use default values (may not work on all platforms).
 */

#include <hip_test_common.hh>
#include <hip/hip_ext.h>
#include <fstream>
#include <string>
#include <chrono>

#define NUM_WARMUP 5
#define NUM_ITER 20

// Problem constants
static const int M     = 64;
static const int N     = 5120;
static const int BATCH = 128;
static const int K     = 1024;
static const int MX_BLOCK = 32;   // OCP MX block size along K

// Append value to byte buffer (no alignment padding)
template <typename T>
static void buf_append(uint8_t* buf, size_t& off, T val) {
    memcpy(buf + off, &val, sizeof(T));
    off += sizeof(T);
}

/**
 * Test class for dynamic data prefetch performance testing
 */
class hipPerfDynDataPrefetch {
 private:
  std::string co_path_;
  std::string kernel_name_;
  hipModule_t mod_;
  hipFunction_t func_;
  int numCUs_;

  void* d_A_;
  void* d_MXSA_;
  void* d_B_;
  void* d_MXSB_;
  void* d_C_;
  void* d_D_;

  size_t sizeA_;
  size_t sizeMXSA_;
  size_t sizeB_;
  size_t sizeMXSB_;
  size_t sizeCD_;

  uint8_t kargs_[256];
  size_t kargs_size_;

  void loadConfiguration();
  void loadKernel();
  void allocateBuffers();
  void buildKernelArgs();
  void cleanup();
  double runBenchmark(bool enablePrefetch, int warmup, int iters);

 public:
  hipPerfDynDataPrefetch();
  ~hipPerfDynDataPrefetch();
  void run();
};

hipPerfDynDataPrefetch::hipPerfDynDataPrefetch()
    : mod_(nullptr), func_(nullptr), numCUs_(0),
      d_A_(nullptr), d_MXSA_(nullptr), d_B_(nullptr),
      d_MXSB_(nullptr), d_C_(nullptr), d_D_(nullptr) {
}

hipPerfDynDataPrefetch::~hipPerfDynDataPrefetch() {
  cleanup();
}

void hipPerfDynDataPrefetch::loadConfiguration() {
  // Read configuration from dyndata_prefetch_config.txt
  // Format:
  //   Line 1: Code object file path (.co)
  //   Line 2: Kernel name
  std::string config_path = "dyndata_prefetch_config.txt";
  std::ifstream config_file(config_path);

  if (!config_file.is_open()) {
    FAIL("Configuration file not found: " << config_path << "\n"
         << "Create dyndata_prefetch_config.txt with:\n"
         << "  Line 1: Code object file path (.co)\n"
         << "  Line 2: Kernel name");
  }

  std::getline(config_file, co_path_);
  std::getline(config_file, kernel_name_);
  config_file.close();

  INFO("Loaded configuration from " << config_path);
  INFO("Code object: " << co_path_);
  INFO("Kernel name: " << kernel_name_.substr(0, 60) << "...");
}

void hipPerfDynDataPrefetch::loadKernel() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  numCUs_ = props.multiProcessorCount;
  INFO("Device: " << props.name << " arch: " << props.gcnArchName << " CUs: " << numCUs_);

  HIP_CHECK(hipModuleLoad(&mod_, co_path_.c_str()));
  HIP_CHECK(hipModuleGetFunction(&func_, mod_, kernel_name_.c_str()));
}

void hipPerfDynDataPrefetch::allocateBuffers() {
  // Calculate sizes
  sizeA_    = (size_t)M * K * BATCH;                 // FP8: 1 byte/elem
  sizeMXSA_ = (size_t)M * (K / MX_BLOCK) * BATCH;   // E8M0: 1 byte/elem
  sizeB_    = (size_t)K * N * BATCH / 2;             // FP4: packed 2/byte
  sizeMXSB_ = (size_t)N * (K / MX_BLOCK) * BATCH;   // E8M0: 1 byte/elem
  sizeCD_   = (size_t)M * N * BATCH;                 // FP8: 1 byte/elem

  // Allocate device buffers
  HIP_CHECK(hipMalloc(&d_A_,    sizeA_));
  HIP_CHECK(hipMalloc(&d_MXSA_, sizeMXSA_));
  HIP_CHECK(hipMalloc(&d_B_,    sizeB_));
  HIP_CHECK(hipMalloc(&d_MXSB_, sizeMXSB_));
  HIP_CHECK(hipMalloc(&d_C_,    sizeCD_));
  HIP_CHECK(hipMalloc(&d_D_,    sizeCD_));

  // Initialize with test pattern (FP8 E4M3: 0x3C ≈ 1.0; E8M0: 0x3F = 2^0)
  HIP_CHECK(hipMemset(d_A_,    0x3C, sizeA_));
  HIP_CHECK(hipMemset(d_MXSA_, 0x3F, sizeMXSA_));
  HIP_CHECK(hipMemset(d_B_,    0x77, sizeB_));
  HIP_CHECK(hipMemset(d_MXSB_, 0x3F, sizeMXSB_));
  HIP_CHECK(hipMemset(d_C_,    0x00, sizeCD_));
  HIP_CHECK(hipMemset(d_D_,    0x00, sizeCD_));
}

void hipPerfDynDataPrefetch::buildKernelArgs() {
  // Build kernel arguments buffer matching Tensile's layout
  uint32_t wgm    = 1;
  uint32_t wgmxcc = 1;
  int32_t  wgmxccg = numCUs_;
  int32_t  internalArgs1 = (int32_t)(((uint32_t)wgmxccg << 22) | (wgmxcc << 16) | (wgm & 0xFFFF));

  size_t koff = 0;

  // kernelArgs block
  buf_append<uint32_t>(kargs_, koff, 1u);                // gemm_count
  buf_append<uint32_t>(kargs_, koff, 1u);                // internalArgs
  buf_append<int32_t> (kargs_, koff, internalArgs1);     // internalArgs1
  buf_append<uint32_t>(kargs_, koff, 1u);                // numWorkGroups

  // singleCallArgs block
  buf_append<uint32_t>(kargs_, koff, (uint32_t)M);
  buf_append<uint32_t>(kargs_, koff, (uint32_t)N);
  buf_append<uint32_t>(kargs_, koff, (uint32_t)BATCH);
  buf_append<uint32_t>(kargs_, koff, (uint32_t)K);

  buf_append<void*>(kargs_, koff, d_A_);
  buf_append<void*>(kargs_, koff, d_MXSA_);
  buf_append<void*>(kargs_, koff, d_B_);
  buf_append<void*>(kargs_, koff, d_MXSB_);

  buf_append<uint32_t>(kargs_, koff, (uint32_t)K);
  buf_append<uint32_t>(kargs_, koff, (uint32_t)(M * K));
  buf_append<uint32_t>(kargs_, koff, (uint32_t)(K / MX_BLOCK));
  buf_append<uint32_t>(kargs_, koff, (uint32_t)(M * (K/MX_BLOCK)));
  buf_append<uint32_t>(kargs_, koff, (uint32_t)K);
  buf_append<uint32_t>(kargs_, koff, (uint32_t)((size_t)K * N));
  buf_append<uint32_t>(kargs_, koff, (uint32_t)(K / MX_BLOCK));
  buf_append<uint32_t>(kargs_, koff, (uint32_t)(N * (K/MX_BLOCK)));

  buf_append<float>(kargs_, koff, 1.0f);    // alpha
  buf_append<float>(kargs_, koff, 0.0f);    // beta

  buf_append<void*>(kargs_, koff, d_D_);
  buf_append<void*>(kargs_, koff, d_C_);

  buf_append<uint32_t>(kargs_, koff, (uint32_t)M);
  buf_append<uint32_t>(kargs_, koff, (uint32_t)((size_t)M * N));
  buf_append<uint32_t>(kargs_, koff, (uint32_t)M);
  buf_append<uint32_t>(kargs_, koff, (uint32_t)((size_t)M * N));

  kargs_size_ = koff;
  REQUIRE(kargs_size_ == 136);
}

double hipPerfDynDataPrefetch::runBenchmark(bool enablePrefetch, int warmup, int iters) {
  // Grid/block dimensions
  const unsigned int gX = 128, gY = 2, gZ = 128;
  const unsigned int bX = 128, bY = 1, bZ = 1;

  // Setup prefetch configuration
  hipExtDynDataPrefetchConfig prefetchConfig = {};
  hipLaunchAttribute prefetchAttr = {};

  if (enablePrefetch) {
    int maxPrefetchRegions = 0;
    HIP_CHECK(hipDeviceGetAttribute(&maxPrefetchRegions,
                                    hipDeviceAttributeMaxDynDataPrefetchRegions, 0));

    if (maxPrefetchRegions < 2) {
      WARN("Device reports insufficient prefetch regions (" << maxPrefetchRegions << "), skipping prefetch test");
      return -1.0;
    }

    prefetchConfig.numRegions = 2;
    prefetchConfig.temporal = hipExtDynDataPrefetchTemporalRegular;

    // Region 0: A (FP8)
    prefetchConfig.regions[0].address = d_A_;
    prefetchConfig.regions[0].stride  = 1024;
    prefetchConfig.regions[0].width   = 256;
    prefetchConfig.regions[0].height  = 64 * BATCH;

    // Region 1: B (FP4 packed)
    prefetchConfig.regions[1].address = d_B_;
    prefetchConfig.regions[1].stride  = K / 2;
    prefetchConfig.regions[1].width   = 256;
    prefetchConfig.regions[1].height  = 512 * BATCH;

    prefetchAttr.id = hipLaunchAttributeExtDynDataPrefetch;
    prefetchAttr.val.dynDataPrefetch = &prefetchConfig;
  }

  // HIP launch params
  void* hip_params[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs_,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kargs_size_,
      HIP_LAUNCH_PARAM_END
  };

  auto do_launch = [&]() {
    HIP_LAUNCH_CONFIG config = {};
    config.gridDimX = gX;
    config.gridDimY = gY;
    config.gridDimZ = gZ;
    config.blockDimX = bX;
    config.blockDimY = bY;
    config.blockDimZ = bZ;
    config.sharedMemBytes = 0;
    config.hStream = nullptr;
    config.attrs = enablePrefetch ? &prefetchAttr : nullptr;
    config.numAttrs = enablePrefetch ? 1 : 0;
    HIP_CHECK(hipDrvLaunchKernelEx(&config, func_, nullptr, (void**)hip_params));
  };

  // Warmup
  for (int i = 0; i < warmup; i++) {
    do_launch();
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Timed run using hipEvent
  hipEvent_t ev_start, ev_stop;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  HIP_CHECK(hipEventRecord(ev_start, nullptr));
  for (int i = 0; i < iters; i++) {
    do_launch();
  }
  HIP_CHECK(hipEventRecord(ev_stop, nullptr));
  HIP_CHECK(hipEventSynchronize(ev_stop));

  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  double avg_ms = elapsed_ms / iters;

  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipEventDestroy(ev_stop));

  return avg_ms;
}

void hipPerfDynDataPrefetch::cleanup() {
  if (d_A_)    HIP_CHECK(hipFree(d_A_));
  if (d_MXSA_) HIP_CHECK(hipFree(d_MXSA_));
  if (d_B_)    HIP_CHECK(hipFree(d_B_));
  if (d_MXSB_) HIP_CHECK(hipFree(d_MXSB_));
  if (d_C_)    HIP_CHECK(hipFree(d_C_));
  if (d_D_)    HIP_CHECK(hipFree(d_D_));
  if (mod_)    HIP_CHECK(hipModuleUnload(mod_));
}

void hipPerfDynDataPrefetch::run() {
  loadConfiguration();
  loadKernel();
  allocateBuffers();
  buildKernelArgs();

  // Run baseline (no prefetch)
  double baseline_ms = runBenchmark(false, NUM_WARMUP, NUM_ITER);
  double baseline_tflops = 2.0 * (double)M * N * K * BATCH / (baseline_ms * 1e-3 * 1e12);

  CONSOLE_PRINT("hipPerfDynDataPrefetch[baseline] avg %.3f ms, %.2f TFlops\n",
                baseline_ms, baseline_tflops);

  // Run with prefetch
  double prefetch_ms = runBenchmark(true, NUM_WARMUP, NUM_ITER);

  if (prefetch_ms > 0) {
    double prefetch_tflops = 2.0 * (double)M * N * K * BATCH / (prefetch_ms * 1e-3 * 1e12);
    double speedup = baseline_ms / prefetch_ms;

    CONSOLE_PRINT("hipPerfDynDataPrefetch[prefetch] avg %.3f ms, %.2f TFlops, speedup %.2fx\n",
                  prefetch_ms, prefetch_tflops, speedup);
  }
}

/**
 * Test case: Dynamic data prefetch performance
 * Loads kernel from code object file and measures performance impact of prefetch hints
 */
TEST_CASE("Perf_hipDynDataPrefetch_GEMM") {
  hipPerfDynDataPrefetch test;
  test.run();
}

/**
* End doxygen group PerformanceTestMemory.
* @}
*/
