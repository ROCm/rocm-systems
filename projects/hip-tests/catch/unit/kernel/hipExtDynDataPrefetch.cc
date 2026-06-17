/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

// Simple DAXPY kernel: out[i] = A[i] + alpha * B[i]
__global__ void daxpyKernel(const float* A, const float* B, float* out,
                            float alpha, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    out[tid] = A[tid] + alpha * B[tid];
  }
}

static hipLaunchAttribute makeDynDataPrefetchAttr(
    const hipExtDynDataPrefetchConfig* prefetchConfig) {
  hipLaunchAttribute attr;
  memset(&attr, 0, sizeof(attr));
  attr.id = hipLaunchAttributeExtDynDataPrefetch;
  attr.val.dynDataPrefetch = prefetchConfig;
  return attr;
}

/**
 * @addtogroup hipExtDynDataPrefetch hipExtDynDataPrefetch
 * @{
 * @ingroup KernelTest
 * Test the hipLaunchAttributeExtDynDataPrefetch launch attribute.
 */

/**
 * Test Description
 * ------------------------
 * Negative tests for hipLaunchAttributeExtDynDataPrefetch validation.
 *
 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipExtDynDataPrefetch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipExtDynDataPrefetch_NegativeTests) {
  constexpr int N = 1024;
  float* d_A = nullptr;
  HIP_CHECK(hipMalloc(&d_A, N * sizeof(float)));

  int maxRegions = 0;
  HIP_CHECK(hipDeviceGetAttribute(&maxRegions, hipDeviceAttributeMaxDynDataPrefetchRegions, 0));

  auto makeConfig = [](hipLaunchAttribute* attr, int numAttrs) {
    hipLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3{4, 1, 1};
    cfg.blockDim = dim3{256, 1, 1};
    cfg.dynamicSmemBytes = 0;
    cfg.stream   = 0;
    cfg.attrs    = attr;
    cfg.numAttrs = numAttrs;
    return cfg;
  };

  // On devices without prefetch support (maxRegions == 0), all attempts to use
  // the attribute return hipErrorNotSupported regardless of other parameters.
  hipError_t unsupportedOrInvalid =
      (maxRegions == 0) ? hipErrorNotSupported : hipErrorInvalidValue;

  SECTION("device query returns non-negative max regions") {
    REQUIRE(maxRegions >= 0);
  }

  SECTION("numRegions == 0 returns error") {
    hipExtDynDataPrefetchRegion region = {};
    region.address = d_A;
    region.width   = 256;
    region.height  = 1;
    region.stride  = 256;

    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = 0;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    prefetchConfig.regions[0] = region;
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  SECTION("numRegions > device max returns error") {
    hipExtDynDataPrefetchRegion region = {};
    region.address = d_A;
    region.width   = 256;
    region.height  = 1;
    region.stride  = 256;

    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = static_cast<unsigned int>(maxRegions) + 1;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    prefetchConfig.regions[0] = region;
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  SECTION("null address in first region returns error") {
    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = 1;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    // regions[0].address is 0 (null) from memset
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  SECTION("null address in region returns error") {
    hipExtDynDataPrefetchRegion region = {};
    region.address = nullptr;
    region.width   = 256;
    region.height  = 1;
    region.stride  = 256;

    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = 1;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    prefetchConfig.regions[0] = region;
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  SECTION("unaligned address returns error") {
    hipExtDynDataPrefetchRegion region = {};
    region.address = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(d_A) + 1);
    region.width   = 256;
    region.height  = 1;
    region.stride  = 256;

    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = 1;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    prefetchConfig.regions[0] = region;
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  SECTION("width == 0 returns error") {
    hipExtDynDataPrefetchRegion region = {};
    region.address = d_A;
    region.width   = 0;
    region.height  = 1;
    region.stride  = 256;

    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = 1;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    prefetchConfig.regions[0] = region;
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  SECTION("width not a multiple of 256 returns error") {
    hipExtDynDataPrefetchRegion region = {};
    region.address = d_A;
    region.width   = 300;
    region.height  = 1;
    region.stride  = 300;

    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = 1;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    prefetchConfig.regions[0] = region;
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  SECTION("height == 0 returns error") {
    hipExtDynDataPrefetchRegion region = {};
    region.address = d_A;
    region.width   = 256;
    region.height  = 0;
    region.stride  = 256;

    hipExtDynDataPrefetchConfig prefetchConfig = {};
    prefetchConfig.numRegions = 1;
    prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
    prefetchConfig.regions[0] = region;
    hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

    auto cfg = makeConfig(&attr, 1);
    void* args[] = {&d_A, &d_A, &d_A, nullptr, nullptr};
    HIP_CHECK_ERROR(hipLaunchKernelExC(&cfg, reinterpret_cast<void*>(daxpyKernel), args),
                    unsupportedOrInvalid);
  }

  HIP_CHECK(hipFree(d_A));
}

/**
 * Test Description
 * ------------------------
 * Functional test: launch a DAXPY kernel with L2 prefetch of both input
 * buffers via hipLaunchAttributeExtDynDataPrefetch and verify correctness.
 * Devices without dynamic data prefetch support skip this test.
 *
 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipExtDynDataPrefetch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipExtDynDataPrefetch_FunctionalDaxpy) {
  int maxRegions = 0;
  HIP_CHECK(hipDeviceGetAttribute(&maxRegions, hipDeviceAttributeMaxDynDataPrefetchRegions, 0));
  if (maxRegions < 2) {
    HIP_SKIP_TEST("Device does not support >= 2 prefetch regions");
    return;
  }

  constexpr int N = 4096;
  constexpr float alpha = 2.0f;

  size_t bytes = N * sizeof(float);
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  HIP_CHECK(hipMalloc(&d_A, bytes));
  HIP_CHECK(hipMalloc(&d_B, bytes));
  HIP_CHECK(hipMalloc(&d_C, bytes));

  std::vector<float> h_A(N), h_B(N), h_C(N);
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 0.5f);
  }
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

  size_t width = (bytes / 256) * 256;

  hipExtDynDataPrefetchRegion regions[2] = {};
  regions[0].address = d_A;
  regions[0].width   = width;
  regions[0].height  = 1;
  regions[0].stride  = width;

  regions[1].address = d_B;
  regions[1].width   = width;
  regions[1].height  = 1;
  regions[1].stride  = width;

  hipExtDynDataPrefetchConfig prefetchConfig = {};
  prefetchConfig.numRegions = 2;
  prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
  prefetchConfig.regions[0] = regions[0];
  prefetchConfig.regions[1] = regions[1];
  hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

  hipLaunchConfig_t config = {};
  constexpr int blockSize = 256;
  config.gridDim  = dim3{(N + blockSize - 1) / blockSize, 1, 1};
  config.blockDim = dim3{blockSize, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream   = 0;
  config.attrs    = &attr;
  config.numAttrs = 1;

  int n = N;
  void* args[] = {&d_A, &d_B, &d_C, const_cast<float*>(&alpha), &n};
  HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(daxpyKernel), args));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    float expected = h_A[i] + alpha * h_B[i];
    REQUIRE(h_C[i] == Catch::Approx(expected).epsilon(1e-5f));
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
}

/**
 * Test Description
 * ------------------------
 * Functional test: launch a kernel with a single prefetch region.
 *
 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipExtDynDataPrefetch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipExtDynDataPrefetch_SingleRegion) {
  int maxRegions = 0;
  HIP_CHECK(hipDeviceGetAttribute(&maxRegions, hipDeviceAttributeMaxDynDataPrefetchRegions, 0));
  if (maxRegions < 1) {
    HIP_SKIP_TEST("Device does not support prefetch regions");
    return;
  }

  constexpr int N = 1024;
  constexpr float alpha = 1.0f;
  size_t bytes = N * sizeof(float);

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  HIP_CHECK(hipMalloc(&d_A, bytes));
  HIP_CHECK(hipMalloc(&d_B, bytes));
  HIP_CHECK(hipMalloc(&d_C, bytes));

  std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N);
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

  size_t width = (bytes / 256) * 256;

  hipExtDynDataPrefetchRegion region = {};
  region.address  = d_A;
  region.width    = width;
  region.height   = 1;
  region.stride   = width;

  hipExtDynDataPrefetchConfig prefetchConfig = {};
  prefetchConfig.numRegions = 1;
  prefetchConfig.temporal   = hipExtDynDataPrefetchTemporalRegular;
  prefetchConfig.regions[0] = region;
  hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

  hipLaunchConfig_t config = {};
  constexpr int blockSize = 256;
  config.gridDim  = dim3{(N + blockSize - 1) / blockSize, 1, 1};
  config.blockDim = dim3{blockSize, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream   = 0;
  config.attrs    = &attr;
  config.numAttrs = 1;

  int n = N;
  void* args[] = {&d_A, &d_B, &d_C, const_cast<float*>(&alpha), &n};
  HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(daxpyKernel), args));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    float expected = 1.0f + alpha * 2.0f;
    REQUIRE(h_C[i] == Catch::Approx(expected).epsilon(1e-5f));
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
}

/**
 * Test Description
 * ------------------------
 * Test all temporal hint combinations via hipLaunchAttributeExtDynDataPrefetch.
 *
 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipExtDynDataPrefetch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.5
 */
HIP_TEST_CASE(Unit_hipExtDynDataPrefetch_TemporalHints) {
  int maxRegions = 0;
  HIP_CHECK(hipDeviceGetAttribute(&maxRegions, hipDeviceAttributeMaxDynDataPrefetchRegions, 0));
  if (maxRegions < 1) {
    HIP_SKIP_TEST("Device does not support prefetch regions");
    return;
  }

  constexpr int N = 256;
  constexpr float alpha = 1.0f;
  size_t bytes = N * sizeof(float);

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  HIP_CHECK(hipMalloc(&d_A, bytes));
  HIP_CHECK(hipMalloc(&d_B, bytes));
  HIP_CHECK(hipMalloc(&d_C, bytes));

  std::vector<float> h_A(N, 3.0f), h_B(N, 4.0f), h_C(N);
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

  auto temporal = GENERATE(hipExtDynDataPrefetchTemporalRegular,
                           hipExtDynDataPrefetchTemporalHigh);

  size_t width = (bytes / 256) * 256;

  hipExtDynDataPrefetchRegion region = {};
  region.address = d_A;
  region.width   = width;
  region.height  = 1;
  region.stride  = width;

  hipExtDynDataPrefetchConfig prefetchConfig = {};
  prefetchConfig.numRegions = 1;
  prefetchConfig.temporal   = temporal;
  prefetchConfig.regions[0] = region;
  hipLaunchAttribute attr = makeDynDataPrefetchAttr(&prefetchConfig);

  hipLaunchConfig_t config = {};
  config.gridDim  = dim3{1, 1, 1};
  config.blockDim = dim3{(unsigned)N, 1, 1};
  config.dynamicSmemBytes = 0;
  config.stream   = 0;
  config.attrs    = &attr;
  config.numAttrs = 1;

  int n = N;
  void* args[] = {&d_A, &d_B, &d_C, const_cast<float*>(&alpha), &n};
  HIP_CHECK(hipLaunchKernelExC(&config, reinterpret_cast<void*>(daxpyKernel), args));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    REQUIRE(h_C[i] == Catch::Approx(3.0f + alpha * 4.0f).epsilon(1e-5f));
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
}

/**
 * End doxygen group KernelTest.
 * @}
 */
