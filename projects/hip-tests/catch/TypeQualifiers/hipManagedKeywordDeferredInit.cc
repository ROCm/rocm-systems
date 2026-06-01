/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <vector>

// Tests in this file each drive a HIP API against a __managed__ symbol to
// verify that, when deferred loading is in effect, the API ensures the
// symbol's storage has been initialized on the host and mapped onto the target
// device before the operation runs. If an entry point skips that step, the
// corresponding test should observe stale or invalid data in g_managed_*.

constexpr int kN = 1024;
constexpr int kBlockSize = 256;
constexpr int kNumBlocks = (kN + kBlockSize - 1) / kBlockSize;

constexpr int kKernelAddValue = 1;
constexpr int kSeed = 7;
constexpr int kSeedMul = 5;

constexpr int k3dDim = 8;
constexpr int kStaticInitLen = 8;

__managed__ int g_managed_a[kN];
__managed__ int g_managed_b[kN];
__managed__ int g_managed_3d[k3dDim][k3dDim][k3dDim];
__managed__ int g_managed_initialized[kStaticInitLen] = {1, 2, 3, 4, 5, 6, 7, 8};

static __global__ void AddConst(int* data, int n, int addend) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += addend;
}

static __global__ void SumTwo(int* a, int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpy) {
  CHECK_MANAGED_MEMORY_SUPPORT

  std::vector<int> host(kN);
  for (int i = 0; i < kN; ++i) {
    host[i] = i;
  }
  HIP_CHECK(hipMemcpy(g_managed_a, host.data(), kN * sizeof(int),
                      hipMemcpyHostToDevice));

  AddConst<<<dim3(kNumBlocks), dim3(kBlockSize)>>>(g_managed_a, kN,
                                                   kKernelAddValue);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kN; ++i) {
    INFO("Index " << i);
    REQUIRE(g_managed_a[i] == i + kKernelAddValue);
  }
}

// Tests that deferred loading works as expected when hipMemcpyHtoD
// is the first HIP call in the test that touches a managed symbol.
HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpyHtoD) {
  CHECK_MANAGED_MEMORY_SUPPORT

  SECTION("Sync") {
    std::vector<int> host(kN);
    for (int i = 0; i < kN; ++i) host[i] = i + kSeed;
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(g_managed_a),
                            host.data(), kN * sizeof(int)));

    AddConst<<<dim3(kNumBlocks), dim3(kBlockSize)>>>(g_managed_a, kN,
                                                     kKernelAddValue);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    for (int i = 0; i < kN; ++i) {
      INFO("Index " << i);
      REQUIRE(g_managed_a[i] == i + kSeed + kKernelAddValue);
    }
  }

  SECTION("Async") {
    std::vector<int> host(kN);
    for (int i = 0; i < kN; ++i) host[i] = i * kSeedMul;
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyHtoDAsync(
        reinterpret_cast<hipDeviceptr_t>(g_managed_a), host.data(),
        kN * sizeof(int), stream));
    AddConst<<<dim3(kNumBlocks), dim3(kBlockSize), 0, stream>>>(
        g_managed_a, kN, kKernelAddValue);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));

    for (int i = 0; i < kN; ++i) {
      INFO("Index " << i);
      REQUIRE(g_managed_a[i] == i * kSeedMul + kKernelAddValue);
    }
  }
}

// Tests that deferred loading works as expected when hipMemcpyDtoH
// is the first HIP call in the test that touches a managed symbol.
HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpyDtoH) {
  CHECK_MANAGED_MEMORY_SUPPORT

  SECTION("Sync") {
    int out[kStaticInitLen] = {};
    HIP_CHECK(hipMemcpyDtoH(out,
                            reinterpret_cast<hipDeviceptr_t>(g_managed_initialized),
                            kStaticInitLen * sizeof(int)));
    for (int i = 0; i < kStaticInitLen; ++i) {
      INFO("Index " << i);
      REQUIRE(out[i] == i + 1);
    }
  }

  SECTION("Async") {
    int out[kStaticInitLen] = {};
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyDtoHAsync(
        out, reinterpret_cast<hipDeviceptr_t>(g_managed_initialized),
        kStaticInitLen * sizeof(int), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));

    for (int i = 0; i < kStaticInitLen; ++i) {
      INFO("Index " << i);
      REQUIRE(out[i] == i + 1);
    }
  }
}

// Tests that deferred loading works as expected when hipMemcpyDtoD
// is the first HIP call in the test that touches a managed symbol.
HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpyDtoD) {
  CHECK_MANAGED_MEMORY_SUPPORT

  SECTION("Sync") {
    HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(g_managed_b),
                            reinterpret_cast<hipDeviceptr_t>(g_managed_initialized),
                            kStaticInitLen * sizeof(int)));

    int out[kStaticInitLen] = {};
    HIP_CHECK(hipMemcpy(out, g_managed_b, kStaticInitLen * sizeof(int),
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < kStaticInitLen; ++i) {
      INFO("Index " << i);
      REQUIRE(out[i] == i + 1);
    }
  }

  SECTION("Async") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyDtoDAsync(
        reinterpret_cast<hipDeviceptr_t>(g_managed_b),
        reinterpret_cast<hipDeviceptr_t>(g_managed_initialized),
        kStaticInitLen * sizeof(int), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));

    int out[kStaticInitLen] = {};
    HIP_CHECK(hipMemcpy(out, g_managed_b, kStaticInitLen * sizeof(int),
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < kStaticInitLen; ++i) {
      INFO("Index " << i);
      REQUIRE(out[i] == i + 1);
    }
  }
}

HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpyWithStream) {
  CHECK_MANAGED_MEMORY_SUPPORT

  std::vector<int> host(kN);
  for (int i = 0; i < kN; ++i) host[i] = i * kSeedMul;

  SECTION("NullStream") {
    HIP_CHECK(hipMemcpyWithStream(g_managed_a, host.data(),
                                  kN * sizeof(int), hipMemcpyHostToDevice,
                                  nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < kN; ++i) {
      INFO("Index " << i);
      REQUIRE(g_managed_a[i] == i * kSeedMul);
    }
  }
  SECTION("CreatedStream") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyWithStream(g_managed_a, host.data(),
                                  kN * sizeof(int), hipMemcpyHostToDevice,
                                  stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    for (int i = 0; i < kN; ++i) {
      INFO("Index " << i);
      REQUIRE(g_managed_a[i] == i * kSeedMul);
    }
  }
}

HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemset) {
  CHECK_MANAGED_MEMORY_SUPPORT

  constexpr unsigned char kFillByte = 0x42;
  constexpr int kExpected = 0x42424242;

  HIP_CHECK(hipMemset(g_managed_a, kFillByte, kN * sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kN; ++i) {
    INFO("Index " << i);
    REQUIRE(g_managed_a[i] == kExpected);
  }
}

// Verifies that more than one managed symbol on the same device is initialized
// (not just the first one touched).
HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpy_MultipleSymbols) {
  CHECK_MANAGED_MEMORY_SUPPORT

  std::vector<int> hostA(kN);
  std::vector<int> hostB(kN);

  for (int i = 0; i < kN; ++i) {
    hostA[i] = i + kSeed;
    hostB[i] = i * kSeedMul;
  }
  HIP_CHECK(hipMemcpy(g_managed_a, hostA.data(), kN * sizeof(int),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(g_managed_b, hostB.data(), kN * sizeof(int),
                      hipMemcpyHostToDevice));

  SumTwo<<<dim3(kNumBlocks), dim3(kBlockSize)>>>(g_managed_a, g_managed_b, kN);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < kN; ++i) {
    INFO("Index " << i);
    REQUIRE(g_managed_a[i] == (i + kSeed) + i * kSeedMul);
  }
}

HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpy3D) {
  CHECK_MANAGED_MEMORY_SUPPORT

  constexpr int kZStride = 10000;
  constexpr int kYStride = 100;
  auto uniqueCellValue = [](int x, int y, int z) {
    return z * kZStride + y * kYStride + x;
  };

  int host[k3dDim][k3dDim][k3dDim];
  for (int z = 0; z < k3dDim; ++z) {
    for (int y = 0; y < k3dDim; ++y) {
      for (int x = 0; x < k3dDim; ++x) {
        host[z][y][x] = uniqueCellValue(x, y, z);
      }
    }
  }

  auto cubePitched = [](void* p) {
    return make_hipPitchedPtr(p, k3dDim * sizeof(int), k3dDim, k3dDim);
  };

  hipMemcpy3DParms parms = {};
  parms.srcPtr = cubePitched(host);
  parms.dstPtr = cubePitched(g_managed_3d);
  parms.extent = make_hipExtent(k3dDim * sizeof(int), k3dDim, k3dDim);
  parms.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&parms));
  HIP_CHECK(hipDeviceSynchronize());

  for (int z = 0; z < k3dDim; ++z) {
    for (int y = 0; y < k3dDim; ++y) {
      for (int x = 0; x < k3dDim; ++x) {
        INFO("(" << x << ',' << y << ',' << z << ')');
        REQUIRE(g_managed_3d[z][y][x] == uniqueCellValue(x, y, z));
      }
    }
  }
}

HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemset3D) {
  CHECK_MANAGED_MEMORY_SUPPORT

  // hipMemset3D writes the low byte of `value` into every byte of the
  // destination region, so each int in the cube ends up as the 4-byte
  // broadcast of that byte.
  constexpr unsigned char kFillByte = 0x42;
  constexpr int kExpected = 0x42424242;

  auto cubePitched = [](void* p) {
    return make_hipPitchedPtr(p, k3dDim * sizeof(int), k3dDim, k3dDim);
  };

  HIP_CHECK(hipMemset3D(cubePitched(g_managed_3d), kFillByte,
                        make_hipExtent(k3dDim * sizeof(int), k3dDim, k3dDim)));
  HIP_CHECK(hipDeviceSynchronize());

  for (int z = 0; z < k3dDim; ++z) {
    for (int y = 0; y < k3dDim; ++y) {
      for (int x = 0; x < k3dDim; ++x) {
        INFO("(" << x << ',' << y << ',' << z << ')');
        REQUIRE(g_managed_3d[z][y][x] == kExpected);
      }
    }
  }
}

// cudaMemcpyBatchAsync was introduced in CUDA 12.8.
#if HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12080)
HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpyBatchAsync) {
  CHECK_MANAGED_MEMORY_SUPPORT

  constexpr size_t kBatchCount = 2;
  constexpr int kSeedA = 100;
  constexpr int kSeedB = 200;

  std::vector<int> hostA(kN);
  std::vector<int> hostB(kN);
  for (int i = 0; i < kN; ++i) {
    hostA[i] = i + kSeedA;
    hostB[i] = i + kSeedB;
  }

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  void *dsts[kBatchCount];
  dsts[0] = reinterpret_cast<void *>(g_managed_a);
  dsts[1] = reinterpret_cast<void *>(g_managed_b);
  void *srcs[kBatchCount] = {reinterpret_cast<void *>(hostA.data()),
                             reinterpret_cast<void *>(hostB.data())};
  size_t sizes[kBatchCount] = {kN * sizeof(int), kN * sizeof(int)};
  size_t attrsIdxs[1] = {0};
  size_t failIdx = 0;

  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, kBatchCount, nullptr,
                                attrsIdxs, 0, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  for (int i = 0; i < kN; ++i) {
    INFO("A Index " << i);
    REQUIRE(g_managed_a[i] == i + kSeedA);
    INFO("B Index " << i);
    REQUIRE(g_managed_b[i] == i + kSeedB);
  }
}
#endif // HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12080)

HIP_TEST_CASE(Unit_hipManagedKeyword_hipMemcpyPeer) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    return;
  }
  CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(0);
  CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(1);
  int canAccess = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, 0, 1));
  if (!canAccess) {
    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
    return;
  }

  HIP_CHECK(hipSetDevice(0));
  for (int i = 0; i < kN; ++i) g_managed_a[i] = i;

  HIP_CHECK(hipMemcpyPeer(g_managed_b, /*dstDevice=*/1, g_managed_a,
                          /*srcDevice=*/0, kN * sizeof(int)));

  for (int i = 0; i < kN; ++i) {
    INFO("Index " << i);
    REQUIRE(g_managed_b[i] == i);
  }
}
