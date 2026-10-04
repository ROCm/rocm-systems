/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
/**
 * @addtogroup hipMemcpy hipMemcpy
 * @{
 * @ingroup MemoryTest
 * `hipError_t 	hipMemcpy (void *dst, const void *src,
 *                         size_t sizeBytes, hipMemcpyKind kind)` -
 * Copy data from src to dst.
 */
static void fillDataTransfer2Dev(int* hostBuf, size_t len) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < len; i++) {
    hostBuf[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}
/**
 * Test Description
 * ------------------------
 *    - Create 2 device memory chunks (Ad, Bd) and 2 host memory chunk (Ah, Bh).
 * On default/user stream, perform the following memcpies in queue => Ah->Ad, Ad->Bd
 * (hipMemcpyDeviceToDeviceNoCU), Kernel Launch (Bd*Bd), Bd->Bh. Verify data in Bh.
 * ------------------------
 *    - catch\unit\memory\hipMemcpyDeviceToDeviceNoCU.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemcpyDeviceToDeviceNoCU_SingleStream) {
  auto testAsync = GENERATE(0, 1);
  auto isDefaultStrm = GENERATE(0, 1);
  constexpr int N = 1 << 18;
  size_t buffer_size = N * sizeof(int);
  constexpr unsigned threadsPerBlock = 256;
  constexpr int blocks =
      (N % threadsPerBlock == 0) ? (N / threadsPerBlock) : ((N / threadsPerBlock) + 1);
  // Allocate device resources
  int *Ad, *Bd;
  HIP_CHECK(hipMalloc(&Ad, buffer_size));
  HIP_CHECK(hipMalloc(&Bd, buffer_size));
  // Allocate host resources
  int* Ah = new int[N];
  REQUIRE(Ah != nullptr);
  int* Bh = new int[N];
  REQUIRE(Bh != nullptr);
  // Check whether to execute on default stream or user stream
  hipStream_t strm = 0;
  if (isDefaultStrm == 1) {
    HIP_CHECK(hipStreamCreate(&strm));
  }
  // fill Ah with random data
  fillDataTransfer2Dev(Ah, N);
  if (0 == testAsync) {
    HIP_CHECK(hipMemcpy(Ad, Ah, N * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, Ad, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU));
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, strm, Bd, Bd,
                       N);
    HIP_CHECK(hipMemcpy(Bh, Bd, N * sizeof(int), hipMemcpyDeviceToHost));
  } else {
    HIP_CHECK(hipMemcpyAsync(Ad, Ah, N * sizeof(int), hipMemcpyHostToDevice, strm));
    HIP_CHECK(hipMemcpyAsync(Bd, Ad, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU, strm));
    hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, strm, Bd, Bd,
                       N);
    HIP_CHECK(hipMemcpyAsync(Bh, Bd, N * sizeof(int), hipMemcpyDeviceToHost, strm));
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < N; i++) {
    REQUIRE(Bh[i] == (Ah[i] * Ah[i]));
  }
  if (isDefaultStrm == 1) {
    HIP_CHECK(hipStreamDestroy(strm));
  }
  // Delete resources
  delete[] Ah;
  delete[] Bh;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
}

/**
 * Test Description
 * ------------------------
 *    - Create 3 device memory chunks (Ad, Bd, Cd) and 2 host memory chunk (Ah, Bh).
 * On default/user stream, perform the following memcpies in queue => Ah->Ad, Ad->Bd
 * (hipMemcpyAsync with flag = hipMemcpyDeviceToDevice), Bd->Cd (with flag =
 * hipMemcpyDeviceToDeviceNoCU, Cd->Bh. Verify data in Bh.
 * ------------------------
 *    - catch\unit\memory\hipMemcpyDeviceToDeviceNoCU.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemcpyDeviceToDeviceNoCU_WithCU_NoCU_Comb_SingleStrm) {
  constexpr int N = 1 << 18;
  size_t buffer_size = N * sizeof(int);
  // Allocate device resources
  int *Ad, *Bd, *Cd;
  HIP_CHECK(hipMalloc(&Ad, buffer_size));
  HIP_CHECK(hipMalloc(&Bd, buffer_size));
  HIP_CHECK(hipMalloc(&Cd, buffer_size));
  // Allocate host resources
  int* Ah = new int[N];
  REQUIRE(Ah != nullptr);
  int* Bh = new int[N];
  REQUIRE(Bh != nullptr);
  // Check whether to execute on default stream or user stream
  hipStream_t strm = 0;
  HIP_CHECK(hipStreamCreate(&strm));
  // fill Ah with random data
  fillDataTransfer2Dev(Ah, N);
  HIP_CHECK(hipMemcpyAsync(Ad, Ah, N * sizeof(int), hipMemcpyHostToDevice, strm));
  SECTION("Memcpy withCU first then without CU") {
    HIP_CHECK(hipMemcpyAsync(Bd, Ad, N * sizeof(int), hipMemcpyDeviceToDevice, strm));
    HIP_CHECK(hipMemcpyAsync(Cd, Bd, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU, strm));
  }
  SECTION("Memcpy without CU first then with CU") {
    HIP_CHECK(hipMemcpyAsync(Bd, Ad, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU, strm));
    HIP_CHECK(hipMemcpyAsync(Cd, Bd, N * sizeof(int), hipMemcpyDeviceToDevice, strm));
  }
  SECTION("Memcpy without CU twice") {
    HIP_CHECK(hipMemcpyAsync(Bd, Ad, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU, strm));
    HIP_CHECK(hipMemcpyAsync(Cd, Bd, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU, strm));
  }
  HIP_CHECK(hipMemcpyAsync(Bh, Cd, N * sizeof(int), hipMemcpyDeviceToHost, strm));
  HIP_CHECK(hipStreamSynchronize(strm));
  for (int i = 0; i < N; i++) {
    REQUIRE(Bh[i] == Ah[i]);
  }
  HIP_CHECK(hipStreamDestroy(strm));
  // Delete resources
  delete[] Ah;
  delete[] Bh;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
}

/**
 * Test Description
 * ------------------------
 *    - Create 3 device memory chunks (Ad, Bd, Cd) and 3 host memory chunk
 * (Ah, Bh, Ch). On user stream1, perform the following memcpies in queue
 * => On default stream, perform Ah->Ad memcpy. On stream1, perform Ad->Bd
 * (hipMemcpyAsync with flag = hipMemcpyDeviceToDeviceNoCU) and Bd->Bh.
 * On stream2, perform Ad->Cd (hipMemcpyAsync with flag =
 * hipMemcpyDeviceToDeviceNoCU) and Cd -> Ch. Wait for both stream1
 * and stream2. Verify output in Bh, Ch.
 * ------------------------
 *    - catch\unit\memory\hipMemcpyDeviceToDeviceNoCU.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemcpyDeviceToDeviceNoCU_NoCU_MulStrm) {
  constexpr int N = 1 << 18;
  size_t buffer_size = N * sizeof(int);
  // Allocate device resources
  int *Ad, *Bd, *Cd;
  HIP_CHECK(hipMalloc(&Ad, buffer_size));
  HIP_CHECK(hipMalloc(&Bd, buffer_size));
  HIP_CHECK(hipMalloc(&Cd, buffer_size));
  // Allocate host resources
  int* Ah = new int[N];
  REQUIRE(Ah != nullptr);
  int* Bh = new int[N];
  REQUIRE(Bh != nullptr);
  int* Ch = new int[N];
  REQUIRE(Ch != nullptr);
  // fill Ah with random data
  fillDataTransfer2Dev(Ah, N);
  HIP_CHECK(hipMemcpyAsync(Ad, Ah, N * sizeof(int), hipMemcpyHostToDevice, 0));
  hipStream_t strm1, strm2;
  HIP_CHECK(hipStreamCreate(&strm1));
  HIP_CHECK(hipStreamCreate(&strm2));
  HIP_CHECK(hipMemcpyAsync(Bd, Ad, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU, strm1));
  HIP_CHECK(hipMemcpyAsync(Cd, Ad, N * sizeof(int), hipMemcpyDeviceToDeviceNoCU, strm2));
  HIP_CHECK(hipMemcpyAsync(Bh, Bd, N * sizeof(int), hipMemcpyDeviceToHost, strm1));
  HIP_CHECK(hipMemcpyAsync(Ch, Cd, N * sizeof(int), hipMemcpyDeviceToHost, strm2));
  HIP_CHECK(hipStreamSynchronize(strm1));
  HIP_CHECK(hipStreamSynchronize(strm2));
  for (int i = 0; i < N; i++) {
    REQUIRE(Bh[i] == Ah[i]);
    REQUIRE(Ch[i] == Ah[i]);
  }
  HIP_CHECK(hipStreamDestroy(strm2));
  HIP_CHECK(hipStreamDestroy(strm1));
  // Delete resources
  delete[] Ah;
  delete[] Bh;
  delete[] Ch;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
}

/**
 * Test Description
 * ------------------------
 *    - Create 3 device memory chunks (Ad, Bd, Cd) and 1 host memory
 * chunk (Ah) of size 1024 MB. Initialize the host memory chunk (Ah)
 * with random data. On default stream, copy from host to device
 * (Ah -> Ad) the random data. Create 2 streams stream1 and stream2.
 * Once completed, on stream1 copy from device to device (Ad -> Bd)
 * using hipMemcpyDeviceToDeviceNoCU (Async Copy) and on stream2,
 * launch a kernel to copy from (Ad -> Cd). Validate data in Bd and
 * Cd are same.
 * ------------------------
 *    - catch\unit\memory\hipMemcpyDeviceToDeviceNoCU.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemcpyDeviceToDeviceNoCU_Memcpy_Kernel_InParallel) {
  constexpr int N = 1 << 24;
  size_t buffer_size = N * sizeof(int);
  constexpr unsigned threadsPerBlock = 1024;
  constexpr int blocks =
      (N % threadsPerBlock == 0) ? (N / threadsPerBlock) : ((N / threadsPerBlock) + 1);
  // Allocate device resources
  int *Ad, *Bd, *Cd;
  HIP_CHECK(hipMalloc(&Ad, buffer_size));
  HIP_CHECK(hipMalloc(&Bd, buffer_size));
  HIP_CHECK(hipMalloc(&Cd, buffer_size));
  // Allocate host resources
  int* Ah = new int[N];
  REQUIRE(Ah != nullptr);
  int* Bh = new int[N];
  REQUIRE(Bh != nullptr);
  int* Ch = new int[N];
  REQUIRE(Ch != nullptr);
  // fill Ah with random data
  fillDataTransfer2Dev(Ah, N);
  HIP_CHECK(hipMemcpy(Ad, Ah, buffer_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  hipStream_t strm1, strm2;
  HIP_CHECK(hipStreamCreate(&strm1));
  HIP_CHECK(hipStreamCreate(&strm2));
  HIP_CHECK(hipMemcpyAsync(Bd, Ad, buffer_size, hipMemcpyDeviceToDeviceNoCU, strm1));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks), dim3(threadsPerBlock), 0, strm2, Ad, Cd,
                     N);
  HIP_CHECK(hipMemcpyAsync(Bh, Bd, buffer_size, hipMemcpyDeviceToHost, strm1));
  HIP_CHECK(hipMemcpyAsync(Ch, Cd, buffer_size, hipMemcpyDeviceToHost, strm2));
  HIP_CHECK(hipStreamSynchronize(strm1));
  HIP_CHECK(hipStreamSynchronize(strm2));
  for (int i = 0; i < N; i++) {
    INFO("index: " << i << " out of : " << N);
    REQUIRE(Bh[i] == Ah[i]);
    REQUIRE(Ch[i] == (Ah[i] * Ah[i]));
  }
  HIP_CHECK(hipStreamDestroy(strm2));
  HIP_CHECK(hipStreamDestroy(strm1));
  // Delete resources
  delete[] Ah;
  delete[] Bh;
  delete[] Ch;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipFree(Cd));
}

/**
 * Test Description
 * ------------------------
 *    - For every peer capable device pair, seed a source chunk on the first device
 * and zero a destination chunk on the second. On a freshly created stream, copy
 * from device to peer device using hipMemcpyDeviceToDeviceNoCU (Async Copy) and
 * validate the data landed on the destination device. The copy is the first SDMA
 * transfer issued on the source device, so it exercises engine selection without
 * any host transfer having already reserved an engine.
 * ------------------------
 *    - catch\unit\memory\hipMemcpyDeviceToDeviceNoCU.cc
 * Test requirements
 * ------------------------
 *    - Multi device
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemcpyDeviceToDeviceNoCU_PeerDevice) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  constexpr int N = 1 << 22;
  constexpr int pattern = 0x5A5A5A5A;
  size_t buffer_size = N * sizeof(int);
  int* Bh = new int[N];
  REQUIRE(Bh != nullptr);

  for (int srcDev = 0; srcDev < numDevices; srcDev++) {
    for (int dstDev = 0; dstDev < numDevices; dstDev++) {
      if (srcDev == dstDev) continue;
      int canAccessPeer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, srcDev, dstDev));
      if (!canAccessPeer) continue;
      INFO("src device: " << srcDev << " dst device: " << dstDev);

      int *Ad, *Bd;
      HIP_CHECK(hipSetDevice(dstDev));
      HIP_CHECK(hipMalloc(&Bd, buffer_size));
      HIP_CHECK(hipMemset(Bd, 0, buffer_size));
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipSetDevice(srcDev));
      HIP_CHECK(hipMalloc(&Ad, buffer_size));
      HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(Ad), pattern, N));
      HIP_CHECK(hipDeviceSynchronize());

      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMemcpyAsync(Bd, Ad, buffer_size, hipMemcpyDeviceToDeviceNoCU, strm));
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipStreamDestroy(strm));
      HIP_CHECK(hipFree(Ad));

      HIP_CHECK(hipSetDevice(dstDev));
      HIP_CHECK(hipMemcpy(Bh, Bd, buffer_size, hipMemcpyDeviceToHost));
      HIP_CHECK(hipFree(Bd));
      size_t mismatches = 0;
      for (int i = 0; i < N; i++) {
        if (Bh[i] != pattern) mismatches++;
      }
      INFO("mismatches: " << mismatches << " out of : " << N);
      REQUIRE(mismatches == 0);
    }
  }
  HIP_CHECK(hipSetDevice(0));
  delete[] Bh;
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
