/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemcpy2DAsync hipMemcpy2DAsync
 * @{
 * @ingroup MemcpyTest
 * `hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
 *                   size_t spitch, size_t width, size_t height,
 *                   hipMemcpyKind kind, hipStream_t stream = 0 )` -
 * Copies data between host and device.
 */


#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

static constexpr auto NUM_W{16};
static constexpr auto NUM_H{16};
static constexpr auto COLUMNS{6};
static constexpr auto ROWS{6};

/**
 * Test Description
 * ------------------------
 *  - This performs the following scenarios of hipMemcpy2DAsync API on same GPU
      1. H2D-D2D-D2H for Host Memory<-->Device Memory
      2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory

      Input : "A_h" initialized based on data type
         "A_h" --> "A_d" using H2D copy
         "A_d" --> "B_d" using D2D copy
         "B_d" --> "B_h" using D2H copy
      Output: Validating A_h with B_h both should be equal for
        the number of COLUMNS and ROWS copied
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEMPLATE_TEST_CASE(Unit_hipMemcpy2DAsync_Host_N_PinnedMem, int, float, double) {
  CHECK_IMAGE_SUPPORT
  // 1 refers to pinned host memory
  auto mem_type = GENERATE(0, 1);
  auto memcpy_d2d_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr}, *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocating memory
  if (mem_type) {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, NUM_W * NUM_H, true);
  } else {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, NUM_W * NUM_H,
                                  false);
  }
  hipMemcpyKind d2d_type;
  if (memcpy_d2d_type) {
    d2d_type = hipMemcpyDeviceToDevice;
  } else {
    d2d_type = hipMemcpyDeviceToDeviceNoCU;
  }
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, NUM_H));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d), &pitch_B, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<TestType>(NUM_W * NUM_H, A_h, B_h, C_h);
  SECTION("Calling Async apis with stream object created by user") {
    // Host to Device
    HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, A_h, COLUMNS * sizeof(TestType),
                               COLUMNS * sizeof(TestType), ROWS, hipMemcpyHostToDevice, stream));

    // Performs D2D on same GPU device
    HIP_CHECK(hipMemcpy2DAsync(B_d, pitch_B, A_d, pitch_A, COLUMNS * sizeof(TestType), ROWS,
                               d2d_type, stream));

    // hipMemcpy2DAsync Device to Host
    HIP_CHECK(hipMemcpy2DAsync(B_h, COLUMNS * sizeof(TestType), B_d, pitch_B,
                               COLUMNS * sizeof(TestType), ROWS, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  SECTION("Calling Async apis with hipStreamPerThread") {
    // Host to Device
    HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, A_h, COLUMNS * sizeof(TestType),
                               COLUMNS * sizeof(TestType), ROWS, hipMemcpyHostToDevice,
                               hipStreamPerThread));

    // Performs D2D on same GPU device
    HIP_CHECK(hipMemcpy2DAsync(B_d, pitch_B, A_d, pitch_A, COLUMNS * sizeof(TestType), ROWS,
                               d2d_type, hipStreamPerThread));

    // hipMemcpy2DAsync Device to Host
    HIP_CHECK(hipMemcpy2DAsync(B_h, COLUMNS * sizeof(TestType), B_d, pitch_B,
                               COLUMNS * sizeof(TestType), ROWS, hipMemcpyDeviceToHost,
                               hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  }

  // Validating the result
  REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);


  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  if (mem_type) {
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, B_h, C_h, true);
  } else {
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, B_h, C_h, false);
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

TEMPLATE_TEST_CASE("Unit_hipMemcpy2DAsync_multiDevice_StreamOnDiffDevice",
                   "[multigpu]", int, float, double) {
  CHECK_IMAGE_SUPPORT
  auto mem_type = GENERATE(0, 1);
  int numDevices = 0;
  int canAccessPeer = 0;
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(TestType)};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  hipStream_t stream;

  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(0));

      // Allocating memory
      if (mem_type) {
        HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, NUM_W * NUM_H,
                                      true);
      } else {
        HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, NUM_W * NUM_H,
                                      false);
      }
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, NUM_H));
      char* X_d{nullptr};
      size_t pitch_X;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&X_d), &pitch_X, width, NUM_H));

      // Initialize the data
      HipTest::setDefaultData<TestType>(NUM_W * NUM_H, A_h, B_h, C_h);

      // Change device
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipStreamCreate(&stream));

      // Host to Device
      HIP_CHECK(hipMemcpy2DAsync(A_d, pitch_A, A_h, COLUMNS * sizeof(TestType),
                                 COLUMNS * sizeof(TestType), ROWS, hipMemcpyHostToDevice, stream));

      // Device to Device
      HIP_CHECK(hipMemcpy2DAsync(X_d, pitch_X, A_d, pitch_A, COLUMNS * sizeof(TestType), ROWS,
                                 hipMemcpyDeviceToDevice, stream));

      // Device to Host
      HIP_CHECK(hipMemcpy2DAsync(B_h, COLUMNS * sizeof(TestType), X_d, pitch_X,
                                 COLUMNS * sizeof(TestType), ROWS, hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(A_d));
      if (mem_type) {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, B_h, C_h, true);
      } else {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr, A_h, B_h, C_h, false);
      }
      HIP_CHECK(hipFree(X_d));
      HIP_CHECK(hipStreamDestroy(stream));
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

static void hipMemcpy2DAsync_Basic_Size_Test(size_t inc) {
  constexpr int defaultProgramSize = 256 * 1024 * 1024;
  constexpr int N = 2;
  constexpr int value = 42;
  int *in, *out, *dev;
  size_t newSize = 0, inp = 0;
  size_t size = sizeof(int) * N * inc;

  size_t free, total;
  HIP_CHECK(hipMemGetInfo(&free, &total));

  if (free < 2 * size)
    newSize = (free - defaultProgramSize) / 2;
  else
    newSize = size;

  INFO("Array size: " << size / 1024.0 / 1024.0 << " MB or " << size << " Bytes.");
  INFO("Free memory: " << free / 1024.0 / 1024.0 << " MB or " << free << " Bytes");
  INFO("NewSize:" << newSize / 1024.0 / 1024.0 << "MB or " << newSize << " Bytes");

  HIP_CHECK(hipHostMalloc(&in, newSize));
  HIP_CHECK(hipHostMalloc(&out, newSize));
  HIP_CHECK(hipMalloc(&dev, newSize));

  inp = newSize / (sizeof(int) * N);
  for (size_t i = 0; i < N; i++) {
    in[i * inp] = value;
  }

  size_t pitch = sizeof(int) * inp;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemcpy2DAsync(dev, pitch, in, pitch, sizeof(int), N, hipMemcpyHostToDevice, stream));
  HIP_CHECK(
      hipMemcpy2DAsync(out, pitch, dev, pitch, sizeof(int), N, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < N; i++) {
    REQUIRE(out[i * inp] == value);
  }

  HIP_CHECK(hipFree(dev));
  HIP_CHECK(hipHostFree(in));
  HIP_CHECK(hipHostFree(out));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs multidevice size check on hipMemcpy2DAsync API
      1. Verify hipMemcpy2DAsync with 1 << 20 size
      2. Verify hipMemcpy2DAsync with 1 << 21 size
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE(Unit_hipMemcpy2DAsync_multiDevice_Basic_Size_Test) {
  CHECK_IMAGE_SUPPORT
  size_t input = 1 << 20;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));

    SECTION("Verify hipMemcpy2DAsync with 1 << 20 size") {
      hipMemcpy2DAsync_Basic_Size_Test(input);
    }
    SECTION("Verify hipMemcpy2DAsync with 1 << 21 size") {
      input <<= 1;
      hipMemcpy2DAsync_Basic_Size_Test(input);
    }
  }
}

/**
 * End doxygen group MemcpyTest.
 * @}
 */
