/*
Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemcpy2D hipMemcpy2D
 * @{
 * @ingroup MemcpyTest
 * `hipMemcpy2D(void* dst, size_t dpitch, const void* src,
 *              size_t spitch, size_t width, size_t height,
 *              hipMemcpyKind kind)` -
 * Copies data between host and device.
 */

// Testcase Description:
// 1) Verifies the working of Memcpy2D API negative scenarios by
//    Pass NULL to destination pointer
//    Pass NULL to Source pointer
//    Pass width greater than spitch/dpitch
// 2) Verifies hipMemcpy2D API by
//    pass 0 to destionation pitch
//    pass 0 to source pitch
//    pass 0 to width
//    pass 0 to height
// 3) Verifies working of Memcpy2D API on host memory and pinned host memory by
//    performing D2H, D2D and H2D memory kind copies on same GPU
// 4) Verifies working of Memcpy2D API for the following scenarios
//      H2D-D2D-D2H on host and device memory
//      H2D-D2D-D2H on pinned host and device memory
//      H2D-D2D-D2H functionalities where memory is allocated in GPU-0
//      and API is triggered from GPU-1

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

static constexpr auto NUM_W{16};
static constexpr auto NUM_H{16};
static constexpr auto COLUMNS{8};
static constexpr auto ROWS{8};

/**
 * Test Description
 * ------------------------
 *  - This testcases performs the following scenarios of hipMemcpy2D API on same GPU
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
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEMPLATE_TEST_CASE("Unit_hipMemcpy2D_H2D-D2D-D2H", "", int, float, double) {
  CHECK_IMAGE_SUPPORT
  // 1 refers to pinned host memory
  auto mem_type = GENERATE(0, 1);
  auto memcpy_d2d_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr}, *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};

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

  // Host to Device
  HIP_CHECK(hipMemcpy2D(A_d, pitch_A, A_h, COLUMNS * sizeof(TestType), COLUMNS * sizeof(TestType),
                        ROWS, hipMemcpyHostToDevice));

  // Performs D2D on same GPU device
  HIP_CHECK(hipMemcpy2D(B_d, pitch_B, A_d, pitch_A, COLUMNS * sizeof(TestType), ROWS, d2d_type));

  // hipMemcpy2D Device to Host
  HIP_CHECK(hipMemcpy2D(B_h, COLUMNS * sizeof(TestType), B_d, pitch_B, COLUMNS * sizeof(TestType),
                        ROWS, hipMemcpyDeviceToHost));

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
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs the following scenarios of hipMemcpy2D API on same GPU.
    1. H2D-D2D-D2H for Host Memory<-->Device Memory
    2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory
    The src and dst input pointers to hipMemCpy2D add an offset to the pointers
    returned by the allocation functions.

    Input : "A_h" initialized based on data type
             "A_h" --> "A_d" using H2D copy
             "A_d" --> "B_d" using D2D copy
             "B_d" --> "B_h" using D2H copy
    Output: Validating A_h with B_h both should be equal for
            the number of COLUMNS and ROWS copied
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEMPLATE_TEST_CASE("Unit_hipMemcpy2D_H2D-D2D-D2H_WithOffset", "", int, float, double) {
  CHECK_IMAGE_SUPPORT
  // 1 refers to pinned host memory
  auto mem_type = GENERATE(0, 1);
  auto memcpy_d2d_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr}, *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};

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

  // Host to Device
  HIP_CHECK(hipMemcpy2D(A_d + COLUMNS * sizeof(TestType), pitch_A, A_h, COLUMNS * sizeof(TestType),
                        COLUMNS * sizeof(TestType), ROWS, hipMemcpyHostToDevice));

  // Performs D2D on same GPU device
  HIP_CHECK(hipMemcpy2D(B_d + COLUMNS * sizeof(TestType), pitch_B, A_d + COLUMNS * sizeof(TestType),
                        pitch_A, COLUMNS * sizeof(TestType), ROWS, d2d_type));

  // hipMemcpy2D Device to Host
  HIP_CHECK(hipMemcpy2D(B_h, COLUMNS * sizeof(TestType), B_d + COLUMNS * sizeof(TestType), pitch_B,
                        COLUMNS * sizeof(TestType), ROWS, hipMemcpyDeviceToHost));


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
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs the following scenarios of hipMemcpy2D API on same GPU.
    1. H2D-D2D-D2H for Host Memory<-->hipMallocManaged memory
    2. H2D-D2D-D2H for Pinned Host Memory<-->hipMallocManaged memory
    3. H2D-D2D-D2H (kind = hipMemcpyDefault) for Host Memory<-->hipMallocManaged memory
    4. H2D-D2D-D2H (kind = hipMemcpyDefault) for Pinned Host Memory<-->hipMallocManaged memory
    The src and dst input pointers to hipMemCpy2D add an offset to the pointers
    returned by the allocation functions.

    Input : "A_h" initialized based on data type
             "A_h" --> "A_d" using H2D copy
             "A_d" --> "B_d" using D2D copy
             "B_d" --> "B_h" using D2H copy
    Output: Validating A_h with B_h both should be equal for
            the number of COLUMNS and ROWS copied
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEMPLATE_TEST_CASE("Unit_hipMemcpy2D_H2D-D2D-D2H_Managed_WithOffset", "", int, float, double) {
  CHECK_IMAGE_SUPPORT
  // 1 refers to pinned host memory
  auto mem_type = GENERATE(0, 1);
  auto memcpy_default = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr}, *B_d{nullptr};

  // Allocating memory
  if (mem_type) {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, NUM_W * NUM_H, true);
  } else {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr, &A_h, &B_h, &C_h, NUM_W * NUM_H,
                                  false);
  }
  HIP_CHECK(
      hipMallocManaged(reinterpret_cast<void**>(&A_d), (COLUMNS * ROWS + 1) * sizeof(TestType)));
  HIP_CHECK(
      hipMallocManaged(reinterpret_cast<void**>(&B_d), (COLUMNS * ROWS + 1) * sizeof(TestType)));

  size_t pitch_A = COLUMNS * sizeof(TestType);
  size_t pitch_B = COLUMNS * sizeof(TestType);
  ;

  // Initialize the data
  HipTest::setDefaultData<TestType>(NUM_W * NUM_H, A_h, B_h, C_h);

  // Host to Device
  HIP_CHECK(hipMemcpy2D(A_d + 1, pitch_A, A_h, COLUMNS * sizeof(TestType),
                        COLUMNS * sizeof(TestType), ROWS,
                        memcpy_default ? hipMemcpyDefault : hipMemcpyHostToDevice));

  // Performs D2D on same GPU device
  HIP_CHECK(hipMemcpy2D(B_d + 1, pitch_B, A_d + 1, pitch_A, COLUMNS * sizeof(TestType), ROWS,
                        memcpy_default ? hipMemcpyDefault : hipMemcpyDeviceToDevice));

  // hipMemcpy2D Device to Host
  HIP_CHECK(hipMemcpy2D(B_h, COLUMNS * sizeof(TestType), B_d + 1, pitch_B,
                        COLUMNS * sizeof(TestType), ROWS,
                        memcpy_default ? hipMemcpyDefault : hipMemcpyDeviceToHost));


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
}

/**
 * Test Description
 * ------------------------
 *  - This testcases performs the following scenarios of hipMemcpy2D API on Peer GPU
    1. H2D-D2D-D2H for Host Memory<-->Device Memory
    2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory
    3. Device context change where memory is allocated in GPU-0
       and API is trigerred from GPU-1

    Input : "A_h" initialized based on data type
             "A_h" --> "A_d" using H2D copy
             "A_d" --> "X_d" using D2D copy
             "X_d" --> "B_h" using D2H copy
    Output: Validating A_h with B_h both should be equal for
            the number of COLUMNS and ROWS copied
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEMPLATE_TEST_CASE("Unit_hipMemcpy2D_multiDevice-D2D", "", int, float, double) {
  CHECK_IMAGE_SUPPORT
  auto mem_type = GENERATE(0, 1);
  int numDevices = 0;
  int canAccessPeer = 0;
  TestType *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(TestType)};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
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

      // Initialize the data
      HipTest::setDefaultData<TestType>(NUM_W * NUM_H, A_h, B_h, C_h);

      char* X_d{nullptr};
      size_t pitch_X;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&X_d), &pitch_X, width, NUM_H));

      // Change device
      HIP_CHECK(hipSetDevice(1));

      // Host to Device
      HIP_CHECK(hipMemcpy2D(A_d, pitch_A, A_h, COLUMNS * sizeof(TestType),
                            COLUMNS * sizeof(TestType), ROWS, hipMemcpyHostToDevice));

      // Device to Device
      HIP_CHECK(hipMemcpy2D(X_d, pitch_X, A_d, pitch_A, COLUMNS * sizeof(TestType), ROWS,
                            hipMemcpyDeviceToDevice));

      // Device to Host
      HIP_CHECK(hipMemcpy2D(B_h, COLUMNS * sizeof(TestType), X_d, pitch_X,
                            COLUMNS * sizeof(TestType), ROWS, hipMemcpyDeviceToHost));

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
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

/**
 * Test Description
 * ------------------------
 *  - This Testcase verifies the null size checks of hipMemcpy2D API
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipMemcpy2D_SizeCheck") {
  CHECK_IMAGE_SUPPORT
  HIP_CHECK(hipSetDevice(0));
  int *A_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(int)};

  // Allocating memory
  HipTest::initArrays<int>(nullptr, nullptr, nullptr, &A_h, nullptr, nullptr, NUM_W * NUM_H);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<int>(NUM_W * NUM_H, A_h, nullptr, nullptr);

  SECTION("hipMemcpy2D API where Source Pitch is zero") {
    REQUIRE(hipMemcpy2D(A_h, 0, A_d, pitch_A, NUM_W, NUM_H, hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API where Destination Pitch is zero") {
    REQUIRE(hipMemcpy2D(A_h, width, A_d, 0, NUM_W, NUM_H, hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API where height is zero") {
    REQUIRE(hipMemcpy2D(A_h, width, A_d, pitch_A, NUM_W, 0, hipMemcpyDeviceToHost) == hipSuccess);
  }

  SECTION("hipMemcpy2D API where width is zero") {
    REQUIRE(hipMemcpy2D(A_h, width, A_d, pitch_A, 0, NUM_H, hipMemcpyDeviceToHost) == hipSuccess);
  }

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *  - This Testcase verifies all the negative scenarios of hipMemcpy2D API
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipMemcpy2D_Negative") {
  CHECK_IMAGE_SUPPORT
  HIP_CHECK(hipSetDevice(0));
  int *A_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(int)};

  // Allocating memory
  HipTest::initArrays<int>(nullptr, nullptr, nullptr, &A_h, nullptr, nullptr, NUM_W * NUM_H);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<int>(NUM_W * NUM_H, A_h, nullptr, nullptr);

  SECTION("hipMemcpy2D API by Passing nullptr to destination") {
    REQUIRE(hipMemcpy2D(nullptr, width, A_d, pitch_A, COLUMNS * sizeof(int), ROWS,
                        hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API by Passing nullptr to destination") {
    REQUIRE(hipMemcpy2D(nullptr, width, nullptr, pitch_A, COLUMNS * sizeof(int), ROWS,
                        hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API where width is greater than destination pitch") {
    REQUIRE(hipMemcpy2D(A_h, 10, A_d, pitch_A, COLUMNS * sizeof(int), ROWS,
                        hipMemcpyDeviceToHost) != hipSuccess);
  }

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

static void hipMemcpy2D_Basic_Size_Test(size_t inc) {
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

  HIP_CHECK(hipMemcpy2D(dev, pitch, in, pitch, sizeof(int), N, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2D(out, pitch, dev, pitch, sizeof(int), N, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    REQUIRE(out[i * inp] == value);
  }

  HIP_CHECK(hipFree(dev));
  HIP_CHECK(hipHostFree(in));
  HIP_CHECK(hipHostFree(out));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs multidevice size check on hipMemcpy2D API
      1. Verify hipMemcpy2D with 1 << 20 size
      2. Verify hipMemcpy2D with 1 << 21 size
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipMemcpy2D_multiDevice_Basic_Size_Test") {
  CHECK_IMAGE_SUPPORT
  size_t input = 1 << 20;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));

    SECTION("Verify hipMemcpy2D with 1 << 20 size") { hipMemcpy2D_Basic_Size_Test(input); }
    SECTION("Verify hipMemcpy2D with 1 << 21 size") {
      input <<= 1;
      hipMemcpy2D_Basic_Size_Test(input);
    }
  }
}

/**
 * End doxygen group MemcpyTest.
 * @}
 */
