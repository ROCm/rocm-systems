/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemcpy hipMemcpy
 * @{
 * @ingroup perfMemoryTest
 * `hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */
// #define ENABLE_DEBUG 1
#include <time.h>
#include <hip_test_common.hh>

#define NUM_SIZE 19   //  size up to 16M
#define NUM_ITER 500  //  Total GPU memory up to 16M*500=8G

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (int i = 0; i < len; i++) {
    A[i] = val;
  }
}

void setup(size_t* size, int* num, int** pA, const size_t totalGlobalMem) {
  for (int i = 0; i < *num; i++) {
    size[i] = 1 << (i + 6);
    if ((NUM_ITER + 1) * size[i] > totalGlobalMem) {
      *num = i;
      break;
    }
  }
  *pA = reinterpret_cast<int*>(malloc(size[*num - 1]));
  valSet(*pA, 1, size[*num - 1]);
}

void testInit(size_t size, int* A) {
  int* Ad;

  clock_t start = clock();
  HIP_CHECK(hipMalloc(&Ad, size));  //  hip::init() will be called
  clock_t end = clock();
  double uS = (end - start) * 1000000. / CLOCKS_PER_SEC;
  CONSOLE_PRINT("Initial: hipMalloc(%zu) cost %.2fus\n", size, uS);

  start = clock();
  HIP_CHECK(hipMemcpy(Ad, A, size, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  end = clock();
  uS = (end - start) * 1000000. / CLOCKS_PER_SEC;
  CONSOLE_PRINT("hipMemcpy(%zu) cost %.2fus\n", size, uS);

  start = clock();
  HIP_CHECK(hipFree(Ad));
  end = clock();
  uS = (end - start) * 1000000. / CLOCKS_PER_SEC;
  CONSOLE_PRINT("hipFree(%zu) cost %.2fus\n", size, uS);
}

static bool hipPerfMemMallocCpyFree_test() {
  double uS;
  clock_t start, end;
  size_t size[NUM_SIZE] = {0};
  int* Ad[NUM_ITER] = {nullptr};
  int* A;
  hipDeviceProp_t props;
  memset(&props, 0, sizeof(props));
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  CONSOLE_PRINT("totalGlobalMem: %zu\n", props.totalGlobalMem);

  int num = NUM_SIZE;
  setup(size, &num, &A, props.totalGlobalMem);
  testInit(size[0], A);

  for (int i = 0; i < num; i++) {
    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      HIP_CHECK(hipMalloc(&Ad[j], size[i]));
    }
    end = clock();
    uS = (end - start) * 1000000. / (NUM_ITER * CLOCKS_PER_SEC);
    CONSOLE_PRINT("hipMalloc(%zu) cost %.2fus\n", size[i], uS);

    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      HIP_CHECK(hipMemcpy(Ad[j], A, size[i], hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());
    end = clock();
    uS = (end - start) * 1000000. / (NUM_ITER * CLOCKS_PER_SEC);
    CONSOLE_PRINT("hipMemcpy(%zu) cost %.2fus\n", size[i], uS);

    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      HIP_CHECK(hipFree(Ad[j]));
      Ad[j] = nullptr;
    }
    end = clock();
    double uS = (end - start) * 1000000. / (NUM_ITER * CLOCKS_PER_SEC);
    CONSOLE_PRINT("hipFree(%zu) cost %.2fus\n", size[i], uS);
  }
  free(A);
  return true;
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipPerfMemMallocCpyFree status.
 * Test source
 * ------------------------
 *  - perftests/memory/hipPerfMemMallocCpyFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE(Perf_hipPerfMemMallocCpyFree_test) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    SUCCEED(
        "Skipped testcase hipPerfDevMemReadSpeed as"
        "there is no device to test.");
  } else {
    REQUIRE(true == hipPerfMemMallocCpyFree_test());
  }
}

/**
 * End doxygen group perfMemoryTest.
 * @}
 */
