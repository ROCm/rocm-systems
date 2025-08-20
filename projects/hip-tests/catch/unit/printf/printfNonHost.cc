/*
   Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */
#include <hip_test_common.hh>

#define KERNEL_ITERATIONS 15
#define BLOCK_SIZE 2
#define THREADS_PER_BLOCK 1024
#define ITER_COUNT_FOR_THREAD (BLOCK_SIZE * THREADS_PER_BLOCK)
#define CONST_STR "Hello World from Device.Iam printing 55 bytes of data.\n"

// info_.printfBufferSize_ = PrintfDbg::WorkitemDebugSize(4096) * info().maxWorkGroupSize_(1024)
#define MAX_BUFF_SIZE (4096 * 1024)
//<control DWord (4 bytes)><format string hash (8 bytes)><printf arguments each aligned to 8 bytes>
#define Item_BUFF_SIZE (4 + 8 + 56) /* 56 = 55 + 1 byte null terminator */
#define VALID_COUNT (MAX_BUFF_SIZE / Item_BUFF_SIZE)
#define ITER_COUNT (VALID_COUNT + 1)

// Kernel Functions
__global__ void run_printf_basic(int* count) { *count = printf("Hello World\n"); }

__global__ void kernel_printf_loop(uint iterCount, int* count) {
  for (uint i = 0; i < iterCount; i++) {
    count[i] = printf("%s", CONST_STR);
  }
}

__global__ void kernel_printf_thread(int* count) {
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  count[tid] = printf("%s", CONST_STR);
}

/**
 * @addtogroup printf printf
 * @{
 * @ingroup PrintfTest
 * `int printf()` -
 * Method to print the content on output device.
 */
/**
 * Test Description
 * ------------------------
 * - Test case to verify the printf return value for -mprintf-kind=buffered compiler option
 * - printf should return 0 for normal buffer.
 * Test source
 * ------------------------
 * - catch/unit/printf/printfNonHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */

TEST_CASE("Unit_NonHost_Printf_basic") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  int *count{nullptr}, *count_d{nullptr};

  count = reinterpret_cast<int*>(malloc(sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, sizeof(int)));

  hipLaunchKernelGGL(run_printf_basic, dim3(1), dim3(1), 0, 0, count_d);
  HIP_CHECK(hipMemcpy(count, count_d, sizeof(int), hipMemcpyDeviceToHost));

  REQUIRE(*count == 0);

  free(count);
  HIP_CHECK(hipFree(count_d));
}

/**
 * Test Description
 * ------------------------
 * - Test case to verify the printf return value for big buffer for -mprintf-kind=buffered compiler
 * option
 * - Call the printf API for number of iterations in the Kernel Function. Printf should return -1
 * for overflow buffer. Test source
 * ------------------------
 * - catch/unit/printf/printfNonHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */

TEST_CASE("Unit_NonHost_Printf_loop") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  int *count{nullptr}, *count_d{nullptr};
  count = reinterpret_cast<int*>(malloc(ITER_COUNT * sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, ITER_COUNT * sizeof(int)));

  hipLaunchKernelGGL(kernel_printf_loop, dim3(1), dim3(1), 0, 0, ITER_COUNT, count_d);

  HIP_CHECK(hipMemcpy(count, count_d, ITER_COUNT * sizeof(int), hipMemcpyDeviceToHost));
  int test = 0;
  for (int i = 0; i < ITER_COUNT; i++) {
    if (count[i] == -1) {
      test = i;
    }
  }
  if (test == (ITER_COUNT - 1)) {
    REQUIRE(true);
  } else {
    REQUIRE(false);
  }
  free(count);
  HIP_CHECK(hipFree(count_d));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the printf return value for big buffer for -mprintf-kind=buffered compiler
 * option
 * - Call the single printf API for multiple threads. Printf should return -1 for overflow buffer.
 * Test source
 * ------------------------
 * - catch/unit/printf/printfNonHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */

TEST_CASE("Unit_NonHost_Printf_multiple_Threads") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  int *count{nullptr}, *count_d{nullptr};
  fprintf(stderr, "VALID_COUNT=%d, ITER_COUNT_FOR_THREAD=%d\n", VALID_COUNT, ITER_COUNT_FOR_THREAD);
  count = reinterpret_cast<int*>(malloc(ITER_COUNT_FOR_THREAD * sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, ITER_COUNT_FOR_THREAD * sizeof(int)));

  hipLaunchKernelGGL(kernel_printf_thread, dim3(BLOCK_SIZE), dim3(THREADS_PER_BLOCK), 0, 0,
                     count_d);

  HIP_CHECK(hipMemcpy(count, count_d, ITER_COUNT_FOR_THREAD * sizeof(int), hipMemcpyDeviceToHost));

  int check = 0;
  for (int i = 0; i < ITER_COUNT_FOR_THREAD; i++) {
    if (count[i] == -1) {
      check = check + 1;
    }
  }
  if (VALID_COUNT < ITER_COUNT_FOR_THREAD) {
    REQUIRE(check == ITER_COUNT_FOR_THREAD - VALID_COUNT);
  } else {
    REQUIRE(check == 0);
  }
  free(count);
  HIP_CHECK(hipFree(count_d));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify buffer availabilty after using it for the maximum limit.
 * - Call the kernel which will use the maximum printf buffer. Verify the printf returned values.
 * - Repeat this for few times to check the buffer is available after the maximum usage of it.
 * Test source
 * ------------------------
 * - catch/unit/printf/printfNonHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */

TEST_CASE("Unit_NonHost_Printf_BufferAvailability") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  int *count{nullptr}, *count_d{nullptr};

  count = reinterpret_cast<int*>(malloc((ITER_COUNT - 1) * sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, (ITER_COUNT - 1) * sizeof(int)));
  int check = 0;
  for (int i = 0; i < KERNEL_ITERATIONS; i++) {
    hipLaunchKernelGGL(kernel_printf_loop, dim3(1), dim3(1), 0, 0, ITER_COUNT - 1, count_d);

    HIP_CHECK(hipMemcpy(count, count_d, (ITER_COUNT - 1) * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    int test = 0;
    for (int i = 0; i < ITER_COUNT - 1; i++) {
      if (count[i] == 0) {
        test = test + 1;
      }
    }
    if (test == (ITER_COUNT - 1)) {
      check = check + 1;
    }
  }
  if (check == KERNEL_ITERATIONS) {
    REQUIRE(true);
  } else {
    REQUIRE(false);
  }

  free(count);
  HIP_CHECK(hipFree(count_d));
}


/**
 * End doxygen group PrintfTest.
 * @}
 */
