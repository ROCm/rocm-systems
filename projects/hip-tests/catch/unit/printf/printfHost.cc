/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <thread>
#include <vector>


// Kernel Function
__global__ void run_printf(int* count) { *count = printf("Hello World"); }

// Uses hostcall printf, so its stream gets a hostcall buffer and the listener, but prints
// nothing: the test passes print == 0.
__global__ void maybe_printf(int print) {
  if (print) printf("unexpected\n");
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
 * - Test case to verify the printf return value(Number of Characters)for -mprintf-kind=hostcall
 * compiler option Test source
 * ------------------------
 * - catch/unit/printf/printfHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.7
 */
HIP_TEST_CASE(Unit_Host_Printf) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPcieAtomicUnsupported);
  }
  int *count{nullptr}, *count_d{nullptr};
  count = reinterpret_cast<int*>(malloc(sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, sizeof(int)));

  hipLaunchKernelGGL(run_printf, dim3(1), dim3(1), 0, 0, count_d);

  HIP_CHECK(hipMemcpy(count, count_d, sizeof(int), hipMemcpyDeviceToHost));

  std::string str = "Hello World";
  int length = str.length();
#if HT_AMD
  REQUIRE(length == *count);
#else
  REQUIRE(*count == 0);
#endif
  free(count);
  HIP_CHECK(hipFree(count_d));
}

/**
 * Test Description
 * ------------------------
 * - Several threads each create a stream, run a kernel that uses hostcall printf, and destroy
 *   the stream, many times. Destroying the last stream with a hostcall buffer shuts the hostcall
 *   listener down while other threads create the next one, so the listener is stopped and
 *   started concurrently. Checks that this neither crashes nor hangs (the listener used to be
 *   deleted twice when two streams were destroyed at once), and that printf on a new stream
 *   afterwards still returns the printed length, i.e. the restarted listener services it.
 * Test source
 * ------------------------
 * - catch/unit/printf/printfHost.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_Host_Printf_ConcurrentStreamTeardown) {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPcieAtomicUnsupported);
  }
  constexpr int kThreads = 8;
  constexpr int kIterations = 50;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([] {
      for (int i = 0; i < kIterations; ++i) {
        hipStream_t stream;
        HIP_CHECK_THREAD(hipStreamCreate(&stream));
        hipLaunchKernelGGL(maybe_printf, dim3(1), dim3(1), 0, stream, 0);
        HIP_CHECK_THREAD(hipGetLastError());
        HIP_CHECK_THREAD(hipStreamSynchronize(stream));
        HIP_CHECK_THREAD(hipStreamDestroy(stream));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  HIP_CHECK_THREAD_FINALIZE();

  // A listener started after all that churn still services printf
  int *count{nullptr}, *count_d{nullptr};
  count = reinterpret_cast<int*>(malloc(sizeof(int)));
  HIP_CHECK(hipMalloc(&count_d, sizeof(int)));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipLaunchKernelGGL(run_printf, dim3(1), dim3(1), 0, stream, count_d);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(count, count_d, sizeof(int), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
#if HT_AMD
  REQUIRE(*count == static_cast<int>(std::string("Hello World").length()));
#endif
  free(count);
  HIP_CHECK(hipFree(count_d));
}

/**
 * End doxygen group PrintfTest.
 * @}
 */
