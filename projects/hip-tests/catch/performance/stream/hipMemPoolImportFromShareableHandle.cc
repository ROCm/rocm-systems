/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "mem_pools_performance_common.hh"

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

class MemPoolImportFromShareableHandleBenchmark
    : public Benchmark<MemPoolImportFromShareableHandleBenchmark> {
 public:
  void operator()() {
    hipMemPool_t mem_pool{nullptr};
    hipMemPool_t mem_pool_shareable{nullptr};
#if defined(_WIN32)
    void* share_handle;
#else
    int64_t share_handle;
#endif

    hipMemPoolProps props = CreateMemPoolProps(0, kHandleType);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &props));
    HIP_CHECK(hipMemPoolExportToShareableHandle(&share_handle, mem_pool, kHandleType, 0));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemPoolImportFromShareableHandle(&mem_pool_shareable, (void*)share_handle,
                                                    kHandleType, 0));
    }

    HIP_CHECK(hipMemPoolDestroy(mem_pool_shareable));
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark() {
  MemPoolImportFromShareableHandleBenchmark benchmark;
  benchmark.Run();
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolImportFromShareableHandle`.
 *  - Uses the same process for import and export operations.
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemPoolImportFromShareableHandle") {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  RunBenchmark();
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */
