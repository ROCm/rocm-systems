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

#include <hip_test_common.hh>
#include <performance_common.hh>

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

class ExtStreamGetCUMaskBenchmark : public Benchmark<ExtStreamGetCUMaskBenchmark> {
 public:
  void operator()() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::vector<uint32_t> cu_mask(props.multiProcessorCount, 0);
    hipStream_t stream{};
    HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, cu_mask.size(), cu_mask.data()));
    std::vector<uint32_t> new_cu_mask(cu_mask.size(), 0);

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipExtStreamGetCUMask(stream, new_cu_mask.size(), new_cu_mask.data()));
    }

    HIP_CHECK(hipStreamDestroy(stream));
  }
};

static void RunBenchmark() {
  ExtStreamGetCUMaskBenchmark benchmark;
  benchmark.Run();
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipExtStreamGetCUMask`.
 *  - Creates basic mask and gets it into the new one.
 * Test source
 * ------------------------
 *  - performance/stream/hipExtStreamGetCUMask.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipExtStreamGetCUMask") { RunBenchmark(); }

/**
 * End doxygen group PerformanceTest.
 * @}
 */
