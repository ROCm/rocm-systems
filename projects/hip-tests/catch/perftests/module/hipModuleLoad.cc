/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

/**
 * @addtogroup hipModuleLoad hipModuleLoad
 * @{
 * @ingroup PerformanceTest
 * `hipError_t hipModuleLoad(hipModule_t *module, const char *fname)` -
 * Loads code object from file into a module the currrent context.
 */

/**
 * Test Description
 * ------------------------
 * - It will verify performance test of hipModuleLoad api
 * Test source
 * ------------------------
 * - perftests/module/hipModuleLoad.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */

TEST_CASE("Perf_hipModuleLoad_test") {
  const std::string CO_FILE_NAME = "saxpy.code";
  constexpr uint32_t WARMUP_ITERATIONS = 100;
  constexpr uint32_t NUM_ITERATIONS = 1000;
  std::vector<hipModule_t> modules{NUM_ITERATIONS};
  std::vector<hipModule_t> warmup_modules{WARMUP_ITERATIONS};

  // warmup
  for (uint32_t i = 0; i < WARMUP_ITERATIONS; ++i) {
    HIP_CHECK(hipModuleLoad(&warmup_modules[i], CO_FILE_NAME.c_str()));
  }
  for (uint32_t i = 0; i < WARMUP_ITERATIONS; ++i) {
    HIP_CHECK(hipModuleUnload(warmup_modules[i]));
  }

  // benchmark
  auto start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
    HIP_CHECK(hipModuleLoad(&modules[i], CO_FILE_NAME.c_str()));
  }
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  double avg_load_time = duration * 1.0 / NUM_ITERATIONS;

  for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
    HIP_CHECK(hipModuleUnload(modules[i]));
  }
  printf("Num iterations: %d, Avg load time: %f us\n", NUM_ITERATIONS,
         avg_load_time);
}

/**
* End doxygen group PerformanceTest.
* @}
*/
