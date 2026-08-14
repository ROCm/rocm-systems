/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Microbenchmark for the production container-ID parser
// (amd::smi::ExtractContainerId).
//
// Registered as a GTest DISABLED_ case so it never runs with the default
// suite. Enable it explicitly:
//
//   ./amdsmitst --gtest_also_run_disabled_tests --gtest_filter='*Bench*'
//
// The harness uses std::chrono::steady_clock and an asm volatile memory
// barrier ("DoNotOptimize" pattern) so the compiler cannot discard the parse.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <string>

#include "container_id_test_util.h"
#include "fixtures.h"

using amdsmi_test::ExtractIdInto;

namespace {
constexpr int kBenchIterations = 10'000'000;
}  // namespace

TEST(ContainerIdParser_Bench, DISABLED_CharsetLoop) {
  const std::string line = std::string("0::/docker/") + amdsmi_test::kDocker64;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kBenchIterations; ++i) {
    char buf[AMDSMI_MAX_STRING_LENGTH];
    ExtractIdInto(line, "docker", buf, sizeof(buf));
    asm volatile("" : : "r,m"(buf) : "memory");
  }
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();
  std::printf("[BENCH] ExtractContainerId (charset + memcpy): %7.1f ns/op\n",
              static_cast<double>(ns) / kBenchIterations);
}
