/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLibraryLoader.h"

namespace {

// Every C++ program already has libm loaded, so these tests never unmap the code they call.
constexpr char kLibrary[] = "libm.so.6";
constexpr int kThreads = 16;
constexpr int kLoadsPerThread = 2000;

using CosFn = double (*)(double);
using StrlenFn = size_t (*)(const char*);

// rdc_connect() from several threads shares one loader. A load must never drop the library
// that another thread is resolving symbols from.
TEST(RdcLibraryLoader, ConcurrentLoadsAlwaysResolveSymbols) {
  amd::rdc::RdcLibraryLoader loader;
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([&loader, &failures] {
      for (int i = 0; i < kLoadsPerThread; i++) {
        CosFn cos_fn = nullptr;
        if (loader.load(kLibrary) != RDC_ST_OK || loader.load_symbol(&cos_fn, "cos") != RDC_ST_OK ||
            cos_fn == nullptr) {
          failures++;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(failures.load(), 0);
}

// Loading a different library must not replace the loaded one: objects made from it may still
// be in use.
TEST(RdcLibraryLoader, LoadingAnotherLibraryKeepsTheFirst) {
  amd::rdc::RdcLibraryLoader loader;
  ASSERT_EQ(loader.load(kLibrary), RDC_ST_OK);
  EXPECT_EQ(loader.load("libc.so.6"), RDC_ST_FAIL_LOAD_MODULE);
  CosFn cos_fn = nullptr;
  EXPECT_EQ(loader.load_symbol(&cos_fn, "cos"), RDC_ST_OK);
  ASSERT_NE(cos_fn, nullptr);
  EXPECT_DOUBLE_EQ(cos_fn(0.0), 1.0);
}

// unload() forgets the library, so the loader can load another one afterwards.
TEST(RdcLibraryLoader, UnloadAllowsAnotherLibrary) {
  amd::rdc::RdcLibraryLoader loader;
  ASSERT_EQ(loader.load(kLibrary), RDC_ST_OK);
  ASSERT_EQ(loader.unload(), RDC_ST_OK);
  EXPECT_EQ(loader.load("libc.so.6"), RDC_ST_OK);
  StrlenFn strlen_fn = nullptr;
  EXPECT_EQ(loader.load_symbol(&strlen_fn, "strlen"), RDC_ST_OK);
  ASSERT_NE(strlen_fn, nullptr);
  EXPECT_EQ(strlen_fn("abc"), 3u);
}

}  // namespace
