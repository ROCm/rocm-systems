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

// ~RdcMetricFetcherImpl() stops the fetcher's worker thread. It stored the stop
// flag without the worker's mutex, so a worker that had checked the flag but not
// yet started to wait missed the wake-up, and the destructor waited for it
// forever. rdcd and embedded RDC destroy their fetcher when they stop.
//
// librdc cannot be loaded without a GPU (it initializes amdsmi when loaded), so
// this binary builds the metric fetcher itself. Building and destroying a
// fetcher makes no amdsmi calls; the binary links libamd_smi only for the calls
// the fetcher's other functions make.

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "rdc_lib/impl/RdcMetricFetcherImpl.h"

// Several threads build and destroy fetchers at once, which hits that window
// within seconds. If the destructor can hang again, this test hangs until the
// ctest or CI runner timeout stops it.
TEST(MetricFetcherLifetimeTest, DestroyingFetchersDoesNotHang) {
  constexpr int kThreads = 16;
  constexpr int kFetchersPerThread = 1000;

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([] {
      for (int f = 0; f < kFetchersPerThread; f++) {
        amd::rdc::RdcMetricFetcherImpl fetcher;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}
