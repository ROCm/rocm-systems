/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "trace_buffer.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>


struct TraceEntry {
  std::atomic<roctracer::TraceEntryState> valid;
};

TRACE_BUFFER_INSTANTIATE();

namespace {

std::ifstream cpuinfo("/proc/cpuinfo");
const std::size_t num_cpu_cores =
    std::count(std::istream_iterator<std::string>(cpuinfo), std::istream_iterator<std::string>(),
               std::string("processor"));

constexpr std::size_t num_iterations = 1000;
constexpr std::size_t min_num_threads = 10;
constexpr std::size_t max_num_threads = 50;

}  // namespace

int main() {
  const std::size_t num_threads = std::clamp(num_cpu_cores, min_num_threads, max_num_threads);
  std::vector<std::thread> threads(num_threads);

  std::atomic<size_t> flush_count(0);  // Count the number of times the flush callback is called.
  roctracer::TraceBuffer<TraceEntry> trace_buffer("Test", 10,
                                                  [&flush_count](auto* entry) { ++flush_count; });

  // Start the worker threads. Each thread will request 'num_iterations' entries from the
  // 'trace_buffer', then exit.
  for (auto&& thread : threads) {
    thread = std::thread([&trace_buffer]() {
      for (std::size_t j = 0; j < num_iterations; ++j) {
        auto& entry = trace_buffer.Emplace();
        entry.valid.store(roctracer::TRACE_ENTRY_COMPLETE, std::memory_order_release);
      }
    });
  }

  // Wait for all the threads to complete, then flush the trace buffer.
  for (auto&& thread : threads) thread.join();
  trace_buffer.Flush();

  std::cout << "number of records flushed = " << flush_count << std::endl;
  if (flush_count != num_iterations * threads.size()) abort();

  return EXIT_SUCCESS;
}