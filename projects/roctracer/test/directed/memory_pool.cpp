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

#include "roctracer.h"
#include "memory_pool.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>

using namespace roctracer;

namespace {

std::ifstream cpuinfo("/proc/cpuinfo");
const std::size_t num_cpu_cores =
    std::count(std::istream_iterator<std::string>(cpuinfo), std::istream_iterator<std::string>(),
               std::string("processor"));

constexpr std::size_t num_iterations = 1000;
constexpr std::size_t min_num_threads = 10;
constexpr std::size_t max_num_threads = 50;

void fatal_error(const char* message) {
  std::cerr << message << std::endl;
  abort();
}

}  // namespace

int main() {
  constexpr size_t buffer_size = 10 * sizeof(roctracer_record_t);
  constexpr size_t max_data_size = buffer_size - sizeof(roctracer_record_t);

  size_t flush_count = 0, record_count = 0;
  auto flush_callback = [&flush_count, &record_count](const char* begin, const char* end) {
    ++flush_count;
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    record_count += (end - begin) / sizeof(roctracer_record_t);
  };

  roctracer_properties_t properties{};
  properties.buffer_callback_fun = [](const char* begin, const char* end, void* arg) {
    (*static_cast<decltype(flush_callback)*>(arg))(begin, end);
  };
  properties.buffer_callback_arg = &flush_callback;
  properties.buffer_size = buffer_size;
  MemoryPool pool(properties);

  const void* original_data;
  std::atomic<int> relocation_count{0};
  auto relocate_data = [&relocation_count, &original_data](roctracer_record_t&, const void* data) {
    if (data != original_data) ++relocation_count;
  };

  // test1: the record and data fit in the buffer: no flush, data should get relocated.
  constexpr char data_fits[max_data_size] = {0};
  original_data = data_fits;
  pool.Write(roctracer_record_t{}, data_fits, sizeof(data_fits), relocate_data);  // F=0, R=1
  pool.Flush();                                                                   // F=1, R=1
  if (flush_count != 1 || relocation_count != 1) fatal_error("failed test1");

  flush_count = record_count = relocation_count = 0;

  // test2: the records and data do not fit in the buffer: 1 flush, data should get relocated.
  pool.Write(roctracer_record_t{});                                               // F=0, R=0
  pool.Write(roctracer_record_t{}, data_fits, sizeof(data_fits), relocate_data);  // F=1, R=1
  pool.Flush();                                                                   // F=2, R=1
  if (flush_count != 2 || relocation_count != 1) fatal_error("failed test2");

  flush_count = record_count = relocation_count = 0;

  // test3: data does not fit in the buffer: 1 Flush, data is not relocated, all records should be
  // processed.
  constexpr char does_not_fit[max_data_size + 1] = {0};
  original_data = does_not_fit;
  pool.Write(roctracer_record_t{}, does_not_fit, sizeof(does_not_fit), relocate_data);  // F=1, R=0
  if (flush_count != 1 || relocation_count != 0 || record_count != 1) fatal_error("failed test3");

  flush_count = record_count = relocation_count = 0;

  // test4: stress test writing and flushing.
  const std::size_t num_threads = std::clamp(num_cpu_cores, min_num_threads, max_num_threads);
  std::vector<std::thread> threads(num_threads);

  // Start the worker threads. Each thread will write 'num_iterations' records in the memory
  // pool, then exit.
  for (auto&& thread : threads) {
    thread = std::thread([&pool]() {
      for (std::size_t j = 0; j < num_iterations; ++j) pool.Write(roctracer_record_t{});
    });
  }

  // Wait for all the threads to complete, then flush the trace buffer.
  for (auto&& thread : threads) thread.join();
  pool.Flush();

  if (record_count != num_iterations * threads.size() ||
      flush_count != (record_count / (buffer_size / sizeof(roctracer_record_t))))
    fatal_error("failed test4");

  return 0;
}