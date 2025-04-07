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

#include <roctracer_hip.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>

// Create as many threads as there are cores, half changing the hipSetDevice roctracer API callback
// and argument, and the other half calling hipSetDevice, all running concurrently. If there is a
// race when setting the API callback and argument, the test aborts.

constexpr int N_ITER = 1000000;

namespace {

std::ifstream cpuinfo("/proc/cpuinfo");
const int num_cpu_cores =
    std::count(std::istream_iterator<std::string>(cpuinfo), std::istream_iterator<std::string>(),
               std::string("processor"));

template <std::size_t N> void callback(uint32_t, uint32_t, const void*, void* arg) {
  // The callback argument must match the callback function.
  if (arg != callback<N>) abort();
}

template <std::size_t... Is> constexpr auto create_callbacks(std::index_sequence<Is...>) {
  return std::array{&callback<Is>...};
}

template <std::size_t N> constexpr auto create_callbacks() {
  return create_callbacks(std::make_index_sequence<N>{});
}

constexpr auto callbacks = create_callbacks<128>();

}  // namespace

int main() {
  if (hipSetDevice(0) != hipSuccess) abort();

  std::vector<std::thread> threads;
  for (int i = 0; i < std::max(2, num_cpu_cores / 2); ++i) {
    threads.emplace_back(
        [](auto callback) {
          for (int n = 0; n < N_ITER; ++n)
            roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipSetDevice, callback,
                                         reinterpret_cast<void*>(callback));
        },
        callbacks[i % callbacks.size()]);
    threads.emplace_back([]() {
      for (int n = 0; n < N_ITER; ++n)
        if (hipSetDevice(0) != hipSuccess) abort();
    });
  }
  for (auto&& thread : threads) thread.join();

  roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
  return 0;
}