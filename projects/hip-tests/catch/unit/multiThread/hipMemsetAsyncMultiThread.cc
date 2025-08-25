/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

enum TestType { SameStream = 0, DifferentStreams };

// Func -> Memset/MemsetD[8/16/32], T - type of data to be worked on
// basically each thread memsets value present in input one by one on its
// allocated buffer
template <typename Func, typename T> void threadCall(Func f, hipStream_t stream) {
  // Should match hipMemsetAsync or hipMemsetD*Async arguments
  static_assert((std::is_same<Func, hipError_t (*)(void*, int, size_t,
                                                   hipStream_t)>::value ||  // hipMemsetAsync
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, int, size_t,
                                                   hipStream_t)>::value ||  // hipMemsetD32Async
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, unsigned short, size_t,
                                                   hipStream_t)>::value ||  // hipMemsetD16Async
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, unsigned char, size_t,
                                                   hipStream_t)>::value) &&  // hipMemsetD8Async
                "Func f should be hipMemsetAsync or hipMemsetD*Async");

  constexpr bool cast_2_void =
      std::is_same<Func, hipError_t (*)(void*, int, size_t, hipStream_t)>::value;

  // Use the unsiged type, since memset concerns with set bit values over a mem
  // address
  typedef typename std::make_unsigned<T>::type unsigned_t;

  unsigned_t min = 0;
  unsigned_t max = std::numeric_limits<unsigned_t>::max();

  std::mt19937_64 engine(std::random_device{}());
  auto distribution = std::uniform_int_distribution<unsigned int>(
      min, max);  // this needs to be unsigned because windows does not treats
                  // char as numeric types

  T* ptr{nullptr};
  constexpr size_t size = 1024;
  constexpr size_t iter = 1024;
  HIP_CHECK_THREAD(hipMalloc(&ptr, sizeof(T) * size));
  hipEvent_t event{};
  HIP_CHECK_THREAD(hipEventCreate(&event));

  union overlay_val_t {
    T t_val;
    unsigned_t u_val;
  } overlay_val;

  std::vector<T> dst(size, 0);
  for (size_t i = 0; i < iter; i++) {
    overlay_val.u_val =
        static_cast<unsigned_t>(distribution(engine));  // generate an unsigned int number
    if constexpr (cast_2_void) {
      HIP_CHECK_THREAD(f((void*)ptr, overlay_val.t_val, size, stream));
    } else {
      HIP_CHECK_THREAD(f(*(hipDeviceptr_t*)&ptr, overlay_val.t_val, size, stream));
    }
    HIP_CHECK_THREAD(
        hipMemcpyAsync(dst.data(), ptr, size * sizeof(T), hipMemcpyDeviceToHost, stream));
    HIP_CHECK_THREAD(hipEventRecord(event, stream));
    HIP_CHECK_THREAD(hipStreamWaitEvent(stream, event, 0));  // wait till memcpy is done
    REQUIRE_THREAD(std::all_of(dst.begin(), dst.end(), [&](T v) {
      // If this test ever fails, add prints here on mismatch
      return v == overlay_val.t_val;
    }));
  }

  HIP_CHECK_THREAD(hipEventDestroy(event));
  HIP_CHECK_THREAD(hipFree(ptr));
}

// Func -> Memset/MemsetD[8/16/32], T - type of data to be worked on
template <typename Func, typename T> void launchThreads(Func f, TestType type) {
  static_assert(!std::is_pointer<T>::value && "Argument cant be a pointer");

  // Should match hipMemsetAsync or hipMemsetD*Async arguments
  static_assert((std::is_same<Func, hipError_t (*)(void*, int, size_t,
                                                   hipStream_t)>::value ||  // hipMemsetAsync
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, int, size_t,
                                                   hipStream_t)>::value ||  // hipMemsetD32Async
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, unsigned short, size_t,
                                                   hipStream_t)>::value ||  // hipMemsetD16Async
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, unsigned char, size_t,
                                                   hipStream_t)>::value) &&  // hipMemsetD8Async
                "Func f should be hipMemsetAsync or hipMemsetD*Async");

  const size_t num_threads = (std::thread::hardware_concurrency() > 8)
                                 ? (((std::thread::hardware_concurrency() / 4) >= 127)
                                        ? 127
                                        : (std::thread::hardware_concurrency() / 4))
                                 : 2;  // thread count between 2 - 127

  const size_t num_streams = (type == SameStream) ? 1 : num_threads;
  std::vector<hipStream_t> streams(num_streams, nullptr);

  for (size_t i = 0; i < num_streams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
    REQUIRE(streams[i] != nullptr);
  }

  std::vector<std::thread> thread_pool;
  thread_pool.reserve(num_threads);
  auto thread_func = threadCall<Func, T>;
  for (size_t i = 0; i < num_threads; i++) {
    auto stream = (type == SameStream) ? streams[0] : streams[i];
    thread_pool.emplace_back(std::thread(thread_func, f, stream));
  }

  for (size_t i = 0; i < num_threads; i++) {
    thread_pool[i].join();
  }

  HIP_CHECK_THREAD_FINALIZE();  // Make sure all thread have exited properly

  for (size_t i = 0; i < num_streams; i++) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

TEST_CASE("Unit_hipMemsetAsync_QueueJobsMultithreaded") {
  using hipMemsetAsync_t = hipError_t (*)(void*, int, const size_t, hipStream_t);
  using hipMemsetAsyncD8_t =
      hipError_t (*)(hipDeviceptr_t, unsigned char, const size_t, hipStream_t);
  using hipMemsetAsyncD16_t =
      hipError_t (*)(hipDeviceptr_t, unsigned short, const size_t, hipStream_t);
  using hipMemsetAsyncD32_t = hipError_t (*)(hipDeviceptr_t, int, const size_t, hipStream_t);

  launchThreads<hipMemsetAsync_t, char>(hipMemsetAsync, SameStream);
  launchThreads<hipMemsetAsync_t, char>(hipMemsetAsync, DifferentStreams);

  launchThreads<hipMemsetAsyncD8_t, unsigned char>(hipMemsetD8Async, SameStream);
  launchThreads<hipMemsetAsyncD8_t, unsigned char>(hipMemsetD8Async, DifferentStreams);
  launchThreads<hipMemsetAsyncD16_t, unsigned short>(hipMemsetD16Async, SameStream);
  launchThreads<hipMemsetAsyncD16_t, unsigned short>(hipMemsetD16Async, DifferentStreams);
  launchThreads<hipMemsetAsyncD32_t, int>(hipMemsetD32Async, SameStream);
  launchThreads<hipMemsetAsyncD32_t, int>(hipMemsetD32Async, DifferentStreams);
}
