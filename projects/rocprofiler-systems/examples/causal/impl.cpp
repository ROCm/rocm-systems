// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "causal.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/times.h>
#include <thread>
#include <unistd.h>

using mutex_t     = std::timed_mutex;
using auto_lock_t = std::unique_lock<mutex_t>;
using clock_type  = std::chrono::high_resolution_clock;
using nanosec     = std::chrono::nanoseconds;

namespace
{
inline __attribute__((always_inline)) int64_t
clock_tick() noexcept;

template <typename Precision, typename Ret = int64_t>
inline __attribute__((always_inline)) Ret
clock_tick() noexcept;

template <typename Tp = int64_t, typename Precision = std::nano>
inline __attribute__((always_inline)) Tp
get_clock_now(clockid_t clock_id) noexcept;

template <typename Tp = int64_t, typename Precision = std::nano>
inline __attribute__((always_inline)) Tp
get_clock_cpu_now() noexcept;
}  // namespace

//
//  This implementation works well for rocprof-sys
//  while COZ makes poor predictions
//
template <bool V>
bool
rng_impl_func(int64_t n, uint64_t rseed)
{
    int64_t _n    = 0;
    auto    _rng  = std::mt19937_64{ rseed };
    auto    _dist = std::uniform_int_distribution<int64_t>{ 1, 1 };
    // clang-format off
    while(_n < n) _n += _dist(_rng);
    // clang-format on
    return V;
}

template bool rng_impl_func<true>(int64_t, uint64_t);
template bool rng_impl_func<false>(int64_t, uint64_t);

//
//  This implementation works well for COZ
//  while rocprof-sys makes poor predictions
//
template <bool V>
bool
cpu_impl_func(int64_t n, int nloop)
{
    auto _t       = clock_type::now();
    auto _cpu_now = get_clock_cpu_now();
    auto _cpu_end = _cpu_now + n;
    // clang-format off
    while(get_clock_cpu_now() < _cpu_end)
    {
        for(volatile int i = 0; i < nloop; ++i) {}
        CAUSAL_PROGRESS_NAMED("cpu_impl");
    }
    // clang-format on
    return V;
}

template bool
cpu_impl_func<true>(int64_t, int);
template bool
cpu_impl_func<false>(int64_t, int);

namespace
{
int64_t
clock_tick() noexcept
{
    static int64_t _val = ::sysconf(_SC_CLK_TCK);
    return _val;
}

template <typename Precision, typename Ret>
Ret
clock_tick() noexcept
{
    return static_cast<Ret>(Precision::den) / static_cast<Ret>(clock_tick());
}

template <typename Tp, typename Precision>
Tp
get_clock_now(clockid_t clock_id) noexcept
{
    constexpr Tp    factor = Precision::den / static_cast<Tp>(std::nano::den);
    struct timespec ts;
    clock_gettime(clock_id, &ts);
    return (ts.tv_sec * std::nano::den + ts.tv_nsec) * factor;
}

template <typename Tp, typename Precision>
Tp
get_clock_cpu_now() noexcept
{
    return get_clock_now<Tp, Precision>(CLOCK_THREAD_CPUTIME_ID);
}
}  // namespace
