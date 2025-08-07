// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using mutex_t     = std::timed_mutex;
using auto_lock_t = std::unique_lock<mutex_t>;
using clock_type  = std::chrono::high_resolution_clock;
using nanosec     = std::chrono::nanoseconds;

namespace
{
template <typename... Args>
inline void
consume_variables(Args&&...)
{}
}  // namespace

template <bool>
bool
rng_impl_func(int64_t n, uint64_t rseed);

template <bool>
bool
cpu_impl_func(int64_t n, int nloop);

void
rng_slow_func(int64_t n, uint64_t rseed) __attribute__((noinline));

void
rng_fast_func(int64_t n, uint64_t rseed) __attribute__((noinline));

void
cpu_slow_func(int64_t n, int nloop) __attribute__((noinline));

void
cpu_fast_func(int64_t n, int nloop) __attribute__((noinline));

#if USE_CPU > 0
#    define CPU_SLOW_FUNC(...) cpu_slow_func(__VA_ARGS__)
#    define CPU_FAST_FUNC(...) cpu_fast_func(__VA_ARGS__)
#else
#    define CPU_SLOW_FUNC(...) consume_variables(__VA_ARGS__)
#    define CPU_FAST_FUNC(...) consume_variables(__VA_ARGS__)
#endif

#if USE_RNG > 0
#    define RNG_SLOW_FUNC(...) rng_slow_func(__VA_ARGS__)
#    define RNG_FAST_FUNC(...) rng_fast_func(__VA_ARGS__)
#else
#    define RNG_SLOW_FUNC(...) consume_variables(__VA_ARGS__)
#    define RNG_FAST_FUNC(...) consume_variables(__VA_ARGS__)
#endif

#define SLOW_FUNC                                                                        \
    [](auto _nsec_v, auto _nseed_v, auto _nloop_v) {                                     \
        CPU_SLOW_FUNC(_nsec_v, _nloop_v);                                                \
        RNG_SLOW_FUNC(_nsec_v / 5, _nseed_v);                                            \
    }

#define FAST_FUNC                                                                        \
    [](auto _nsec_v, auto _nseed_v, auto _nloop_v) {                                     \
        CPU_FAST_FUNC(_nsec_v, _nloop_v);                                                \
        RNG_FAST_FUNC(_nsec_v / 5, _nseed_v);                                            \
    }
