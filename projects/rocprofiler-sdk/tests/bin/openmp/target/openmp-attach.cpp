// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Long-running OpenMP GPU offload workload for live process-attachment tests.
// Uses a single process (stable PID) instead of respawning openmp-target in a shell loop.

#include <rocprofiler-sdk-roctx/roctx.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <thread>
#include <unistd.h>

namespace
{
volatile sig_atomic_t g_stop = 0;

void
handle_stop(int)
{
    g_stop = 1;
}

#pragma omp declare target
template <typename T>
T
mul(T a, T b)
{
    volatile T c = a * b;
    return c;
}
#pragma omp end declare target

template <typename T>
void
vmul(T* a, T* b, T* c, int N)
{
#pragma omp target map(to : a [0:N], b [0:N]) map(from : c [0:N])
#pragma omp teams distribute parallel for
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < 100000; ++j)
            c[i] = mul(a[i], b[i]);
    }
}

int
parse_positive_int(const char* text, int fallback)
{
    if(!text || !*text) return fallback;
    char*             end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if(end == text || parsed <= 0) return fallback;
    return static_cast<int>(parsed);
}

}  // namespace

int
main(int argc, char** argv)
{
    // duration_sec: wall-clock run time (default 90s, enough for attach + detach)
    // host_sleep_ms: pause on host between GPU waves (default 100ms)
    const int duration_sec  = parse_positive_int(argc > 1 ? argv[1] : nullptr, 90);
    const int host_sleep_ms = parse_positive_int(argc > 2 ? argv[2] : nullptr, 100);
    constexpr int N         = 100000;

    std::signal(SIGINT, handle_stop);
    std::signal(SIGTERM, handle_stop);

    printf(
        "openmp-attach pid=%d duration_sec=%d host_sleep_ms=%d N=%d\n",
        static_cast<int>(getpid()),
        duration_sec,
        host_sleep_ms,
        N);
    fflush(stdout);

    auto range_id = roctxRangeStart("openmp_attach_main");
    roctxMark("initialization");

    static int a_i[N];
    static int b_i[N];
    static int c_i[N];

#pragma omp parallel for
    for(int i = 0; i < N; ++i)
    {
        a_i[i] = i + 1;
        b_i[i] = i + 2;
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(duration_sec);
    int iteration = 0;

    roctxMark("gpu_loop");
    while(!g_stop && std::chrono::steady_clock::now() < deadline)
    {
        ++iteration;
        vmul(a_i, b_i, c_i, N);

        if(host_sleep_ms > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(host_sleep_ms));
    }

    roctxRangeStop(range_id);
    printf("openmp-attach finished iterations=%d\n", iteration);
    return g_stop ? 130 : 0;
}
