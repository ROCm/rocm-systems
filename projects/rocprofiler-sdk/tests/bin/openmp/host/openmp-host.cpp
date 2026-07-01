// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Host-only OpenMP workload used to exercise the CPU-side OMPT callbacks that
// rocprofv3 --ompt-trace exposes (parallel regions, worksharing, explicit
// tasks, and synchronization). It deliberately contains *no* target-offload
// regions so that it can run on any machine with an OMPT-capable OpenMP
// runtime, regardless of whether a GPU/target-offload toolchain is present.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

// Host-side worksharing + reduction. Drives:
//   omp_parallel_begin / omp_parallel_end, omp_implicit_task, omp_work,
//   omp_sync_region (implicit barrier) and omp_sync_region_wait.
static int64_t
parallel_reduction(const std::vector<int>& data)
{
    int64_t total = 0;
#pragma omp parallel for reduction(+ : total)
    for(std::size_t i = 0; i < data.size(); ++i)
    {
        total += data[i];
    }
    return total;
}

// Explicit tasking + taskwait. Drives:
//   omp_task_create, omp_task_schedule and an omp_sync_region (taskwait).
static int64_t
task_sum(const std::vector<int>& data)
{
    int64_t total = 0;
#pragma omp parallel
    {
#pragma omp single
        {
            const std::size_t chunks = 4;
            const std::size_t span   = (data.size() + chunks - 1) / chunks;
            for(std::size_t c = 0; c < chunks; ++c)
            {
                const std::size_t begin = c * span;
                const std::size_t end   = std::min(begin + span, data.size());
#pragma omp task firstprivate(begin, end) shared(total)
                {
                    int64_t local = 0;
                    for(std::size_t i = begin; i < end; ++i)
                        local += data[i];
// critical section -> omp_mutex_acquire / omp_mutex_acquired / omp_mutex_released
#pragma omp critical
                    total += local;
                }
            }
#pragma omp taskwait
        }
    }
    return total;
}

int
main(int argc, char** argv)
{
    const char* exe_name = argv[0];
    uint64_t    nitr     = 4;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string_view{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr, "usage: %s [NUM_ITERATION (%lu)]\n", exe_name, nitr);
            return 0;
        }
    }
    if(argc > 1) nitr = std::stoul(argv[1]);

    constexpr int    N = 100000;
    std::vector<int> data(N);
    for(int i = 0; i < N; ++i)
        data[i] = i + 1;

    const int64_t expected = static_cast<int64_t>(N) * (N + 1) / 2;

    int n_errors = 0;
    for(uint64_t it = 0; it < nitr; ++it)
    {
        if(parallel_reduction(data) != expected) ++n_errors;
        if(task_sum(data) != expected) ++n_errors;
    }

    if(n_errors == 0)
    {
        printf("Success\n");
        return 0;
    }

    printf("Total %d failures\n", n_errors);
    printf("Fail\n");
    return 1;
}
