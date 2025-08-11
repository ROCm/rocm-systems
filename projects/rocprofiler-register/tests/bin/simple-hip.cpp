// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime_api.h>

#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#define HIP_API_CALL(CALL)                                                               \
    if(hipError_t error_ = (CALL); error_ != hipSuccess)                                 \
    {                                                                                    \
        auto _hip_api_print_lk = auto_lock_t{ print_lock };                              \
        fprintf(stderr,                                                                  \
                "%s:%d :: HIP error : %s\n",                                             \
                __FILE__,                                                                \
                __LINE__,                                                                \
                hipGetErrorString(error_));                                              \
        fflush(stderr);                                                                  \
        std::exit(EXIT_FAILURE);                                                         \
    }

static constexpr int64_t Bi  = 1;
static constexpr int64_t KiB = 1024 * Bi;
static constexpr int64_t MiB = 1024 * KiB;

void
run(size_t rank, size_t tid, const char* prefix);

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto   print_lock = std::mutex{};
size_t xdim       = 4960 * 2;
size_t ydim       = 4960 * 2;
size_t nthreads   = 2;
size_t nitr       = 1;
}  // namespace

int
main(int argc, char** argv)
{
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{ argv[i] };
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: %s [M (%zu)] [N (%zu)] [NUM_THREADS (%zu)] [NUM_ITERATION "
                    "(%zu)]\n",
                    ::basename(argv[0]),
                    xdim,
                    ydim,
                    nthreads,
                    nitr);
            exit(EXIT_SUCCESS);
        }
    }

    if(argc > 1) xdim = atoll(argv[1]);
    if(argc > 2) ydim = atoll(argv[2]);
    if(argc > 3) nthreads = atoll(argv[3]);
    if(argc > 4) nitr = atoll(argv[4]);

    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));

    printf("[%s] Number of devices found: %i\n", ::basename(argv[0]), ndevice);
    printf("[%s] Number of matrix: %zu x %zu\n", ::basename(argv[0]), xdim, ydim);
    printf("[%s] Number of threads: %zu\n", ::basename(argv[0]), nthreads);
    printf("[%s] Number of iterations: %zu\n", ::basename(argv[0]), nitr);

    for(size_t j = 0; j < nitr; ++j)
    {
        printf("\n[%s] Iteration #%zu\n\n", ::basename(argv[0]), j);

        auto threads = std::vector<std::thread>{};
        threads.reserve(nthreads);

        for(size_t i = 0; i < nthreads; ++i)
            threads.emplace_back(run, getpid(), i, ::basename(argv[0]));

        for(auto& itr : threads)
            itr.join();
    }

    return 0;
}

template <typename Tp>
auto
allocate_memory(size_t M, size_t N, Tp val)
{
    const auto sz = M * N;
    const auto nb = sz * sizeof(Tp);

    Tp* ptr = new Tp[M * N];
    ::memset(ptr, val, nb);

    HIP_API_CALL(hipHostRegister(ptr, nb, hipHostRegisterDefault))

    return ptr;
}

template <typename Tp>
auto
deallocate_memory(Tp* ptr)
{
    HIP_API_CALL(hipHostUnregister(ptr))
    delete[] ptr;
}

void
run(size_t rank, size_t tid, const char* label)
{
    const auto M = xdim;
    const auto N = ydim;

    auto* inp = allocate_memory<int>(M, N, 0);
    auto* out = allocate_memory<int>(M, N, 1);

    auto _engine =
        std::default_random_engine{ std::random_device{}() * (rank + 1) * (tid + 1) };
    std::uniform_int_distribution<int> _dist{ 0, 1000 };

    for(size_t i = 0; i < (M * N); i++)
        inp[i] = _dist(_engine);

    // lock during malloc to get more accurate memory info
    {
        auto_lock_t _lk{ print_lock };
        size_t      free_gpu_mem  = 0;
        size_t      total_gpu_mem = 0;

        HIP_API_CALL(hipMemGetInfo(&free_gpu_mem, &total_gpu_mem));
        free_gpu_mem /= MiB;
        total_gpu_mem /= MiB;

        std::cout << "[" << label << "][" << rank << "][" << tid
                  << "] Available GPU memory (MiB): " << std::setw(6) << free_gpu_mem
                  << " / " << std::setw(6) << total_gpu_mem << std::endl;
    }

    deallocate_memory(inp);
    deallocate_memory(out);
}
