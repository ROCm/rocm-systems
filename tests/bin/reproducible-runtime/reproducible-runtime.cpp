// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime.h"
#include "rocprofiler-sdk-roctx/roctx.h"

#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

#if defined(USE_MPI)
#    include <mpi.h>
#endif

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            auto _hip_api_print_lk = auto_lock_t{print_lock};                                      \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error %i : %s\n",                                                \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    static_cast<int>(error_),                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

namespace
{
using auto_lock_t   = std::unique_lock<std::mutex>;
auto     print_lock = std::mutex{};
double   nruntime   = 500.0;  // ms
uint32_t nspin      = 128 * 10000;
size_t   nthreads   = 2;

void
check_hip_error(void);
}  // namespace

__global__ void
reproducible_runtime(uint32_t nspin);

void
run(int tid, int devid);

int
main(int argc, char** argv)
{
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: reproducible-runtime [KERNEL RUNTIME PER THREAD (default: %f msec)] "
                    "[NUM_THREADS (default: %zu)] [SPIN CYCLES PER KERNEL LAUNCH (default: %u)]\n",
                    nruntime,
                    nthreads,
                    nspin);
            exit(EXIT_SUCCESS);
        }
    }

    if(argc > 1) nruntime = std::stod(argv[1]);
    if(argc > 2) nthreads = std::stoll(argv[2]);
    if(argc > 3) nspin = std::stoll(argv[3]);

    printf("[reproducible-runtime] Kernel runtime per thread: %.3f msec\n", nruntime);
    printf("[reproducible-runtime] Spin time per kernel: %u cycles\n", nspin);
    printf("[reproducible-runtime] Number of threads: %zu\n", nthreads);

    // this is a temporary workaround in omnitrace when HIP + MPI is enabled
    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    printf("[reproducible-runtime] Number of devices found: %i\n", ndevice);
    auto _threads = std::vector<std::thread>{};
    for(size_t i = 0; i < nthreads; ++i)
        _threads.emplace_back(run, i, i % ndevice);
    for(auto& itr : _threads)
        itr.join();
    HIP_API_CALL(hipDeviceSynchronize());
    HIP_API_CALL(hipDeviceReset());

    return 0;
}

__global__ void
reproducible_runtime(uint32_t nspin_v)
{
    for(uint32_t i = 0; i < nspin_v / 64; i++)
        asm volatile("s_sleep 1");
    if(nspin_v > 64)
        for(uint32_t i = 0; i < nspin_v % 64; i++)
            asm volatile("s_sleep 1");
}

void
run(int tid, int devid)
{
    auto roctx_range_id = roctxRangeStart("run");

    constexpr int min_avail_simd = 128;
    dim3          grid(min_avail_simd);
    dim3          block(32);
    double        time    = 0.0;
    hipStream_t   stream  = {};
    hipEvent_t    start   = {};
    hipEvent_t    stop    = {};
    uint64_t      nlaunch = 0;

    HIP_API_CALL(hipSetDevice(devid));
    HIP_API_CALL(hipStreamCreate(&stream));
    HIP_API_CALL(hipEventCreate(&start));
    HIP_API_CALL(hipEventCreate(&stop));

    do
    {
        roctxMark("iteration");
        uint32_t cyclesleft = 1000 * 1000 * (nruntime - static_cast<double>(time));
        HIP_API_CALL(hipEventRecord(start, stream));
        reproducible_runtime<<<grid, block, 0, stream>>>(std::min<uint32_t>(nspin, cyclesleft));
        HIP_API_CALL(hipEventRecord(stop, stream));
        check_hip_error();
        HIP_API_CALL(hipEventSynchronize(stop));
        float elapsed = 0.0f;
        HIP_API_CALL(hipEventElapsedTime(&elapsed, start, stop));
        time += static_cast<double>(elapsed);
        ++nlaunch;
    } while(time < nruntime);

    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipEventDestroy(start));
    HIP_API_CALL(hipEventDestroy(stop));

    {
        auto _msg = std::stringstream{};
        _msg << '[' << getpid() << "][" << tid << "] Runtime of reproducible-runtime is "
             << std::setprecision(2) << std::fixed << time << " ms (" << std::setprecision(3)
             << (time / 1000.0f) << " sec). Kernels dispatched: " << nlaunch << "\n";
        auto_lock_t _lk{print_lock};
        std::cout << _msg.str() << std::flush;
    }

    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipStreamDestroy(stream));

    roctxRangeStop(roctx_range_id);

    constexpr auto scale = 1.1;
    if(time > scale * nruntime)
    {
        auto _msg = std::stringstream{};
        _msg << "total kernel runtime exceeded (" << scale << " * " << nruntime << " = "
             << (scale * nruntime) << ") :: " << time << " ms";
        throw std::runtime_error{_msg.str()};
    }
}

namespace
{
void
check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        auto_lock_t _lk{print_lock};
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hip_api_call");
    }
}
}  // namespace
