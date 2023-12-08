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

#include "hip/hip_runtime.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>

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
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto   print_lock = std::mutex{};
double nruntime   = 1.0;
size_t nspin      = 500000;
size_t nthreads   = 2;
size_t nitr       = 2;
size_t nsync      = 1;

void
check_hip_error(void);
}  // namespace

__global__ void
reproducible_runtime(int64_t nspin);

void
run(int rank, int tid, hipStream_t stream);

int
main(int argc, char** argv)
{
    int rank = 0;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: reproducible-runtime [KERNEL SPIN CYCLES (%zu)] [NUM_THREADS (%zu)] "
                    "[NUM_ITERATION (%zu)] [SYNC_EVERY_N_ITERATIONS (%zu)]\n",
                    nspin,
                    nthreads,
                    nitr,
                    nsync);
            exit(EXIT_SUCCESS);
        }
    }

    if(argc > 1) nruntime = std::stod(argv[1]);
    if(argc > 2) nspin = std::stoll(argv[2]);
    if(argc > 3) nthreads = std::stoll(argv[3]);
    if(argc > 4) nitr = std::stoll(argv[4]);
    if(argc > 5) nsync = std::stoll(argv[5]);

    printf("[reproducible-runtime] Kernel spin time: %zu cycles\n", nspin);
    printf("[reproducible-runtime] Number of threads: %zu\n", nthreads);
    printf("[reproducible-runtime] Number of iterations: %zu\n", nitr);
    printf("[reproducible-runtime] Syncing every %zu iterations\n", nsync);

    // this is a temporary workaround in omnitrace when HIP + MPI is enabled
    int ndevice = 0;
    int devid   = rank;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    printf("[reproducible-runtime] Number of devices found: %i\n", ndevice);
    if(ndevice > 0)
    {
        devid = rank % ndevice;
        HIP_API_CALL(hipSetDevice(devid));
        printf("[reproducible-runtime] Rank %i assigned to device %i\n", rank, devid);
    }
    if(rank == devid && rank < ndevice)
    {
        std::vector<std::thread> _threads{};
        std::vector<hipStream_t> _streams(nthreads);
        for(size_t i = 0; i < nthreads; ++i)
            HIP_API_CALL(hipStreamCreate(&_streams.at(i)));
        for(size_t i = 1; i < nthreads; ++i)
            _threads.emplace_back(run, rank, i, _streams.at(i));
        run(rank, 0, _streams.at(0));
        for(auto& itr : _threads)
            itr.join();
        for(size_t i = 0; i < nthreads; ++i)
            HIP_API_CALL(hipStreamDestroy(_streams.at(i)));
    }
    HIP_API_CALL(hipDeviceSynchronize());
    HIP_API_CALL(hipDeviceReset());

    return 0;
}

__global__ void
reproducible_runtime(int64_t nspin_v)
{
    for(int i = 0; i < nspin_v / 64; i++)
        asm volatile("s_sleep 1");  // ~64 cycles
}

void
run(int rank, int tid, hipStream_t stream)
{
    dim3   grid(4096);
    dim3   block(64);
    double time = 0.0;
    auto   t1   = std::chrono::high_resolution_clock::now();

    do
    {
        for(size_t i = 0; i < nitr; ++i)
        {
            reproducible_runtime<<<grid, block, 0, stream>>>(nspin);
            check_hip_error();
            if(i % nsync == (nsync - 1)) HIP_API_CALL(hipStreamSynchronize(stream));
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        HIP_API_CALL(hipStreamSynchronize(stream));
        time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    } while(time < nruntime);

    {
        auto_lock_t _lk{print_lock};
        std::cout << "[" << rank << "][" << tid << "] Runtime of reproducible-runtime is " << time
                  << " sec\n"
                  << std::flush;
    }

    HIP_API_CALL(hipStreamSynchronize(stream));
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
