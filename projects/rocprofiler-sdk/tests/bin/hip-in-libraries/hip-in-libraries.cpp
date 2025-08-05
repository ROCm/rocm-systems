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

#include "transpose.hpp"
#include "vector-ops.hpp"

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
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
size_t nqueues    = 4;
size_t nthreads   = 4;
size_t nitr       = 500;
size_t nsync      = 10;
}  // namespace

int
main(int argc, char** argv)
{
    int rank = 0;
    int size = 1;

#if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    (void) size;
#endif

    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            if(rank == 0)
            {
                fprintf(stderr,
                        "usage: hip-in-libraries [NUM_QUEUES (%zu)] [NUM_THREADS (%zu)] "
                        "[NUM_ITERATION (%zu)] "
                        "[SYNC_EVERY_N_ITERATIONS (%zu)]\n",
                        nqueues,
                        nthreads,
                        nitr,
                        nsync);
            }
            exit(EXIT_SUCCESS);
        }
    }

    if(argc > 1) nqueues = atoll(argv[1]);
    if(argc > 2) nthreads = atoll(argv[2]);
    if(argc > 3) nitr = atoll(argv[3]);
    if(argc > 4) nsync = atoll(argv[4]);

    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));

    printf("[hip-in-libraries] Number of devices found: %i\n", ndevice);
    printf("[hip-in-libraries] Number of queues: %zu\n", nqueues);
    printf("[hip-in-libraries] Number of threads: %zu\n", nthreads);
    printf("[hip-in-libraries] Number of iterations: %zu\n", nitr);
    printf("[hip-in-libraries] Syncing every %zu iterations\n", nsync);

    {
        auto vector_ops_thread = std::thread{run_vector_ops, nthreads, nqueues};
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        auto transpose_thread = std::thread{run_transpose, nthreads, nitr, nsync};

        vector_ops_thread.join();
        transpose_thread.join();
    }

    // this is a temporary workaround in omnitrace when HIP + MPI is enabled

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    for(int i = 0; i < ndevice; ++i)
    {
        HIP_API_CALL(hipSetDevice(i));
        HIP_API_CALL(hipDeviceSynchronize());
    }

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if(rank == 0)
    {
        for(int i = 0; i < ndevice; ++i)
        {
            HIP_API_CALL(hipSetDevice(i));
            HIP_API_CALL(hipDeviceReset());
        }
    }

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    return 0;
}
