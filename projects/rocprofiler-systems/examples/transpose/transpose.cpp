/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"

#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

static std::mutex print_lock{};
using auto_lock_t = std::unique_lock<std::mutex>;

#define HIP_API_CALL(CALL)                                                               \
    {                                                                                    \
        hipError_t error_ = (CALL);                                                      \
        if(error_ != hipSuccess)                                                         \
        {                                                                                \
            auto_lock_t _lk{ print_lock };                                               \
            fprintf(stderr, "%s:%d :: HIP error : %s\n", __FILE__, __LINE__,             \
                    hipGetErrorString(error_));                                          \
            throw std::runtime_error("hip_api_call");                                    \
        }                                                                                \
    }

void
check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        auto_lock_t _lk{ print_lock };
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hip_api_call");
    }
}

void
verify(int* in, int* out, int M, int N)
{
    for(int i = 0; i < 10; i++)
    {
        int row = rand() % M;
        int col = rand() % N;
        if(in[row * N + col] != out[col * M + row])
        {
            auto_lock_t _lk{ print_lock };
            std::cout << "mismatch: " << row << ", " << col << " : " << in[row * N + col]
                      << " | " << out[col * M + row] << "\n";
        }
    }
}

const unsigned TILE_DIM = 32;
__global__ void
transpose_a(int* in, int* out, int M, int N)
{
    __shared__ int tile[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * TILE_DIM + tx;
    int y = blockIdx.y * TILE_DIM + ty;

    int v = 0;
    if(x < M && y < N) v = in[y * M + x];  // guarded load
    tile[ty][tx] = v;
    __syncthreads();

    int xt = blockIdx.y * TILE_DIM + tx;
    int yt = blockIdx.x * TILE_DIM + ty;
    if(xt < N && yt < M) out[yt * N + xt] = tile[tx][ty];  // guarded store
}

namespace
{
constexpr size_t k_default_nitr  = 500;
constexpr size_t k_default_nsync = 10;
constexpr float  k_bytes_per_gb  = 1073741824.0F;

size_t            g_nthreads = 2;
size_t            g_nitr  = k_default_nitr;  // 0 means run until signaled (infinite mode)
size_t            g_nsync = k_default_nsync;
std::atomic<bool> g_stop_requested{ false };

void
handle_stop_signal(int /*signum*/) noexcept
{
    g_stop_requested.store(true, std::memory_order_relaxed);
}
}  // namespace

void
run(int rank, int tid, int devid, hipStream_t stream, int argc, char** argv)
{
    // HIP tracks the active device per host thread, so each worker thread must
    // select the same device its stream was created on. Otherwise kernel
    // launches fail with "invalid resource handle" whenever a rank is mapped to
    // a non-zero device (e.g. rank 1 -> device 1 under MPI).
    HIP_API_CALL(hipSetDevice(devid));

    unsigned int M = 4960 * 2;
    unsigned int N = 4960 * 2;

    auto_lock_t _lk{ print_lock };
    std::cout << "[" << rank << "][" << tid << "] M: " << M << " N: " << N << std::endl;
    _lk.unlock();

    std::default_random_engine _engine{ std::random_device{}() * (rank + 1) * (tid + 1) };
    std::uniform_int_distribution<int> _dist{ 0, 1000 };

    const size_t elems      = static_cast<size_t>(M) * static_cast<size_t>(N);
    const size_t size       = elems * sizeof(int);
    int*         inp_matrix = new int[elems];
    int*         out_matrix = new int[elems];

    for(size_t i = 0; i < elems; i++)
    {
        inp_matrix[i] = _dist(_engine);
        out_matrix[i] = 0;
    }
    int* in  = nullptr;
    int* out = nullptr;

    HIP_API_CALL(hipMalloc(&in, size));
    HIP_API_CALL(hipMalloc(&out, size));
    HIP_API_CALL(hipMemsetAsync(in, 0, size, stream));
    HIP_API_CALL(hipMemsetAsync(out, 0, size, stream));
    HIP_API_CALL(hipMemcpyAsync(in, inp_matrix, size, hipMemcpyHostToDevice, stream));
    HIP_API_CALL(hipStreamSynchronize(stream));

    dim3 grid(M / 32, N / 32, 1);
    dim3 block(32, 32, 1);  // transpose_a

    size_t     completed_itr = 0;
    const auto start_time    = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < g_nitr; ++i)
    {
        if(g_stop_requested.load(std::memory_order_relaxed))
        {
            break;
        }
        transpose_a<<<grid, block, 0, stream>>>(in, out, M, N);
        check_hip_error();
        if(i % g_nsync == (g_nsync - 1))
        {
            HIP_API_CALL(hipStreamSynchronize(stream));
        }
        ++completed_itr;
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipMemcpyAsync(out_matrix, out, size, hipMemcpyDeviceToHost, stream));
    double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
            .count();
    const float gigabytes = static_cast<float>(size) * static_cast<float>(completed_itr) *
                            2.0F / k_bytes_per_gb;

    print_lock.lock();
    std::cout << "[" << rank << "][" << tid << "] Runtime of transpose is " << time
              << " sec\n"
              << "The average performance of transpose is " << gigabytes / time
              << " GBytes/sec" << std::endl;
    print_lock.unlock();

    HIP_API_CALL(hipStreamSynchronize(stream));

    // cpu_transpose(matrix, out_matrix, M, N);
    verify(inp_matrix, out_matrix, M, N);

    HIP_API_CALL(hipFree(in));
    HIP_API_CALL(hipFree(out));

    delete[] inp_matrix;
    delete[] out_matrix;
}

#if defined(USE_MPI)
#    include <mpi.h>

void
do_a2a(int rank)
{
    // Define my value
    int values[3];
    for(int i = 0; i < 3; ++i)
        values[i] = rank * 300 + i * 100;
    printf("Process %d, values = %d, %d, %d.\n", rank, values[0], values[1], values[2]);

    int buffer_recv[3];
    MPI_Alltoall(&values, 1, MPI_INT, buffer_recv, 1, MPI_INT, MPI_COMM_WORLD);
    printf("Values collected on process %d: %d, %d, %d.\n", rank, buffer_recv[0],
           buffer_recv[1], buffer_recv[2]);
}
#endif

int
main(int argc, char** argv)
{
    int rank = 0;
    int size = 1;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{ argv[i] };
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: transpose [NUM_THREADS (%zu)] [NUM_ITERATION (%zu)] "
                    "[SYNC_EVERY_N_ITERATIONS (%zu)]\n"
                    "\n"
                    "  NUM_ITERATION = 0 enables infinite-workload mode: each round\n"
                    "  runs the default 500 iterations end-to-end (alloc, kernel\n"
                    "  loop, verify, free) and rounds repeat until SIGTERM/SIGINT.\n",
                    g_nthreads, g_nitr, g_nsync);
            exit(EXIT_SUCCESS);
        }
    }
    if(argc > 1)
    {
        g_nthreads = atoll(argv[1]);
    }
    if(argc > 2)
    {
        g_nitr = atoll(argv[2]);
    }
    if(argc > 3)
    {
        g_nsync = atoll(argv[3]);
    }

    const bool infinite_workload = (g_nitr == 0);
    if(infinite_workload)
    {
        g_nitr = k_default_nitr;
    }

    std::signal(SIGTERM, handle_stop_signal);
    std::signal(SIGINT, handle_stop_signal);

    printf("[transpose] Number of threads: %zu\n", g_nthreads);
    if(infinite_workload)
    {
        printf("[transpose] Number of iterations per round: %zu (infinite "
               "rounds until SIGTERM/SIGINT)\n",
               g_nitr);
    }
    else
    {
        printf("[transpose] Number of iterations: %zu\n", g_nitr);
    }
    printf("[transpose] Syncing every %zu iterations\n", g_nsync);

#if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    (void) size;
#endif
    // Map each rank round-robin onto the available GPUs. When there are more
    // ranks than devices, multiple ranks share a device (e.g. 2 ranks on 1 GPU
    // both use device 0), so every rank performs GPU work under MPI.
    int ndevice = 0;
    int devid   = rank;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    printf("[transpose] Number of devices found: %i\n", ndevice);
    if(ndevice > 0)
    {
        devid = rank % ndevice;
        HIP_API_CALL(hipSetDevice(devid));
        printf("[transpose] Rank %i assigned to device %i\n", rank, devid);
    }
    if(rank == devid && rank < ndevice)
    {
        size_t round = 0;
        do
        {
            if(infinite_workload)
            {
                print_lock.lock();
                std::cout << "[transpose] Starting round " << round++ << std::endl;
                print_lock.unlock();
            }
            std::vector<std::thread> threads{};
            std::vector<hipStream_t> streams(g_nthreads);
            for(size_t i = 0; i < g_nthreads; ++i)
            {
                HIP_API_CALL(hipStreamCreate(&streams.at(i)));
            }
            for(size_t i = 1; i < g_nthreads; ++i)
            {
                threads.emplace_back(run, rank, i, devid, streams.at(i), argc, argv);
            }
            run(rank, 0, devid, streams.at(0), argc, argv);
            for(auto& itr : threads)
            {
                itr.join();
            }
            for(size_t i = 0; i < g_nthreads; ++i)
            {
                HIP_API_CALL(hipStreamDestroy(streams.at(i)));
            }
        } while(infinite_workload && !g_stop_requested.load(std::memory_order_relaxed));
    }
    HIP_API_CALL(hipDeviceSynchronize());
    HIP_API_CALL(hipDeviceReset());

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    do_a2a(rank);
    MPI_Finalize();
#endif

    return 0;
}
