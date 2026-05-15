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

#if defined(USE_ROCTRACER_ROCTX)
#    include <roctracer/roctx.h>
#else
#    include <rocprofiler-sdk-roctx/roctx.h>
#endif

#include <hip/hip_runtime.h>

#if defined(USE_MPI)
#    include <mpi.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

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
using auto_lock_t                      = std::unique_lock<std::mutex>;
auto               print_lock          = std::mutex{};
size_t             nthreads            = 2;
size_t             nitr                = 500;
size_t             nsync               = 10;
constexpr unsigned shared_mem_tile_dim = 32;

void
check_hip_error(void);

void
verify(int* in, int* out, int M, int N);
}  // namespace

__global__ void
transpose(const int* in, int* out, int M, int N);

void
run(int rank, int tid, int ndevice, int argc, char** argv);

int
main(int argc, char** argv)
{
    int rank = 0;
    int size = 1;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: transpose [NUM_THREADS (%zu)] [NUM_ITERATION (%zu)] "
                    "[SYNC_EVERY_N_ITERATIONS (%zu)]\n",
                    nthreads,
                    nitr,
                    nsync);
            exit(EXIT_SUCCESS);
        }
    }
    if(argc > 1) nthreads = atoll(argv[1]);
    if(argc > 2) nitr = atoll(argv[2]);
    if(argc > 3) nsync = atoll(argv[3]);

    printf("[transpose] Number of threads: %zu\n", nthreads);
    printf("[transpose] Number of iterations: %zu\n", nitr);
    printf("[transpose] Syncing every %zu iterations\n", nsync);

#if defined(USE_ROCTRACER_ROCTX)
    {
        auto _roctracer_roctx_ss = std::stringstream{};
        _roctracer_roctx_ss << "roctracer/roctx v" << roctx_version_major() << "."
                            << roctx_version_minor();
        roctxMark(_roctracer_roctx_ss.str().c_str());
    }
#endif

#if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    (void) size;
#endif
    // this is a temporary workaround in omnitrace when HIP + MPI is enabled
    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    printf("[transpose] Number of devices found: %i\n", ndevice);
    auto devids = std::vector<int>{};
    devids.resize(size * nthreads, 0);
    int devid = 0;
    for(size_t i = 0; i < nthreads; ++i)
    {
        for(int j = 0; j < size; ++j)
        {
            auto idx       = (j * nthreads) + i;
            devids.at(idx) = devid++ % ndevice;
        }
    }
    auto devid_offset = (rank * nthreads);
    auto _threads     = std::vector<std::thread>{};
    for(size_t i = 1; i < nthreads; ++i)
        _threads.emplace_back(run, rank, i, devids.at(devid_offset + i), argc, argv);
    run(rank, 0, devids.at(devid_offset + 0), argc, argv);
    for(auto& itr : _threads)
        itr.join();

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

__global__ void
transpose(const int* in, int* out, int M, int N)
{
    __shared__ int tile[shared_mem_tile_dim][shared_mem_tile_dim];

    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * M + blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    idx      = (blockIdx.x * blockDim.x + threadIdx.y) * N + blockIdx.y * blockDim.y + threadIdx.x;
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

void
run(int rank, int tid, int devid, int argc, char** argv)
{
    auto roctx_run_id = roctxRangeStart("run");

    const auto mark = [rank, tid, devid](std::string_view suffix) {
        auto _ss = std::stringstream{};
        _ss << "run/rank-" << rank << "/thread-" << tid << "/device-" << devid << "/" << suffix;
        roctxMark(_ss.str().c_str());
    };

    mark("begin");

    constexpr unsigned int M = 4960 * 2;
    constexpr unsigned int N = 4960 * 2;

    if(argc > 2) nitr = atoll(argv[2]);
    if(argc > 3) nsync = atoll(argv[3]);

    hipStream_t stream = {};

    printf("[transpose] Rank %i, thread %i assigned to device %i\n", rank, tid, devid);
    HIP_API_CALL(hipSetDevice(devid));
    HIP_API_CALL(hipStreamCreate(&stream));

    auto_lock_t _lk{print_lock};
    std::cout << "[transpose][" << rank << "][" << tid << "] M: " << M << " N: " << N << std::endl;
    _lk.unlock();

    std::default_random_engine         _engine{std::random_device{}() * (rank + 1) * (tid + 1)};
    std::uniform_int_distribution<int> _dist{0, 1000};

    size_t size       = sizeof(int) * M * N;
    int*   inp_matrix = new int[size];
    int*   out_matrix = new int[size];
    for(size_t i = 0; i < M * N; i++)
    {
        inp_matrix[i] = _dist(_engine);
        out_matrix[i] = 0;
    }
    int* in  = nullptr;
    int* out = nullptr;

    // lock during malloc to get more accurate memory info
    {
        _lk.lock();
        constexpr auto MiB           = (1024UL * 1024UL);
        size_t         free_gpu_mem  = 0;
        size_t         total_gpu_mem = 0;

        HIP_API_CALL(hipMemGetInfo(&free_gpu_mem, &total_gpu_mem));
        free_gpu_mem /= MiB;
        total_gpu_mem /= MiB;

        std::cout << "[transpose][" << rank << "][" << tid
                  << "] Available GPU memory (MiB): " << std::setw(6) << free_gpu_mem << " / "
                  << std::setw(6) << total_gpu_mem << std::endl;

        HIP_API_CALL(hipMallocAsync(&in, size, stream));
        HIP_API_CALL(hipMallocAsync(&out, size, stream));

        _lk.unlock();
    }

    HIP_API_CALL(hipMemsetAsync(in, 0, size, stream));
    HIP_API_CALL(hipMemsetAsync(out, 0, size, stream));
    HIP_API_CALL(hipMemcpyAsync(in, inp_matrix, size, hipMemcpyHostToDevice, stream));
    HIP_API_CALL(hipStreamSynchronize(stream));

    dim3 grid(M / 32, N / 32, 1);
    dim3 block(32, 32, 1);  // transpose

    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nitr; ++i)
    {
        roctxRangePush("run/iteration");
        transpose<<<grid, block, 0, stream>>>(in, out, M, N);
        check_hip_error();
        if(i % nsync == (nsync - 1))
        {
            roctxRangePush("run/iteration/sync");
            HIP_API_CALL(hipStreamSynchronize(stream));
            roctxRangePop();
        }
        roctxRangePop();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipMemcpyAsync(out_matrix, out, size, hipMemcpyDeviceToHost, stream));
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    float  GB   = (float) size * nitr * 2 / (1 << 30);

    print_lock.lock();
    std::cout << "[transpose][" << rank << "][" << tid << "] Runtime of transpose is " << time
              << " sec\n";
    std::cout << "[transpose][" << rank << "][" << tid
              << "] The average performance of transpose is " << GB / time << " GBytes/sec"
              << std::endl;
    print_lock.unlock();

    HIP_API_CALL(hipStreamSynchronize(stream));

    // cpu_transpose(matrix, out_matrix, M, N);
    verify(inp_matrix, out_matrix, M, N);

    HIP_API_CALL(hipFreeAsync(in, stream));
    HIP_API_CALL(hipFreeAsync(out, stream));

    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipStreamDestroy(stream));

    delete[] inp_matrix;
    delete[] out_matrix;

    mark("end");

    roctxRangeStop(roctx_run_id);
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

void
verify(int* in, int* out, int M, int N)
{
    for(int i = 0; i < 10; i++)
    {
        int row = rand() % M;
        int col = rand() % N;
        if(in[row * N + col] != out[col * M + row])
        {
            auto_lock_t _lk{print_lock};
            std::cout << "mismatch: " << row << ", " << col << " : " << in[row * N + col] << " | "
                      << out[col * M + row] << "\n";
        }
    }
}
}  // namespace
