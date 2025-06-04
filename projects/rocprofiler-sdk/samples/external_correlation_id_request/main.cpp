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

#include "common/defines.hpp"
#include "hip/hip_runtime.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
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
size_t             nthread_per_device  = 2;
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
run(int rank, int tid, int devid, int argc, char** argv);

void
run_transpose(int rank, int tid, hipStream_t stream, int argc, char** argv);

void
run_migrate(int rank, int tid, hipStream_t stream, int, char** argv);

void
run_scratch(int rank, int tid, hipStream_t stream, int argc, char** argv);

int
main(int argc, char** argv)
{
    auto* exe_name = ::basename(argv[0]);

    int rank = 0;
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: %s [NUM_THREADS_PER_DEVICE (%zu)] [NUM_ITERATION (%zu)] "
                    "[SYNC_EVERY_N_ITERATIONS (%zu)]\n",
                    exe_name,
                    nthread_per_device,
                    nitr,
                    nsync);
            exit(EXIT_SUCCESS);
        }
    }
    if(argc > 1) nthread_per_device = atoll(argv[1]);
    if(argc > 2) nitr = atoll(argv[2]);
    if(argc > 3) nsync = atoll(argv[3]);

    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));

    auto nthreads = (ndevice * nthread_per_device);

    printf("[%s] Number of devices found: %i\n", exe_name, ndevice);
    printf("[%s] Number of threads (per device): %zu\n", exe_name, nthread_per_device);
    printf("[%s] Number of threads (total): %zu\n", exe_name, nthreads);
    printf("[%s] Number of iterations: %zu\n", exe_name, nitr);
    printf("[%s] Syncing every %zu iterations\n", exe_name, nsync);

    {
        auto _threads = std::vector<std::thread>{};
        for(size_t i = 0; i < nthreads; ++i)
            _threads.emplace_back(run, rank, i, i % ndevice, argc, argv);
        for(auto& itr : _threads)
            itr.join();
    }

    HIP_API_CALL(hipDeviceSynchronize());
    HIP_API_CALL(hipDeviceReset());

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

template <typename Tp>
__global__ void
test_page_migrate(Tp* data, Tp val)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    data[idx] += val;
}

__global__ void
test_kern_large(uint64_t* output)
{
    uint64_t result = 0;
    int      test[4000];
    memset(test, 5, 4000);
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

__global__ void
test_kern_medium(uint64_t* output)
{
    uint64_t result = 0;
    int      test[175];
    memset(test, 5, 175);
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

__global__ void
test_kern_small(uint64_t* output)
{
    uint64_t result = 0;
    int      test[2];
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

void
run(int rank, int tid, int devid, int argc, char** argv)
{
    auto* stream = hipStream_t{};
    HIP_API_CALL(hipSetDevice(devid));
    HIP_API_CALL(hipStreamCreate(&stream));

    run_migrate(rank, tid, stream, argc, argv);
    run_scratch(rank, tid, stream, argc, argv);
    run_transpose(rank, tid, stream, argc, argv);

    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipStreamDestroy(stream));
}

void
run_transpose(int rank, int tid, hipStream_t stream, int argc, char** argv)
{
    auto* exe_name = ::basename(argv[0]);

    unsigned int M = 4960 * 2;
    unsigned int N = 4960 * 2;
    if(argc > 2) nitr = atoll(argv[2]);
    if(argc > 3) nsync = atoll(argv[3]);

    auto_lock_t _lk{print_lock};
    std::cout << "[" << exe_name << "][transpose][" << rank << "][" << tid << "] M: " << M
              << " N: " << N << std::endl;
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

    HIP_API_CALL(hipMalloc(&in, size));
    HIP_API_CALL(hipMalloc(&out, size));
    HIP_API_CALL(hipMemsetAsync(in, 0, size, stream));
    HIP_API_CALL(hipMemsetAsync(out, 0, size, stream));
    HIP_API_CALL(hipMemcpyAsync(in, inp_matrix, size, hipMemcpyHostToDevice, stream));
    HIP_API_CALL(hipStreamSynchronize(stream));

    dim3 grid(M / 32, N / 32, 1);
    dim3 block(32, 32, 1);  // transpose

    print_lock.lock();
    printf("[%s][transpose][%i][%i] grid=(%i,%i,%i), block=(%i,%i,%i)\n",
           exe_name,
           rank,
           tid,
           grid.x,
           grid.y,
           grid.z,
           block.x,
           block.y,
           block.z);
    print_lock.unlock();

    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nitr; ++i)
    {
        transpose<<<grid, block, 0, stream>>>(in, out, M, N);
        check_hip_error();
        if(i % nsync == (nsync - 1)) HIP_API_CALL(hipStreamSynchronize(stream));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipMemcpyAsync(out_matrix, out, size, hipMemcpyDeviceToHost, stream));
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    float  GB   = (float) size * nitr * 2 / (1 << 30);

    print_lock.lock();
    std::cout << "[" << exe_name << "][transpose][" << rank << "][" << tid
              << "] Runtime of transpose is " << time << " sec\n";
    std::cout << "[" << exe_name << "][transpose][" << rank << "][" << tid
              << "] The average performance of transpose is " << GB / time << " GBytes/sec"
              << std::endl;
    print_lock.unlock();

    HIP_API_CALL(hipStreamSynchronize(stream));

    // cpu_transpose(matrix, out_matrix, M, N);
    verify(inp_matrix, out_matrix, M, N);

    HIP_API_CALL(hipFree(in));
    HIP_API_CALL(hipFree(out));

    delete[] inp_matrix;
    delete[] out_matrix;
}

void
run_scratch(int rank, int tid, hipStream_t stream, int, char** argv)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_API_CALL(hipStreamSynchronize(stream));

    const auto* exe_name = ::basename(argv[0]);

    uint64_t* data_ptr = nullptr;
    HIP_API_CALL(HIP_HOST_ALLOC_FUNC(&data_ptr, sizeof(uint64_t), 0));
    *data_ptr = 0;

    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    test_kern_medium<<<1000, 1, 0, stream>>>(data_ptr);
    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    test_kern_large<<<1100, 1, 0, stream>>>(data_ptr);
    HIP_API_CALL(hipStreamSynchronize(stream));

    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    HIP_API_CALL(hipStreamSynchronize(stream));

    test_kern_medium<<<1000, 1, 0, stream>>>(data_ptr);
    HIP_API_CALL(hipStreamSynchronize(stream));

    test_kern_small<<<1000, 1, 0, stream>>>(data_ptr);
    HIP_API_CALL(hipStreamSynchronize(stream));

    test_kern_large<<<1100, 1, 0, stream>>>(data_ptr);
    HIP_API_CALL(hipStreamSynchronize(stream));

    auto   t2   = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    print_lock.lock();
    std::cout << "[" << exe_name << "][scratch][" << rank << "][" << tid
              << "] Runtime of scratch is " << time << " sec\n";
    print_lock.unlock();
}

void
run_migrate(int rank, int tid, hipStream_t stream, int, char** argv)
{
    using data_type            = uint64_t;
    constexpr data_type init_v = 1;
    constexpr data_type incr_v = 1;

    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_API_CALL(hipStreamSynchronize(stream));

    const auto* exe_name  = ::basename(argv[0]);
    auto        page_data = std::vector<data_type>(1024, 0);

    HIP_API_CALL(hipHostRegister(
        page_data.data(), page_data.size() * sizeof(data_type), hipHostRegisterDefault));

    for(auto& itr : page_data)
        itr = init_v;

    test_page_migrate<<<1, 1024, 0, stream>>>(page_data.data(), incr_v);

    HIP_API_CALL(hipStreamSynchronize(stream));

    for(auto& itr : page_data)
    {
        auto diff = (itr - incr_v);
        if(diff != init_v)
        {
            auto msg = std::stringstream{};
            msg << "invalid diff: " << diff << ". expected: " << init_v;
            throw std::runtime_error{msg.str()};
        }
    }

    HIP_API_CALL(hipHostUnregister(page_data.data()));

    auto   t2   = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

    print_lock.lock();
    std::cout << "[" << exe_name << "][migrate][" << rank << "][" << tid
              << "] Runtime of migrate is " << time << " sec\n";
    print_lock.unlock();
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
