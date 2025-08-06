// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <vector>
#include "transpose_kernels.hpp"

#define PRINT_ALIGN 36

namespace
{
using lock_guard_t = std::lock_guard<std::mutex>;
auto print_lock    = std::mutex{};
}  // namespace

enum TransposeType
{
    TRANSPOSE_NAIVE,
    TRANSPOSE_INPLACE_LDS,
    TRANSPOSE_NO_BANK_CONFLICTS
};

class ITranspose
{
public:
    virtual void run(TransposeType ttype, int numThreadsY, int num_iter) = 0;
    virtual ~ITranspose(){};
};

template <typename T>
class Transpose : public ITranspose
{
public:
    Transpose(int dev, size_t _M)
    : devID(dev)
    , M(_M)
    , databytes(_M * _M * sizeof(T))
    {
        HIP_API_CALL(hipSetDevice(devID));
        HIP_API_CALL(hipStreamCreate(&stream));

        std::default_random_engine         _engine{std::random_device{}() * rand()};
        std::uniform_int_distribution<int> _dist{0, 1000};

        inp_matrix = new T[M * M];
        out_matrix = new T[M * M];

        for(size_t i = 0; i < M * M; i++)
            inp_matrix[i] = static_cast<T>(_dist(_engine));
        memset(out_matrix, 0, databytes);

        HIP_API_CALL(hipMalloc(&in, databytes));
        HIP_API_CALL(hipMalloc(&out, databytes));
        HIP_API_CALL(hipMemsetAsync(in, 0, databytes, stream));
        HIP_API_CALL(hipMemsetAsync(out, 0, databytes, stream));
        HIP_API_CALL(hipMemcpyAsync(in, inp_matrix, databytes, hipMemcpyDefault, stream));

        HIP_API_CALL(hipEventCreate(&start));
        HIP_API_CALL(hipEventCreate(&stop));
    }

    void run(TransposeType ttype, int numThreadsY, int num_iter) override
    {
        HIP_API_CALL(hipSetDevice(devID));
        dim3 grid(M / TILE_DIM, M / TILE_DIM, 1);
        dim3 block(TILE_DIM, numThreadsY, 1);

        auto        Kernel     = transposeNaive<T>;
        std::string KernelName = "transposeNaive";
        if(ttype == TransposeType::TRANSPOSE_NO_BANK_CONFLICTS)
        {
            Kernel     = transposeLdsNoBankConflicts<T>;
            KernelName = "transposeLdsNoBankConflicts";
        }
        else if(ttype == TransposeType::TRANSPOSE_INPLACE_LDS)
        {
            Kernel     = transposeLdsSwapInplace<T>;
            KernelName = "transposeLdsSwapInplace";
        }

        {
            std::string functypeid = __PRETTY_FUNCTION__;
            auto        it_beg     = functypeid.rfind("[T = ");
            auto        it_end     = functypeid.rfind(']');

            if(it_beg != std::string::npos) it_beg += std::string("[T = ").size();

            if(it_beg < it_end && it_end != std::string::npos)
                KernelName += '<' + functypeid.substr(it_beg, it_end - it_beg) + '>';
        }

        HIP_API_CALL(hipStreamSynchronize(stream));
        HIP_API_CALL(hipEventRecord(start, stream));

        for(int i = 0; i < num_iter; i++)
        {
            Kernel<<<grid, block, 0, stream>>>(out, in, M);
            HIP_API_CALL(hipGetLastError());
        }

        HIP_API_CALL(hipEventRecord(stop, stream));
        HIP_API_CALL(hipMemcpyAsync(out_matrix, out, databytes, hipMemcpyDefault, stream));
        HIP_API_CALL(hipEventSynchronize(stop));

        float time;
        HIP_API_CALL(hipEventElapsedTime(&time, start, stop));
        float GB = databytes * num_iter * 2 / float(1 << 30);

        {
            lock_guard_t _lk{print_lock};
            std::cout << "The average performance of " << std::setw(38) << KernelName << " : "
                      << (1000 * GB / time) << " GB/s" << std::endl;
        }

        verify();
    }

    void verify() const
    {
        HIP_API_CALL(hipStreamSynchronize(stream));
        for(int i = 0; i < 10; i++)
        {
            int row = rand() % M;
            int col = rand() % M;
            if(inp_matrix[row * M + col] != out_matrix[col * M + row])
            {
                lock_guard_t _lk{print_lock};
                std::cout << "mismatch: " << row << ", " << col << " : "
                          << inp_matrix[row * M + col] << " | " << out_matrix[col * M + row]
                          << std::endl;
            }
        }
    }

    virtual ~Transpose()
    {
        HIP_API_CALL(hipSetDevice(devID));
        HIP_API_CALL(hipEventDestroy(start));
        HIP_API_CALL(hipEventDestroy(stop));

        HIP_API_CALL(hipFree(in));
        HIP_API_CALL(hipFree(out));
        HIP_API_CALL(hipStreamDestroy(stream));

        delete[] inp_matrix;
        delete[] out_matrix;
    }

    const int    devID;
    const size_t M;
    const size_t databytes;

    hipStream_t stream;
    hipEvent_t  start, stop;

    T* inp_matrix = nullptr;
    T* out_matrix = nullptr;

    T* in  = nullptr;
    T* out = nullptr;
};

int
main(int argc, char** argv)
{
    int deviceId  = 0;
    int blockDimY = 8;
    int num_iter  = 3;
    int mat_size  = 8192;

    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            std::cout << "usage: transpose "
                      << "[MatrixSize (" << mat_size << ")] "
                      << "[numIter (" << num_iter << ")] "
                      << "[blockDimY (" << blockDimY << ")] "
                      << "[DEVICE_ID (" << deviceId << ")] " << std::endl;
            exit(EXIT_SUCCESS);
        }
    }
    if(argc > 1) mat_size = atoll(argv[1]);
    if(argc > 2) num_iter = atoll(argv[2]);
    if(argc > 3) blockDimY = atoll(argv[3]);
    if(argc > 4) deviceId = atoll(argv[4]);

    printf("[transpose] Matrix size: %d, device ID: %d, num iter: %d, blockDimY: %d\n",
           mat_size,
           deviceId,
           num_iter,
           blockDimY);

    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));
    printf("[transpose] Number of devices found: %i\n", ndevice);
    assert(ndevice > 0);

    if(deviceId >= ndevice) exit(EXIT_FAILURE);

    {
        std::vector<std::unique_ptr<ITranspose>> kernels;
        kernels.push_back(std::make_unique<Transpose<int>>(deviceId, mat_size));
        kernels.push_back(std::make_unique<Transpose<float>>(deviceId, mat_size));
        kernels.push_back(std::make_unique<Transpose<double>>(deviceId, mat_size));

        for(auto& kernel : kernels)
        {
            kernel->run(TransposeType::TRANSPOSE_NAIVE, blockDimY, num_iter);
            kernel->run(TransposeType::TRANSPOSE_INPLACE_LDS, blockDimY, num_iter);
            kernel->run(TransposeType::TRANSPOSE_NO_BANK_CONFLICTS, blockDimY, num_iter);
        }
    }

    HIP_API_CALL(hipDeviceSynchronize());
    HIP_API_CALL(hipDeviceReset());

    return 0;
}
