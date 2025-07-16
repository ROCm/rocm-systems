// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <random>

namespace
{
#define M                          8192
#define N                          8192
#define K                          8192
#define TileSize                   16
#define BLOCK_SIZE_X               16
#define BLOCK_SIZE_Y               16
#define GRID_SIZE_X                (M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X
#define GRID_SIZE_Y                (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
#define WAVES_PER_BLOCK_MI200_PLUS (BLOCK_SIZE_X * BLOCK_SIZE_Y) / 64

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }
}  // namespace

namespace
{
void
check_hip_error(void);
}  // namespace

__global__ void
matrix_multiply(float* A, float* B, float* Out, int /*m*/, int n, int k)
{
    int gid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int gid_y = blockDim.y * blockIdx.y + threadIdx.y;

    if(gid_x < N && gid_y < M)
    {
        float sum = 0;
        for(int i = 0; i < k; ++i)
        {
            sum += A[gid_y * k + i] * B[i * n + gid_x];
        }

        Out[gid_y * n + gid_x] = sum;
    }
}

#if 1
__global__ void
matrix_multiply_tile(float* A, float* B, float* Out, int m, int n, int k)
{
    __shared__ float subTileM[TileSize][TileSize];
    __shared__ float subTileN[TileSize][TileSize];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TileSize + ty;
    int col = bx * TileSize + tx;

    float sum = 0;
    for(int i = 0; i < ((k - 1) / TileSize + 1); i++)
    {
        int curr_l = row * k + i * TileSize + tx;
        int curr_r = (i * TileSize + ty) * n + col;

        if(i * TileSize + tx < k && row < m)
        {
            subTileM[ty][tx] = A[curr_l];
        }
        else
        {
            subTileM[ty][tx] = 0.0;
        }

        if(i * TileSize + ty < k && col < n)
        {
            subTileN[ty][tx] = B[curr_r];
        }
        else
        {
            subTileN[ty][tx] = 0.0;
        }

        __syncthreads();

        for(int j = 0; j < TileSize; j++)
        {
            if(j + TileSize * i < k)
            {
                sum += subTileM[ty][j] * subTileN[j][tx];
            }
        }

        __syncthreads();
    }

    if(row < m && col < n)
    {
        Out[row * n + col] = sum;
    }
}
#endif

void
run_hip_app()
{
    hipDeviceProp_t prop{};
    HIP_API_CALL(hipGetDeviceProperties(&prop, 0));
    std::string_view gfxip(prop.gcnArchName);
    // We run at full scale on all architectures except on Navi4x
    // that requires reduced workload due to LDS contention.
    bool full_scale_run = gfxip.find("gfx120") == std::string_view::npos;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> Out(M * N);

    // Randomly initialize the matrices
    for(int i = 0; i < M * K; ++i)
    {
        A[i] = (float) rand() / (float) RAND_MAX;
    }

    for(int i = 0; i < K * N; ++i)
    {
        B[i] = (float) rand() / (float) RAND_MAX;
    }

    // Allocate GPU Memory
    float *d_A, *d_B, *d_Out;
    HIP_API_CALL(hipMalloc(&d_A, sizeof(float) * M * K));
    HIP_API_CALL(hipMalloc(&d_B, sizeof(float) * K * N));
    HIP_API_CALL(hipMalloc(&d_Out, sizeof(float) * M * N));

    // Copy data to GPU
    HIP_API_CALL(hipMemcpy(d_A, A.data(), sizeof(float) * M * K, hipMemcpyHostToDevice));
    HIP_API_CALL(hipMemcpy(d_B, B.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));

    // Run the kernel
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);
    matrix_multiply<<<grid_size, block_size>>>(d_A, d_B, d_Out, M, N, K);
    check_hip_error();

    // Check whether it's safe to launch additional matmul kernel that utilizes LDS.
    if(full_scale_run)
    {
        matrix_multiply_tile<<<grid_size, block_size>>>(d_A, d_B, d_Out, M, N, K);
        check_hip_error();
    }

    // Copy data back to CPU
    HIP_API_CALL(hipMemcpy(Out.data(), d_Out, sizeof(float) * M * N, hipMemcpyDeviceToHost));

    // Free GPU Memory
    HIP_API_CALL(hipFree(d_A));
    HIP_API_CALL(hipFree(d_B));
    HIP_API_CALL(hipFree(d_Out));
}

#define DEVICE_ID 0

int
main(int /*argc*/, char** /*argv*/)
{
    int deviceId = DEVICE_ID;

    auto status = hipSetDevice(deviceId);
    assert(status == hipSuccess);
    HIP_API_CALL(status);

    int currDeviceId = -1;
    status           = hipGetDevice(&currDeviceId);
    HIP_API_CALL(status);
    assert(status == hipSuccess);
    assert(deviceId == currDeviceId);

    for(int i = 0; i < 1; i++)
    {
        std::cout << "<<< MatMul starts" << std::endl;
        run_hip_app();
        std::cout << ">>> MatMul ends" << std::endl;
    }

    return 0;
}

namespace
{
void
check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hip_api_call");
    }
}
}  // namespace
