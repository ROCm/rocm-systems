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
#include <string>

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
matrix_multiply(float* A, float* B, float* Out, int m, int n, int k)
{
    int gid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int gid_y = blockDim.y * blockIdx.y + threadIdx.y;

    if(gid_x < n && gid_y < m)
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
run_hip_app(const std::string& /*arch_name*/)
{
    size_t m = M, n = N, k = K;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> Out(m * n);

    for(size_t i = 0; i < m * k; ++i)
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for(size_t i = 0; i < k * n; ++i)
        B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    float *d_A, *d_B, *d_Out;
    HIP_API_CALL(hipMalloc(&d_A, sizeof(float) * m * k));
    HIP_API_CALL(hipMalloc(&d_B, sizeof(float) * k * n));
    HIP_API_CALL(hipMalloc(&d_Out, sizeof(float) * m * n));

    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

    matrix_multiply<<<grid_size, block_size>>>(
        d_A, d_B, d_Out, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    check_hip_error();

    matrix_multiply_tile<<<grid_size, block_size>>>(
        d_A, d_B, d_Out, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    check_hip_error();

    HIP_API_CALL(hipMemcpy(Out.data(), d_Out, sizeof(float) * m * n, hipMemcpyDeviceToHost));

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

    hipDeviceProp_t device_props{};
    HIP_API_CALL(hipGetDeviceProperties(&device_props, deviceId));
    std::string arch_name = device_props.gcnArchName;

    for(int i = 0; i < 1; i++)
    {
        std::cout << "<<< MatMul starts" << std::endl;
        run_hip_app(arch_name);
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
