/*
Copyright (c) 2015-2020 Advanced Micro Devices, Inc. All rights reserved.

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
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

void
check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
        exit(err);
    }
}

__global__ void
transpose_naive(int* in, int* out, int M, int N)
{
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    for(int i = idx; i < M * N; i += hipBlockDim_x * hipGridDim_x)
    {
        int row            = i / N;
        int col            = i % N;
        out[col * M + row] = in[row * N + col];
    }
}

void
cpu_transpose(int* in, int* out, int M, int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            out[j * M + i] = in[i * N + j];
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
            std::cout << "mismatch: " << row << ", " << col << " : " << in[row * N + col]
                      << " | " << out[col * M + row] << "\n";
        }
    }
}
const unsigned TILE_DIM = 32;
__global__ void
transpose_a(int* in, int* out, int M, int N)
{
    int iidx = (blockIdx.x * blockDim.x + threadIdx.x) * N + blockIdx.y * blockDim.y +
               threadIdx.y;
    int oidx =
        (blockIdx.y * blockDim.y + threadIdx.y) * +blockIdx.x * blockDim.x + threadIdx.x;

    out[oidx] = in[iidx];
}

const int NUM_ITEM = 8;
__global__ void
transpose_e(int* A, int* B, int n1, int n2)
{
    __shared__ int Cs[64][64 + 1];
    int            index;

    index = (blockIdx.y * blockDim.y * NUM_ITEM + threadIdx.y) * n1 +
            blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < NUM_ITEM; ++i)
    {
        Cs[threadIdx.y + i * NUM_ITEM][threadIdx.x] = A[index + i * NUM_ITEM * n1];
    }
    __syncthreads();
    index = (blockIdx.x * blockDim.x + threadIdx.y) * n2 +
            blockIdx.y * blockDim.y * NUM_ITEM + threadIdx.x;
    for(int i = 0; i < NUM_ITEM; ++i)
    {
        B[index + i * NUM_ITEM * n2] = Cs[threadIdx.x][threadIdx.y + i * NUM_ITEM];
    }
}

__global__ void
transpose_s(int* out, int* in, int n1, int n2)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    int val = in[x];

    for(int i = 0; i < n1; i++)
    {
        for(int j = 0; j < n2; j++)
            out[i * n2 + j] = __shfl(val, j * n2 + i);
    }
}

int
main(int argc, char** argv)
{
    int nx = 32;
    int ny = 32;
    if(argc > 1) nx = atoi(argv[1]);
    if(argc > 2) ny = atoi(argv[2]);

    unsigned int M = 4960;
    unsigned int N = 4960;

    if(argc > 3) M = atoi(argv[3]);
    if(argc > 4) N = atoi(argv[4]);

    std::cout << "M: " << M << " N: " << N << std::endl;
    size_t size   = sizeof(int) * M * N;
    int*   matrix = (int*) malloc(size);
    for(int i = 0; i < M * N; i++)
        matrix[i] = rand() % 1002;
    int *in, *out;

    std::chrono::high_resolution_clock::time_point t1, t2;

    hipMalloc(&in, size);
    hipMalloc(&out, size);
    hipMemset(in, 0, size);
    hipMemset(out, 0, size);
    check_hip_error();
    hipMemcpy(in, matrix, size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    check_hip_error();
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);

    dim3 grid(M / nx, N / ny, 1);
    dim3 block(nx, ny, 1);  // transpose_a
    // dim3 grid(M/64, N/64, 1); dim3 block(64, 8, 1); // transpose_e

#define TRANSPOSE_KERNEL transpose_a

    // warmup
    hipLaunchKernelGGL(TRANSPOSE_KERNEL, grid, block, 0, 0, in, out, M, N);
    check_hip_error();

    t1                   = std::chrono::high_resolution_clock::now();
    const unsigned times = 10000;
    for(int i = 0; i < times; i++)
    {
        hipLaunchKernelGGL(TRANSPOSE_KERNEL, grid, block, 0, 0, in, out, M, N);
        check_hip_error();
        hipDeviceSynchronize();
        check_hip_error();
    }
    t2 = std::chrono::high_resolution_clock::now();
    double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    float GB = (float) size * times * 2 / (1 << 30);
    std::cout << "The average performance of transpose is " << GB / time << " GBytes/sec"
              << std::endl;

    int* out_matrix = (int*) malloc(size);
    hipMemcpy(out_matrix, out, size, hipMemcpyDeviceToHost);
    check_hip_error();

    // cpu_transpose(matrix, out_matrix, M, N);
    verify(matrix, out_matrix, M, N);

    hipFree(in);
    hipFree(out);
    check_hip_error();

    free(matrix);
    free(out_matrix);

    return 0;
}

/*
dim3 threads(256,1,1);                //3D dimensions of a block of threads
dim3 blocks((N+256-1)/256,1,1);       //3D dimensions the grid of blocks

hipLaunchKernelGGL(myKernel,          //Kernel name (__global__ void function)
             blocks,           //Grid dimensions
             threads,          //Block dimensions
             0,                //Bytes of dynamic LDS space (see extra slides)
             0,                //Stream (0=NULL stream)
             N, a);            //Kernel arguments
*/
