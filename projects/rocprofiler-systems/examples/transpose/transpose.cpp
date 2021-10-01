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

#define HIP_API_CALL(CALL)                                                               \
    {                                                                                    \
        hipError_t error_ = (CALL);                                                      \
        if(error_ != hipSuccess)                                                         \
        {                                                                                \
            fprintf(stderr, "%s:%d :: HIP error : %s\n", __FILE__, __LINE__,             \
                    hipGetErrorString(error_));                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

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
    __shared__ int tile[TILE_DIM][TILE_DIM];

    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * M + blockIdx.x * blockDim.x +
              threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = in[idx];
    __syncthreads();
    idx = (blockIdx.x * blockDim.x + threadIdx.y) * N + blockIdx.y * blockDim.y +
          threadIdx.x;
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

void
run(int rank, int argc, char** argv)
{
    (void) argc;
    (void) argv;
    unsigned int M = 4960;
    unsigned int N = 4960;

    std::cout << "[" << rank << "] M: " << M << " N: " << N << std::endl;
    size_t size   = sizeof(int) * M * N;
    int*   matrix = (int*) malloc(size);
    for(size_t i = 0; i < M * N; i++)
        matrix[i] = rand() % 1002;
    int *in, *out;

    std::chrono::high_resolution_clock::time_point t1, t2;

    HIP_API_CALL(hipMalloc(&in, size));
    HIP_API_CALL(hipMalloc(&out, size));
    check_hip_error();
    HIP_API_CALL(hipMemset(in, 0, size));
    HIP_API_CALL(hipMemset(out, 0, size));
    HIP_API_CALL(hipMemcpy(in, matrix, size, hipMemcpyHostToDevice));
    HIP_API_CALL(hipDeviceSynchronize());
    check_hip_error();
    hipDeviceProp_t props;
    HIP_API_CALL(hipGetDeviceProperties(&props, 0));

    dim3 grid(M / 32, N / 32, 1);
    dim3 block(32, 32, 1);  // transpose_a

    // warmup
    hipLaunchKernelGGL(transpose_a, grid, block, 0, 0, in, out, M, N);
    check_hip_error();

    t1                   = std::chrono::high_resolution_clock::now();
    const unsigned times = 10000;
    for(size_t i = 0; i < times; i++)
    {
        hipLaunchKernelGGL(transpose_a, grid, block, 0, 0, in, out, M, N);
    }
    check_hip_error();
    HIP_API_CALL(hipDeviceSynchronize());
    t2 = std::chrono::high_resolution_clock::now();
    double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    float GB = (float) size * times * 2 / (1 << 30);
    std::cout << "[" << rank << "] Runtime of transpose is " << time << " sec\n"
              << "The average performance of transpose is " << GB / time << " GBytes/sec"
              << std::endl;

    int* out_matrix = (int*) malloc(size);
    HIP_API_CALL(hipMemcpy(out_matrix, out, size, hipMemcpyDeviceToHost));
    check_hip_error();

    // cpu_transpose(matrix, out_matrix, M, N);
    verify(matrix, out_matrix, M, N);

    HIP_API_CALL(hipFree(in));
    HIP_API_CALL(hipFree(out));
    check_hip_error();

    free(matrix);
    free(out_matrix);
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
#if defined(USE_MPI)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if(rank == 0) run(rank, argc, argv);
#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    do_a2a(rank);
    MPI_Finalize();
#endif
    return 0;
}
