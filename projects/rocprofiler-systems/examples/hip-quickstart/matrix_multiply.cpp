/*
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * @file matrix_multiply.cpp
 * @brief Matrix multiplication example showing optimization opportunities
 *
 * This example demonstrates:
 * - Naive matrix multiplication (unoptimized)
 * - Tiled matrix multiplication (optimized with shared memory)
 * - Performance comparison between implementations
 *
 * PROFILING THIS EXAMPLE:
 *
 *   # Quick profiling
 *   rocprof-sys-sample --quick --hip-trace -- ./matrix_multiply
 *
 *   # Compare kernel performance
 *   rocprof-sys-run --hip-trace --rocm-events=SQ_WAVES,MemUnitBusy -- ./matrix_multiply.inst
 *
 * WHAT TO LOOK FOR:
 * - Execution time difference between naive and tiled kernels
 * - Memory access patterns (tiled should have better locality)
 * - GPU utilization during each kernel
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define HIP_CHECK(cmd)                                                                       \
    {                                                                                        \
        hipError_t error = (cmd);                                                            \
        if(error != hipSuccess)                                                              \
        {                                                                                    \
            std::cerr << "HIP error: " << hipGetErrorString(error)                           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;                 \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    }

// Tile size for optimized kernel
constexpr int TILE_SIZE = 16;

/**
 * @brief Naive matrix multiplication kernel
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M,K,N Matrix dimensions
 *
 * PROFILING INSIGHT: This kernel has poor memory locality.
 * Each thread reads entire rows/columns from global memory repeatedly.
 * Expect high memory latency and low cache hit rates.
 */
__global__ void
matmul_naive(const float* A, const float* B, float* C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N)
    {
        float sum = 0.0f;
        
        // Each thread computes one element of C
        // Problem: Redundant global memory accesses
        for(int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

/**
 * @brief Tiled matrix multiplication using shared memory
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M,K,N Matrix dimensions
 *
 * PROFILING INSIGHT: This kernel uses shared memory (LDS) to cache tiles.
 * Should see improved performance due to better memory locality.
 * Look for reduced memory unit busy time compared to naive version.
 */
__global__ void
matmul_tiled(const float* A, const float* B, float* C, int M, int K, int N)
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for(int t = 0; t < num_tiles; ++t)
    {
        // Load tile from A into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        if(row < M && a_col < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if(b_row < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product from this tile
        for(int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if(row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

/**
 * @brief CPU reference implementation for verification
 */
void
matmul_cpu(const std::vector<float>& A, const std::vector<float>& B,
           std::vector<float>& C, int M, int K, int N)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Verify GPU results against CPU reference
 */
bool
verify_results(const std::vector<float>& gpu_result,
               const std::vector<float>& cpu_result,
               float tolerance = 1e-3f)
{
    for(size_t i = 0; i < gpu_result.size(); ++i)
    {
        if(std::abs(gpu_result[i] - cpu_result[i]) > tolerance)
        {
            std::cerr << "Verification failed at index " << i
                      << ": GPU=" << gpu_result[i]
                      << ", CPU=" << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

int
main(int argc, char** argv)
{
    // Parse matrix dimensions
    int M = 1024, K = 1024, N = 1024;
    if(argc > 1) M = std::atoi(argv[1]);
    if(argc > 2) K = std::atoi(argv[2]);
    if(argc > 3) N = std::atoi(argv[3]);

    std::cout << "Matrix Multiplication Example\n";
    std::cout << "=============================\n";
    std::cout << "Matrix A: " << M << " x " << K << "\n";
    std::cout << "Matrix B: " << K << " x " << N << "\n";
    std::cout << "Matrix C: " << M << " x " << N << "\n\n";

    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    // Allocate and initialize host matrices
    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_naive(size_C);
    std::vector<float> h_C_tiled(size_C);
    std::vector<float> h_C_cpu(size_C);

    // Initialize matrices
    for(size_t i = 0; i < size_A; ++i) h_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for(size_t i = 0; i < size_B; ++i) h_B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, size_C * sizeof(float)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice));

    // Configure kernel launch
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);

    std::cout << "Grid size: " << grid_size.x << " x " << grid_size.y << "\n";
    std::cout << "Block size: " << block_size.x << " x " << block_size.y << "\n\n";

    // === NAIVE KERNEL ===
    std::cout << "Running NAIVE kernel...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(matmul_naive, grid_size, block_size, 0, 0,
                       d_A, d_B, d_C, M, K, N);
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    double naive_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Execution time: " << naive_time << " ms\n";

    // Copy result
    HIP_CHECK(hipMemcpy(h_C_naive.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));

    // === TILED KERNEL ===
    std::cout << "Running TILED kernel...\n";
    start = std::chrono::high_resolution_clock::now();
    
    hipLaunchKernelGGL(matmul_tiled, grid_size, block_size, 0, 0,
                       d_A, d_B, d_C, M, K, N);
    HIP_CHECK(hipDeviceSynchronize());
    
    end = std::chrono::high_resolution_clock::now();
    double tiled_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Execution time: " << tiled_time << " ms\n\n";

    // Copy result
    HIP_CHECK(hipMemcpy(h_C_tiled.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));

    // Compute speedup
    double speedup = naive_time / tiled_time;
    std::cout << "SPEEDUP: " << speedup << "x\n\n";

    // Verify results (using small subset for CPU reference)
    if(M <= 512 && N <= 512)
    {
        std::cout << "Computing CPU reference...\n";
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        
        std::cout << "Verifying naive kernel...\n";
        if(verify_results(h_C_naive, h_C_cpu))
            std::cout << "  Naive kernel: CORRECT\n";
        
        std::cout << "Verifying tiled kernel...\n";
        if(verify_results(h_C_tiled, h_C_cpu))
            std::cout << "  Tiled kernel: CORRECT\n";
    }
    else
    {
        std::cout << "Matrix too large for CPU verification, checking GPU consistency...\n";
        if(verify_results(h_C_naive, h_C_tiled))
            std::cout << "  Naive and tiled results match!\n";
    }

    // Clean up
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    std::cout << "\nPROFILING TIPS:\n";
    std::cout << "1. Compare memory access patterns between naive and tiled\n";
    std::cout << "2. Check MemUnitBusy counter (should be lower for tiled)\n";
    std::cout << "3. Look at cache hit rates (L2 should be better for tiled)\n";
    std::cout << "4. Observe arithmetic intensity differences\n";
    std::cout << "5. Try different matrix sizes to see scaling behavior\n";

    return EXIT_SUCCESS;
}

