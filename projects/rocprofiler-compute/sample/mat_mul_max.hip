#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#define TILE_SIZE 32     // Maximum block size: 32 x 32 = 1024 threads/block
#define N 4096           // Matrix size: 4096 x 4096 (~67M elements)

__global__ void matMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int width) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < width / TILE_SIZE; ++t) {
        tileA[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * width + col];
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    C[row * width + col] = sum;
}

int main() {
    size_t size = N * N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *d_A, *d_B, *d_C;

    // Initialize matrices with dummy values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) * 0.01f;
        h_B[i] = static_cast<float>((i + 1) % 100) * 0.01f;
    }

    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 32 x 32 = 1024 threads
    dim3 gridDim(N / TILE_SIZE, N / TILE_SIZE);  // 128 x 128 = 16,384 thread blocks

    std::cout << "Launching kernel with grid: (" << gridDim.x << ", " << gridDim.y
              << ") and block: (" << blockDim.x << ", " << blockDim.y << ")\n";

    auto start = std::chrono::high_resolution_clock::now();

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    hipDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    delete[] h_A;
    delete[] h_B;

    return 0;
}