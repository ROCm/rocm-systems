// GEMM kernel for embedded ELF test fixture.
// Compiled offline with hipcc --genco, then embedded as a byte array.
//
// This is a realistic kernel with:
//   - Multiple global_load instructions (A and B matrix tiles)
//   - A global_store instruction (C matrix output)
//   - A loop (k-dimension accumulation)
//   - Conditional control flow (bounds checking)
//   - Non-trivial register pressure from FMA accumulation
//   - Shared memory (LDS) usage for tiling
//
// NOT a toy vectorAdd — this exercises real compiler register allocation,
// spilling, and instruction scheduling.

#include <hip/hip_runtime.h>

#define TILE_SIZE 16

extern "C" __global__
void sgemm_naive(const float* __restrict__ A,
                 const float* __restrict__ B,
                 float* __restrict__ C,
                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

extern "C" __global__
void sgemm_tiled(const float* __restrict__ A,
                 const float* __restrict__ B,
                 float* __restrict__ C,
                 int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Accumulate tile product
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
