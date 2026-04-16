#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Mixed VMEM+LDS kernel: interleaved global loads/stores with LDS reads/writes.
// Tests that SiteAnalyzer correctly tracks both categories and that the
// trampoline builder handles mixed-type instrumentation sites.

#define TILE_DIM 32

__shared__ float tile_A[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts
__shared__ float tile_B[TILE_DIM][TILE_DIM + 1];

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void mixed_tiled_matmul(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K) {
    int tx = threadIdx.x % TILE_DIM;
    int ty = threadIdx.x / TILE_DIM;
    int row = blockIdx.x * TILE_DIM + ty;
    int col = blockIdx.y * TILE_DIM + tx;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Unrolled K-tiles: each iteration has 2 VMEM loads + 2 LDS writes + barrier + LDS reads + VALU
    #pragma unroll 4
    for (int t = 0; t < K; t += TILE_DIM) {
        // VMEM: load tile from A (global_load)
        if (row < M && (t + tx) < K)
            tile_A[ty][tx] = A[row * K + t + tx];  // 1 global_load
        else
            tile_A[ty][tx] = 0.0f;

        // VMEM: load tile from B (global_load)
        if ((t + ty) < K && col < N)
            tile_B[ty][tx] = B[(t + ty) * N + col];  // 1 global_load
        else
            tile_B[ty][tx] = 0.0f;

        __syncthreads();

        // LDS: 32 reads from tile_A + 32 reads from tile_B per iteration
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val = tile_A[ty][k];    // ds_read
            float b_val = tile_B[k][tx];    // ds_read
            acc0 += a_val * b_val;
        }

        __syncthreads();
    }

    // VMEM: store result (global_store)
    if (row < M && col < N)
        C[row * N + col] = acc0 + acc1 + acc2 + acc3;
}

// Second kernel: scattered mixed operations
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void mixed_scatter_lds(const float* __restrict__ input,
                       const int* __restrict__ indices,
                       float* __restrict__ output,
                       int N, int num_indices) {
    __shared__ float smem[1024];

    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;

    // Phase 1: Global gather -> LDS (VMEM loads then LDS writes)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = indices[(gtid * 4 + i) % num_indices] % N;
        smem[(tid * 4 + i) % 1024] = input[idx];  // 1 VMEM + 1 LDS write each
    }
    __syncthreads();

    // Phase 2: LDS read-modify-write (pure LDS)
    float vals[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        vals[i] = smem[(tid * 8 + i) % 1024];  // 8 LDS reads
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        smem[(tid * 8 + i + 128) % 1024] = vals[i] * 2.0f;  // 8 LDS writes
    }
    __syncthreads();

    // Phase 3: LDS -> Global scatter (LDS reads then VMEM stores)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float v = smem[(tid * 4 + i + 256) % 1024];  // 4 LDS reads
        int out_idx = indices[(gtid * 4 + i + num_indices/2) % num_indices] % N;
        output[out_idx] = v;  // 4 VMEM stores
    }
}

int main() {
    const int M = 512, N = 512, K = 512;

    float *hA = (float*)malloc(M * K * sizeof(float));
    float *hB = (float*)malloc(K * N * sizeof(float));
    float *hC = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(rand() % 100) / 100.0f;

    float *dA, *dB, *dC;
    hipMalloc(&dA, M * K * sizeof(float));
    hipMalloc(&dB, K * N * sizeof(float));
    hipMalloc(&dC, M * N * sizeof(float));
    hipMemcpy(dA, hA, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, K * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    mixed_tiled_matmul<<<grid, 256>>>(dA, dB, dC, M, N, K);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("FAIL: matmul GPU error: %s\n", hipGetErrorString(err));
        return 1;
    }

    hipMemcpy(hC, dC, M * N * sizeof(float), hipMemcpyDeviceToHost);
    double matmul_sum = 0.0;
    for (int i = 0; i < M * N; i++) matmul_sum += hC[i];
    printf("CHECKSUM_MATMUL: sum=%.6f first=%.6f\n", matmul_sum, hC[0]);

    // Scatter kernel
    const int num_idx = 16384;
    int *hIdx = (int*)malloc(num_idx * 128 * sizeof(int));
    for (int i = 0; i < num_idx * 128; i++) hIdx[i] = rand() % (M * N);

    float *dInput, *dOutput;
    int *dIdx;
    hipMalloc(&dInput, M * N * sizeof(float));
    hipMalloc(&dOutput, M * N * sizeof(float));
    hipMalloc(&dIdx, num_idx * 128 * sizeof(int));
    hipMemcpy(dInput, hC, M * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dIdx, hIdx, num_idx * 128 * sizeof(int), hipMemcpyHostToDevice);

    mixed_scatter_lds<<<num_idx / 256, 256>>>(dInput, dIdx, dOutput, M * N, num_idx);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("FAIL: scatter GPU error: %s\n", hipGetErrorString(err));
        return 1;
    }

    printf("PASS\n");

    hipFree(dA); hipFree(dB); hipFree(dC);
    hipFree(dInput); hipFree(dOutput); hipFree(dIdx);
    free(hA); free(hB); free(hC); free(hIdx);
    return 0;
}
