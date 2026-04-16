#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Heavily unrolled tiled GEMM designed to produce 2000+ memory sites.
// Each unrolled iteration of the inner loop produces multiple global loads
// and the outer accumulation produces stores.
//
// Tile: 64x64, K-block: 8, inner loop unrolled 256 times -> ~2000+ VMEM ops
//
// KNOWN ISSUE: GPU hang on big_gemm kernel under instrumentation.
//   321/321 VMEM sites fully instrumented (no partial), but the patched
//   dispatch never completes. 39 BBs, 288 VGPRs (accum_offset=256),
//   scratch spill. Likely register clobbering or corrupted displaced
//   instruction across basic block boundaries with conditional control flow.
//   scatter_gather kernel never reached.

#define TILE_M 64
#define TILE_N 64
#define TILE_K 8

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void big_gemm(const float* __restrict__ A,
              const float* __restrict__ B,
              float* __restrict__ C,
              int M, int N, int K) {
    int tx = threadIdx.x % TILE_N;
    int ty = threadIdx.x / TILE_N;
    int bx = blockIdx.x * TILE_M;
    int by = blockIdx.y * TILE_N;

    int row = bx + ty;
    int col = by + tx;

    if (row >= M || col >= N) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Main K-loop: each iteration loads from A and B, accumulates.
    // With #pragma unroll, each iteration becomes distinct VMEM instructions.
    #pragma unroll 1
    for (int k_outer = 0; k_outer < K; k_outer += TILE_K * 32) {
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_K * 32; k_inner += TILE_K) {
            int k = k_outer + k_inner;
            if (k >= K) break;

            // Each of these loads is a separate memory site
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk++) {
                float a_val = A[row * K + k + kk];
                float b_val = B[(k + kk) * N + col];
                acc[kk % 4] += a_val * b_val;
            }
        }
    }

    C[row * N + col] = acc[0] + acc[1] + acc[2] + acc[3];
}

// Even more memory-heavy: a kernel that explicitly does many scattered loads
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void scatter_gather(const float* __restrict__ src,
                    const int* __restrict__ indices,
                    float* __restrict__ dst,
                    int num_indices) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_indices) return;

    float sum = 0.0f;

    // Each unrolled iteration is a separate global load site
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        int idx = indices[tid * 128 + i];
        sum += src[idx];
    }

    // Multiple stores to different locations
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        dst[tid * 16 + i] = sum * (float)(i + 1);
    }
}

int main() {
    // Compile-only: we just need the code object.
    // The aegisbit tool will extract and analyze the kernels.
    int M = 1024, N = 1024, K = 2048;
    float *dA, *dB, *dC;
    hipMalloc(&dA, M * K * sizeof(float));
    hipMalloc(&dB, K * N * sizeof(float));
    hipMalloc(&dC, M * N * sizeof(float));

    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 block(256);
    big_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    hipDeviceSynchronize();

    int *dIdx;
    float *dSrc, *dDst;
    int num = 4096;
    hipMalloc(&dSrc, 1024 * 1024 * sizeof(float));
    hipMalloc(&dIdx, num * 128 * sizeof(int));
    hipMalloc(&dDst, num * 16 * sizeof(float));

    scatter_gather<<<(num + 255) / 256, 256>>>(dSrc, dIdx, dDst, num);
    hipDeviceSynchronize();

    printf("Kernels launched OK\n");

    hipFree(dA); hipFree(dB); hipFree(dC);
    hipFree(dSrc); hipFree(dIdx); hipFree(dDst);
    return 0;
}
