#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Multi-kernel binary with mixed VMEM+LDS in each kernel.
// Tests: (a) multiple kernels in one code object, (b) interleaved
// global and LDS memory ops, (c) different register pressure profiles
// per kernel. Each kernel validates its output.
//
// KNOWN ISSUE: tiled_reduce crashes with "illegal memory access" under
//   instrumentation. Fully instrumented (21/21 sites: 2 VMEM + 19 LDS,
//   21 BBs, 572 bytes). 3 extra VGPRs above-the-count. Likely LDS
//   address/register clobbering by trampoline in the tree reduction
//   pattern with conditional `if (tid < s)` basic blocks.
//   Remaining 3 kernels (transpose, histogram, prefix_sum) never reached.

// Kernel 1: Tiled reduction — moderate VMEM + moderate LDS
__shared__ float smem1[256];

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void tiled_reduce(const float* __restrict__ input,
                  float* __restrict__ output,
                  int N) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // VMEM load
    smem1[tid] = (gtid < N) ? input[gtid] : 0.0f;
    __syncthreads();

    // LDS tree reduction (8 stages = 16 LDS ops per thread path)
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem1[tid] += smem1[tid + s];
        }
        __syncthreads();
    }

    // VMEM store
    if (tid == 0) {
        output[blockIdx.x] = smem1[0];
    }
}

// Kernel 2: Transpose via LDS — balanced VMEM + LDS
#define TILE 32
__shared__ float tile[TILE][TILE + 1];

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void matrix_transpose(const float* __restrict__ in,
                      float* __restrict__ out,
                      int rows, int cols) {
    int tx = threadIdx.x % TILE;
    int ty = threadIdx.x / TILE;
    int x_in = blockIdx.x * TILE + tx;
    int y_in = blockIdx.y * TILE + ty;

    // VMEM loads -> LDS writes (coalesced read)
    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        if (x_in < cols && (y_in + j) < rows)
            tile[ty + j][tx] = in[(y_in + j) * cols + x_in];
    }
    __syncthreads();

    // LDS reads -> VMEM stores (coalesced write in transposed layout)
    int x_out = blockIdx.y * TILE + tx;
    int y_out = blockIdx.x * TILE + ty;
    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        if (x_out < rows && (y_out + j) < cols)
            out[(y_out + j) * rows + x_out] = tile[tx][ty + j];
    }
}

// Kernel 3: Histogram via LDS atomics — VMEM gather + LDS atomic ops
#define NUM_BINS 256
__shared__ int shared_hist[NUM_BINS];

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void histogram(const int* __restrict__ data,
               int* __restrict__ hist,
               int N) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // LDS init
    shared_hist[tid] = 0;
    __syncthreads();

    // VMEM gather + LDS atomic
    if (gtid < N) {
        int bin = data[gtid] % NUM_BINS;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();

    // LDS read + VMEM atomic store
    atomicAdd(&hist[tid], shared_hist[tid]);
}

// Kernel 4: VMEM-heavy prefix sum with LDS scan
__shared__ float scan_buf[512];

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void prefix_sum_block(const float* __restrict__ input,
                      float* __restrict__ output,
                      float* __restrict__ block_sums,
                      int N) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x * 2 + tid;

    // VMEM: load 2 elements per thread
    scan_buf[tid] = (gtid < N) ? input[gtid] : 0.0f;
    scan_buf[tid + 256] = (gtid + 256 < N) ? input[gtid + 256] : 0.0f;
    __syncthreads();

    // LDS: Blelloch scan up-sweep
    #pragma unroll
    for (int d = 1; d < 512; d <<= 1) {
        int idx = (tid + 1) * 2 * d - 1;
        if (idx < 512) {
            scan_buf[idx] += scan_buf[idx - d];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = scan_buf[511];
        scan_buf[511] = 0.0f;
    }
    __syncthreads();

    // LDS: down-sweep
    #pragma unroll
    for (int d = 256; d >= 1; d >>= 1) {
        int idx = (tid + 1) * 2 * d - 1;
        if (idx < 512) {
            float tmp = scan_buf[idx - d];
            scan_buf[idx - d] = scan_buf[idx];
            scan_buf[idx] += tmp;
        }
        __syncthreads();
    }

    // VMEM: store results
    if (gtid < N) output[gtid] = scan_buf[tid];
    if (gtid + 256 < N) output[gtid + 256] = scan_buf[tid + 256];
}

int main() {
    const int N = 256 * 1024;

    // --- Kernel 1: tiled_reduce ---
    float *hIn = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hIn[i] = 1.0f;

    float *dIn, *dOut;
    hipMalloc(&dIn, N * sizeof(float));
    hipMalloc(&dOut, N * sizeof(float));
    hipMemcpy(dIn, hIn, N * sizeof(float), hipMemcpyHostToDevice);

    int blocks = N / 256;
    tiled_reduce<<<blocks, 256>>>(dIn, dOut, N);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: tiled_reduce: %s\n", hipGetErrorString(err)); return 1; }

    float *hOut = (float*)malloc(blocks * sizeof(float));
    hipMemcpy(hOut, dOut, blocks * sizeof(float), hipMemcpyDeviceToHost);
    float reduce_sum = 0;
    for (int i = 0; i < blocks; i++) reduce_sum += hOut[i];
    printf("tiled_reduce: sum=%.0f (expect %.0f) %s\n",
           reduce_sum, (float)N, fabsf(reduce_sum - N) < 1.0f ? "OK" : "MISMATCH");

    // --- Kernel 2: matrix_transpose ---
    const int R = 512, C = 512;
    float *hMat = (float*)malloc(R * C * sizeof(float));
    for (int i = 0; i < R * C; i++) hMat[i] = (float)(i % 100);

    float *dMat, *dTrans;
    hipMalloc(&dMat, R * C * sizeof(float));
    hipMalloc(&dTrans, C * R * sizeof(float));
    hipMemcpy(dMat, hMat, R * C * sizeof(float), hipMemcpyHostToDevice);

    dim3 tgrid((C + TILE - 1) / TILE, (R + TILE - 1) / TILE);
    matrix_transpose<<<tgrid, 256>>>(dMat, dTrans, R, C);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: transpose: %s\n", hipGetErrorString(err)); return 1; }

    float *hTrans = (float*)malloc(C * R * sizeof(float));
    hipMemcpy(hTrans, dTrans, C * R * sizeof(float), hipMemcpyDeviceToHost);
    int t_errs = 0;
    for (int i = 0; i < R && t_errs < 5; i++)
        for (int j = 0; j < C && t_errs < 5; j++)
            if (hTrans[j * R + i] != hMat[i * C + j]) t_errs++;
    printf("transpose: errors=%d %s\n", t_errs, t_errs == 0 ? "OK" : "MISMATCH");

    // --- Kernel 3: histogram ---
    int *hData = (int*)malloc(N * sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) hData[i] = rand();

    int *dData, *dHist;
    hipMalloc(&dData, N * sizeof(int));
    hipMalloc(&dHist, NUM_BINS * sizeof(int));
    hipMemset(dHist, 0, NUM_BINS * sizeof(int));
    hipMemcpy(dData, hData, N * sizeof(int), hipMemcpyHostToDevice);

    histogram<<<N / 256, 256>>>(dData, dHist, N);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: histogram: %s\n", hipGetErrorString(err)); return 1; }

    int hHist[NUM_BINS];
    hipMemcpy(hHist, dHist, NUM_BINS * sizeof(int), hipMemcpyDeviceToHost);
    int hist_sum = 0;
    for (int i = 0; i < NUM_BINS; i++) hist_sum += hHist[i];
    printf("histogram: total=%d (expect %d) %s\n",
           hist_sum, N, hist_sum == N ? "OK" : "MISMATCH");

    // --- Kernel 4: prefix_sum ---
    float *hScan = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hScan[i] = 1.0f;

    float *dScanIn, *dScanOut, *dBlockSums;
    hipMalloc(&dScanIn, N * sizeof(float));
    hipMalloc(&dScanOut, N * sizeof(float));
    hipMalloc(&dBlockSums, (N / 512 + 1) * sizeof(float));
    hipMemcpy(dScanIn, hScan, N * sizeof(float), hipMemcpyHostToDevice);

    prefix_sum_block<<<N / 512, 256>>>(dScanIn, dScanOut, dBlockSums, N);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: prefix_sum: %s\n", hipGetErrorString(err)); return 1; }

    float *hScanOut = (float*)malloc(N * sizeof(float));
    hipMemcpy(hScanOut, dScanOut, N * sizeof(float), hipMemcpyDeviceToHost);
    printf("prefix_sum: first_block_last=%.0f (expect 511) %s\n",
           hScanOut[511], fabsf(hScanOut[511] - 511.0f) < 1.0f ? "OK" : "MISMATCH");

    int all_ok = (fabsf(reduce_sum - N) < 1.0f) && (t_errs == 0)
                 && (hist_sum == N) && (fabsf(hScanOut[511] - 511.0f) < 1.0f);
    printf(all_ok ? "PASS\n" : "FAIL\n");

    hipFree(dIn); hipFree(dOut); hipFree(dMat); hipFree(dTrans);
    hipFree(dData); hipFree(dHist); hipFree(dScanIn); hipFree(dScanOut); hipFree(dBlockSums);
    free(hIn); free(hOut); free(hMat); free(hTrans); free(hData); free(hScan); free(hScanOut);
    return all_ok ? 0 : 1;
}
