#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Wide vector load/store kernel using float4.
// Each float4 load/store is a single GLOBAL_LOAD_DWORDX4 / GLOBAL_STORE_DWORDX4
// instruction — tests that SiteAnalyzer handles wide VMEM ops correctly
// and that displaced instructions for 8-byte encodings work.
// Also exercises: strided access patterns, reduction across vector elements.
//
// FINDING: Correctness OK under instrumentation, but scratch-VGPR overlap
//   silently drops sites in SiteAnalyzer — wide_vector_fma has 4 actual
//   VMEM ops but only 2 are reported (2× GLOBAL_LOAD_DWORDX4 skipped
//   because dst VGPRs overlap scratch v17/v18/v19). mixed_width_loads
//   similarly loses 1 site. These aren't reported as "partial" — the
//   discovered count already excludes them.

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void wide_vector_fma(const float4* __restrict__ A,
                     const float4* __restrict__ B,
                     const float4* __restrict__ C,
                     float4* __restrict__ out,
                     int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // 3 wide loads (GLOBAL_LOAD_DWORDX4)
    float4 a = A[tid];
    float4 b = B[tid];
    float4 c = C[tid];

    // FMA
    float4 r;
    r.x = a.x * b.x + c.x;
    r.y = a.y * b.y + c.y;
    r.z = a.z * b.z + c.z;
    r.w = a.w * b.w + c.w;

    // 1 wide store (GLOBAL_STORE_DWORDX4)
    out[tid] = r;
}

// Multiple passes with wide loads at different strides
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void wide_gather_multi(const float4* __restrict__ data,
                       float* __restrict__ out,
                       int N, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float sum = 0.0f;

    // 16 wide loads at different strides — each a GLOBAL_LOAD_DWORDX4
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = (tid + i * stride) % N;
        float4 v = data[idx];
        sum += v.x + v.y + v.z + v.w;
    }

    out[tid] = sum;
}

// Mix of float, float2, float4 loads — different instruction widths
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void mixed_width_loads(const float* __restrict__ f1,
                       const float2* __restrict__ f2,
                       const float4* __restrict__ f4,
                       float* __restrict__ out,
                       int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float sum = 0.0f;

    // 8 scalar loads (GLOBAL_LOAD_DWORD)
    #pragma unroll
    for (int i = 0; i < 8; i++)
        sum += f1[(tid + i * N / 8) % N];

    // 8 double loads (GLOBAL_LOAD_DWORDX2)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float2 v = f2[(tid + i * N / 8) % N];
        sum += v.x + v.y;
    }

    // 8 quad loads (GLOBAL_LOAD_DWORDX4)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float4 v = f4[(tid + i * N / 8) % N];
        sum += v.x + v.y + v.z + v.w;
    }

    out[tid] = sum;
}

int main() {
    const int N = 256 * 1024;

    // --- Kernel 1: wide_vector_fma ---
    float4 *hA = (float4*)malloc(N * sizeof(float4));
    float4 *hB = (float4*)malloc(N * sizeof(float4));
    float4 *hC = (float4*)malloc(N * sizeof(float4));
    float4 *hOut = (float4*)malloc(N * sizeof(float4));

    for (int i = 0; i < N; i++) {
        hA[i] = {1.0f, 2.0f, 3.0f, 4.0f};
        hB[i] = {2.0f, 2.0f, 2.0f, 2.0f};
        hC[i] = {0.5f, 0.5f, 0.5f, 0.5f};
    }

    float4 *dA, *dB, *dC, *dOut4;
    hipMalloc(&dA, N * sizeof(float4));
    hipMalloc(&dB, N * sizeof(float4));
    hipMalloc(&dC, N * sizeof(float4));
    hipMalloc(&dOut4, N * sizeof(float4));
    hipMemcpy(dA, hA, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, N * sizeof(float4), hipMemcpyHostToDevice);
    hipMemcpy(dC, hC, N * sizeof(float4), hipMemcpyHostToDevice);

    wide_vector_fma<<<N / 256, 256>>>(dA, dB, dC, dOut4, N);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: wide_fma: %s\n", hipGetErrorString(err)); return 1; }

    hipMemcpy(hOut, dOut4, N * sizeof(float4), hipMemcpyDeviceToHost);
    int fma_ok = (fabsf(hOut[0].x - 2.5f) < 0.01f && fabsf(hOut[0].y - 4.5f) < 0.01f &&
                  fabsf(hOut[0].z - 6.5f) < 0.01f && fabsf(hOut[0].w - 8.5f) < 0.01f);
    printf("wide_fma: [%.1f,%.1f,%.1f,%.1f] expect [2.5,4.5,6.5,8.5] %s\n",
           hOut[0].x, hOut[0].y, hOut[0].z, hOut[0].w, fma_ok ? "OK" : "MISMATCH");

    // --- Kernel 2: wide_gather_multi ---
    float *hScalar = (float*)malloc(N * sizeof(float));
    float *dData, *dScalar;
    hipMalloc(&dData, N * sizeof(float4));
    hipMalloc(&dScalar, N * sizeof(float));

    wide_gather_multi<<<N / 256, 256>>>((float4*)dA, dScalar, N, 1024);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: wide_gather: %s\n", hipGetErrorString(err)); return 1; }

    hipMemcpy(hScalar, dScalar, N * sizeof(float), hipMemcpyDeviceToHost);
    printf("wide_gather: first=%.1f %s\n", hScalar[0], hScalar[0] != 0.0f ? "OK" : "ZERO");

    // --- Kernel 3: mixed_width_loads ---
    float *dF1;
    float2 *dF2;
    float4 *dF4;
    float *dMixOut;
    hipMalloc(&dF1, N * sizeof(float));
    hipMalloc(&dF2, N * sizeof(float2));
    hipMalloc(&dF4, N * sizeof(float4));
    hipMalloc(&dMixOut, N * sizeof(float));

    // Init with known values
    float *hF1 = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) hF1[i] = 1.0f;
    hipMemcpy(dF1, hF1, N * sizeof(float), hipMemcpyHostToDevice);
    float2 *hF2 = (float2*)malloc(N * sizeof(float2));
    for (int i = 0; i < N; i++) hF2[i] = {1.0f, 1.0f};
    hipMemcpy(dF2, hF2, N * sizeof(float2), hipMemcpyHostToDevice);
    hipMemcpy(dF4, hA, N * sizeof(float4), hipMemcpyHostToDevice);

    mixed_width_loads<<<N / 256, 256>>>(dF1, dF2, dF4, dMixOut, N);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: mixed_width: %s\n", hipGetErrorString(err)); return 1; }

    float *hMixOut = (float*)malloc(N * sizeof(float));
    hipMemcpy(hMixOut, dMixOut, N * sizeof(float), hipMemcpyDeviceToHost);
    printf("mixed_width: first=%.1f %s\n", hMixOut[0], hMixOut[0] != 0.0f ? "OK" : "ZERO");

    int all_ok = fma_ok && (hScalar[0] != 0.0f) && (hMixOut[0] != 0.0f);
    printf(all_ok ? "PASS\n" : "FAIL\n");

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dOut4);
    hipFree(dData); hipFree(dScalar);
    hipFree(dF1); hipFree(dF2); hipFree(dF4); hipFree(dMixOut);
    free(hA); free(hB); free(hC); free(hOut);
    free(hScalar); free(hF1); free(hF2); free(hMixOut);
    return all_ok ? 0 : 1;
}
