#include <hip/hip_runtime.h>
#include <cstdio>

// Kernel designed to produce 2000+ distinct memory instruction sites.
// Uses moderate unroll (64 iterations) across multiple independent gather passes,
// keeping compile time reasonable while maximizing VMEM instruction count.
// 8 passes × 64 iters × 4 loads = 2048 load sites + stores.
//
// KNOWN ISSUE: Full instrumentation (2080/2080 sites) succeeds but the
//   patched kernel dispatch hangs — never completes on GPU. The signal
//   monitor thread waits forever; process must be killed externally.
//   1.5MB patched ELF, 512 VGPRs, accum_offset=256, scratch spill at
//   offset 8064. Likely corrupted instruction or infinite loop in
//   trampoline for one of the 2080 sites.

#define DO_PASS(arr, acc, base)                  \
    _Pragma("unroll")                            \
    for (int i = 0; i < 64; i++) {               \
        acc += arr[tid + (base + i) * stride];   \
    }

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void mega_gather(const float* __restrict__ A,
                 const float* __restrict__ B,
                 const float* __restrict__ C,
                 const float* __restrict__ D,
                 float* __restrict__ out,
                 int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    float b4 = 0, b5 = 0, b6 = 0, b7 = 0;

    // Pass 1-8 on array A: 8 × 64 = 512 load sites
    DO_PASS(A, a0,   0) DO_PASS(A, a1,  64) DO_PASS(A, a2, 128) DO_PASS(A, a3, 192)
    DO_PASS(A, a4, 256) DO_PASS(A, a5, 320) DO_PASS(A, a6, 384) DO_PASS(A, a7, 448)

    // Pass 1-8 on array B: 8 × 64 = 512 load sites
    DO_PASS(B, b0,   0) DO_PASS(B, b1,  64) DO_PASS(B, b2, 128) DO_PASS(B, b3, 192)
    DO_PASS(B, b4, 256) DO_PASS(B, b5, 320) DO_PASS(B, b6, 384) DO_PASS(B, b7, 448)

    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    float c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    float d4 = 0, d5 = 0, d6 = 0, d7 = 0;

    // Pass 1-8 on array C: 8 × 64 = 512 load sites
    DO_PASS(C, c0,   0) DO_PASS(C, c1,  64) DO_PASS(C, c2, 128) DO_PASS(C, c3, 192)
    DO_PASS(C, c4, 256) DO_PASS(C, c5, 320) DO_PASS(C, c6, 384) DO_PASS(C, c7, 448)

    // Pass 1-8 on array D: 8 × 64 = 512 load sites
    DO_PASS(D, d0,   0) DO_PASS(D, d1,  64) DO_PASS(D, d2, 128) DO_PASS(D, d3, 192)
    DO_PASS(D, d4, 256) DO_PASS(D, d5, 320) DO_PASS(D, d6, 384) DO_PASS(D, d7, 448)

    // Total: 2048 load sites

    float total = a0+a1+a2+a3+a4+a5+a6+a7 + b0+b1+b2+b3+b4+b5+b6+b7 +
                  c0+c1+c2+c3+c4+c5+c6+c7 + d0+d1+d2+d3+d4+d5+d6+d7;

    // 32 store sites
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        out[tid + i * stride] = total + (float)i;
    }
}

int main() {
    float *dA, *dB, *dC, *dD, *dOut;
    size_t N = (size_t)256 * 512 * 256;
    hipMalloc(&dA, N * sizeof(float));
    hipMalloc(&dB, N * sizeof(float));
    hipMalloc(&dC, N * sizeof(float));
    hipMalloc(&dD, N * sizeof(float));
    hipMalloc(&dOut, N * sizeof(float));

    mega_gather<<<256, 256>>>(dA, dB, dC, dD, dOut, 256);
    hipDeviceSynchronize();

    printf("Kernel launched OK\n");

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dD); hipFree(dOut);
    return 0;
}
