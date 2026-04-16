#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// LDS-heavy kernel: hundreds of ds_read/ds_write instructions.
// Tests that SiteAnalyzer correctly identifies all LDS ops and that
// trampolines handle LDS-category instructions without clobbering.
//
// KNOWN ISSUE: Segfault (exit 139) during LDS Bank Conflicts report.
//   All 151 sites (15 VMEM + 136 LDS) instrument successfully and the
//   kernel produces correct output, but the process crashes in
//   CoalescingAnalyzer's LDS report after printing "136 sites, 557056 samples".
//   2 VMEM sites skipped due to scratch-VGPR operand overlap (v33/v34/v35).

#define LDS_SIZE 4096

__shared__ float lds_buf[LDS_SIZE];

#define LDS_READ_BLOCK(dst, base) \
    dst[0] = lds_buf[(ltid + base +  0) % LDS_SIZE]; \
    dst[1] = lds_buf[(ltid + base +  1) % LDS_SIZE]; \
    dst[2] = lds_buf[(ltid + base +  2) % LDS_SIZE]; \
    dst[3] = lds_buf[(ltid + base +  3) % LDS_SIZE]; \
    dst[4] = lds_buf[(ltid + base +  4) % LDS_SIZE]; \
    dst[5] = lds_buf[(ltid + base +  5) % LDS_SIZE]; \
    dst[6] = lds_buf[(ltid + base +  6) % LDS_SIZE]; \
    dst[7] = lds_buf[(ltid + base +  7) % LDS_SIZE];

#define LDS_WRITE_BLOCK(src, base) \
    lds_buf[(ltid + base +  0) % LDS_SIZE] = src[0]; \
    lds_buf[(ltid + base +  1) % LDS_SIZE] = src[1]; \
    lds_buf[(ltid + base +  2) % LDS_SIZE] = src[2]; \
    lds_buf[(ltid + base +  3) % LDS_SIZE] = src[3]; \
    lds_buf[(ltid + base +  4) % LDS_SIZE] = src[4]; \
    lds_buf[(ltid + base +  5) % LDS_SIZE] = src[5]; \
    lds_buf[(ltid + base +  6) % LDS_SIZE] = src[6]; \
    lds_buf[(ltid + base +  7) % LDS_SIZE] = src[7];

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void lds_stress(const float* __restrict__ input,
                float* __restrict__ output,
                int N) {
    int ltid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;

    // Phase 1: Load from global into LDS (256 VMEM loads + 256 LDS writes via shared mem)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        lds_buf[(ltid * 16 + i) % LDS_SIZE] = input[(gtid * 16 + i) % N];
    }
    __syncthreads();

    // Phase 2: Repeated LDS read-modify-write passes (main stress)
    float accum[8];

    // Pass 1: Read 8 + Write 8 = 16 LDS ops
    LDS_READ_BLOCK(accum, 0)
    for (int j = 0; j < 8; j++) accum[j] *= 1.01f;
    LDS_WRITE_BLOCK(accum, 256)
    __syncthreads();

    // Pass 2
    LDS_READ_BLOCK(accum, 256)
    for (int j = 0; j < 8; j++) accum[j] += 0.5f;
    LDS_WRITE_BLOCK(accum, 512)
    __syncthreads();

    // Pass 3
    LDS_READ_BLOCK(accum, 512)
    for (int j = 0; j < 8; j++) accum[j] *= 0.99f;
    LDS_WRITE_BLOCK(accum, 768)
    __syncthreads();

    // Pass 4
    LDS_READ_BLOCK(accum, 768)
    for (int j = 0; j < 8; j++) accum[j] += 0.25f;
    LDS_WRITE_BLOCK(accum, 1024)
    __syncthreads();

    // Pass 5
    LDS_READ_BLOCK(accum, 1024)
    for (int j = 0; j < 8; j++) accum[j] *= 1.02f;
    LDS_WRITE_BLOCK(accum, 1280)
    __syncthreads();

    // Pass 6
    LDS_READ_BLOCK(accum, 1280)
    for (int j = 0; j < 8; j++) accum[j] += 0.1f;
    LDS_WRITE_BLOCK(accum, 1536)
    __syncthreads();

    // Pass 7
    LDS_READ_BLOCK(accum, 1536)
    for (int j = 0; j < 8; j++) accum[j] *= 0.98f;
    LDS_WRITE_BLOCK(accum, 1792)
    __syncthreads();

    // Pass 8
    LDS_READ_BLOCK(accum, 1792)
    for (int j = 0; j < 8; j++) accum[j] += 0.3f;
    LDS_WRITE_BLOCK(accum, 2048)
    __syncthreads();

    // Pass 9
    LDS_READ_BLOCK(accum, 2048)
    for (int j = 0; j < 8; j++) accum[j] *= 1.03f;
    LDS_WRITE_BLOCK(accum, 2304)
    __syncthreads();

    // Pass 10
    LDS_READ_BLOCK(accum, 2304)
    for (int j = 0; j < 8; j++) accum[j] += 0.15f;
    LDS_WRITE_BLOCK(accum, 2560)
    __syncthreads();

    // Pass 11
    LDS_READ_BLOCK(accum, 2560)
    for (int j = 0; j < 8; j++) accum[j] *= 0.97f;
    LDS_WRITE_BLOCK(accum, 2816)
    __syncthreads();

    // Pass 12
    LDS_READ_BLOCK(accum, 2816)
    for (int j = 0; j < 8; j++) accum[j] += 0.2f;
    LDS_WRITE_BLOCK(accum, 3072)
    __syncthreads();

    // Pass 13
    LDS_READ_BLOCK(accum, 3072)
    for (int j = 0; j < 8; j++) accum[j] *= 1.04f;
    LDS_WRITE_BLOCK(accum, 3328)
    __syncthreads();

    // Pass 14
    LDS_READ_BLOCK(accum, 3328)
    for (int j = 0; j < 8; j++) accum[j] += 0.35f;
    LDS_WRITE_BLOCK(accum, 3584)
    __syncthreads();

    // Pass 15
    LDS_READ_BLOCK(accum, 3584)
    for (int j = 0; j < 8; j++) accum[j] *= 0.96f;
    LDS_WRITE_BLOCK(accum, 3840)
    __syncthreads();

    // Pass 16: final read
    LDS_READ_BLOCK(accum, 3840)

    // Phase 3: Write back to global
    float sum = 0.0f;
    for (int j = 0; j < 8; j++) sum += accum[j];
    output[gtid] = sum;
}

int main() {
    const int N = 256 * 1024;
    float *hInput = (float*)malloc(N * sizeof(float));
    float *hOutput = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++)
        hInput[i] = (float)(rand() % 1000) / 1000.0f;

    float *dInput, *dOutput;
    hipMalloc(&dInput, N * sizeof(float));
    hipMalloc(&dOutput, N * sizeof(float));
    hipMemcpy(dInput, hInput, N * sizeof(float), hipMemcpyHostToDevice);

    lds_stress<<<N / 256, 256>>>(dInput, dOutput, N);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("FAIL: GPU error: %s\n", hipGetErrorString(err));
        return 1;
    }

    hipMemcpy(hOutput, dOutput, N * sizeof(float), hipMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += hOutput[i];
    printf("CHECKSUM: sum=%.6f first=%.6f last=%.6f\n",
           sum, hOutput[0], hOutput[N-1]);
    printf("PASS\n");

    hipFree(dInput); hipFree(dOutput);
    free(hInput); free(hOutput);
    return 0;
}
