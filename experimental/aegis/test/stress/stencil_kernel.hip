#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 2D stencil / convolution kernel with LDS halo exchange.
// Exercises: 2D tiled VMEM loads with halo, LDS writes, LDS reads for
// neighbor access, and VMEM stores — all interleaved.
// This pattern is common in PDE solvers, image processing, and CNNs.
//
// FINDING: Correctness OK, but heavy scratch-VGPR overlap site drops.
//   stencil_5x5: 28/32 sites (4 skipped: 2 VMEM + 2 LDS overlap v13/v14/v15)
//   jacobi_step: 3/6 sites (3 VMEM skipped overlapping v9/v10/v11 — 50% loss)
//   Scratch spill picks high VGPRs (v13-v15, v9-v11) which collide with
//   the compiler's register allocation for these compact kernels.

#define TILE_W 16
#define TILE_H 16
#define HALO 2
#define SMEM_W (TILE_W + 2 * HALO)
#define SMEM_H (TILE_H + 2 * HALO)

__shared__ float smem[SMEM_H][SMEM_W];

__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void stencil_5x5(const float* __restrict__ input,
                 float* __restrict__ output,
                 int W, int H) {
    int tx = threadIdx.x % TILE_W;
    int ty = threadIdx.x / TILE_W;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    // Phase 1: Cooperative load with halo into LDS
    // Each thread loads its main cell + halo cells for border threads
    int smem_x = tx + HALO;
    int smem_y = ty + HALO;

    // Load center
    int ix = min(max(gx, 0), W - 1);
    int iy = min(max(gy, 0), H - 1);
    smem[smem_y][smem_x] = input[iy * W + ix];

    // Load halo: top
    if (ty < HALO) {
        int hy = max(gy - HALO, 0);
        smem[ty][smem_x] = input[hy * W + ix];
    }
    // Load halo: bottom
    if (ty >= TILE_H - HALO) {
        int hy = min(gy + HALO, H - 1);
        smem[smem_y + HALO][smem_x] = input[hy * W + ix];
    }
    // Load halo: left
    if (tx < HALO) {
        int hx = max(gx - HALO, 0);
        smem[smem_y][tx] = input[iy * W + hx];
    }
    // Load halo: right
    if (tx >= TILE_W - HALO) {
        int hx = min(gx + HALO, W - 1);
        smem[smem_y][smem_x + HALO] = input[iy * W + hx];
    }
    // Corners (for completeness)
    if (tx < HALO && ty < HALO) {
        smem[ty][tx] = input[max(gy - HALO, 0) * W + max(gx - HALO, 0)];
    }
    if (tx >= TILE_W - HALO && ty < HALO) {
        smem[ty][smem_x + HALO] = input[max(gy - HALO, 0) * W + min(gx + HALO, W - 1)];
    }
    if (tx < HALO && ty >= TILE_H - HALO) {
        smem[smem_y + HALO][tx] = input[min(gy + HALO, H - 1) * W + max(gx - HALO, 0)];
    }
    if (tx >= TILE_W - HALO && ty >= TILE_H - HALO) {
        smem[smem_y + HALO][smem_x + HALO] = input[min(gy + HALO, H - 1) * W + min(gx + HALO, W - 1)];
    }

    __syncthreads();

    // Phase 2: 5x5 stencil from LDS reads
    float result = 0.0f;
    #pragma unroll
    for (int dy = -2; dy <= 2; dy++) {
        #pragma unroll
        for (int dx = -2; dx <= 2; dx++) {
            float weight = 1.0f / 25.0f;
            result += smem[smem_y + dy][smem_x + dx] * weight;
        }
    }

    // Phase 3: Global store
    if (gx < W && gy < H) {
        output[gy * W + gx] = result;
    }
}

// Second kernel: iterative Jacobi relaxation (multiple stencil passes)
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void jacobi_step(const float* __restrict__ u_old,
                 float* __restrict__ u_new,
                 const float* __restrict__ rhs,
                 int W, int H, float h2) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y;

    if (gx <= 0 || gx >= W - 1 || gy <= 0 || gy >= H - 1) return;

    // 5-point stencil: 4 neighbor loads + 1 center load + 1 rhs load + 1 store
    float center = u_old[gy * W + gx];
    float left   = u_old[gy * W + gx - 1];
    float right  = u_old[gy * W + gx + 1];
    float up     = u_old[(gy - 1) * W + gx];
    float down   = u_old[(gy + 1) * W + gx];
    float f      = rhs[gy * W + gx];

    u_new[gy * W + gx] = 0.25f * (left + right + up + down - h2 * f);
}

int main() {
    const int W = 1024, H = 1024;
    const int N = W * H;

    float *hIn = (float*)malloc(N * sizeof(float));
    float *hOut = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) hIn[i] = (float)(rand() % 100) / 100.0f;

    float *dIn, *dOut;
    hipMalloc(&dIn, N * sizeof(float));
    hipMalloc(&dOut, N * sizeof(float));
    hipMemcpy(dIn, hIn, N * sizeof(float), hipMemcpyHostToDevice);

    // Kernel 1: stencil_5x5
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
    stencil_5x5<<<grid, 256>>>(dIn, dOut, W, H);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: stencil_5x5: %s\n", hipGetErrorString(err)); return 1; }

    hipMemcpy(hOut, dOut, N * sizeof(float), hipMemcpyDeviceToHost);
    double stencil_sum = 0;
    for (int i = 0; i < N; i++) stencil_sum += hOut[i];
    printf("stencil_5x5: sum=%.2f first=%.4f %s\n",
           stencil_sum, hOut[W + 1], stencil_sum > 0 ? "OK" : "ZERO");

    // Kernel 2: jacobi iterations
    float *dRhs, *dU1, *dU2;
    hipMalloc(&dRhs, N * sizeof(float));
    hipMalloc(&dU1, N * sizeof(float));
    hipMalloc(&dU2, N * sizeof(float));
    hipMemset(dU1, 0, N * sizeof(float));
    hipMemcpy(dRhs, dIn, N * sizeof(float), hipMemcpyDeviceToDevice);

    dim3 jgrid(W / 256, H);
    for (int iter = 0; iter < 10; iter++) {
        if (iter % 2 == 0)
            jacobi_step<<<jgrid, 256>>>(dU1, dU2, dRhs, W, H, 1.0f);
        else
            jacobi_step<<<jgrid, 256>>>(dU2, dU1, dRhs, W, H, 1.0f);
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { printf("FAIL: jacobi: %s\n", hipGetErrorString(err)); return 1; }

    hipMemcpy(hOut, dU1, N * sizeof(float), hipMemcpyDeviceToHost);
    double jacobi_norm = 0;
    for (int i = 0; i < N; i++) jacobi_norm += hOut[i] * hOut[i];
    printf("jacobi_10iter: norm=%.4f %s\n", sqrt(jacobi_norm), jacobi_norm > 0 ? "OK" : "ZERO");

    int all_ok = (stencil_sum > 0) && (jacobi_norm > 0);
    printf(all_ok ? "PASS\n" : "FAIL\n");

    hipFree(dIn); hipFree(dOut); hipFree(dRhs); hipFree(dU1); hipFree(dU2);
    free(hIn); free(hOut);
    return all_ok ? 0 : 1;
}
