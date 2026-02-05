#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            fflush(stderr); \
            exit(1); \
        } \
    } while(0)

// Baseline: Many scattered VMEM ops fill command FIFO
__global__ void vmem_stall_baseline(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float vals[16];
    for (int i = 0; i < 16; i++) {
        int load_idx = (idx + i * 1024) % N;
        vals[i] = data[load_idx];
    }

    for (int i = 0; i < 16; i++) {
        int store_idx = (idx + i * 2048) % N;
        data[store_idx] = vals[i] * 2.0f;
    }
}

// Optimized: Shared memory reduces VMEM FIFO pressure
__global__ void vmem_stall_optimized(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    __shared__ float sdata[256];

    sdata[threadIdx.x] = data[idx];
    __syncthreads();

    float result = sdata[threadIdx.x];
    for (int i = 0; i < 16; i++) {
        result = result * 1.01f + 0.1f;
    }

    data[idx] = result;
}

int main(int argc, char** argv) {
    bool use_optimized = (argc > 1 && std::string(argv[1]) == "opt");

    const int N = 256 * 1024 * 1024;  // 1GB
    const size_t bytes = N * sizeof(float);

    float* d_data;
    CHECK_HIP(hipMalloc(&d_data, bytes));

    // Host init + copy (no hipMemset)
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    delete[] h_data;

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    if (use_optimized) {
        hipLaunchKernelGGL(vmem_stall_optimized, grid, block, 0, 0, d_data, N);
    } else {
        hipLaunchKernelGGL(vmem_stall_baseline, grid, block, 0, 0, d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    if (use_optimized) {
        hipLaunchKernelGGL(vmem_stall_optimized, grid, block, 0, 0, d_data, N);
    } else {
        hipLaunchKernelGGL(vmem_stall_baseline, grid, block, 0, 0, d_data, N);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << (use_optimized ? "Optimized" : "Baseline") << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    CHECK_HIP(hipFree(d_data));
    return 0;
}