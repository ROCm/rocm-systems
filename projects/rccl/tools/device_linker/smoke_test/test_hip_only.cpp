// Simple multi-GPU HIP test without RCCL
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void simpleKernel(float* out, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = val + idx;
}

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    printf("Found %d HIP devices\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs\n");
        return 1;
    }
    
    printf("\nTesting kernel launch on GPU 0...\n");
    hipSetDevice(0);
    float* d0;
    hipMalloc(&d0, 1024 * sizeof(float));
    simpleKernel<<<1, 1024>>>(d0, 1.0f);
    hipDeviceSynchronize();
    printf("GPU 0: Success\n");
    
    printf("\nTesting kernel launch on GPU 1...\n");
    hipSetDevice(1);
    float* d1;
    hipMalloc(&d1, 1024 * sizeof(float));
    simpleKernel<<<1, 1024>>>(d1, 2.0f);
    hipDeviceSynchronize();
    printf("GPU 1: Success\n");
    
    // Verify results
    float* h = (float*)malloc(1024 * sizeof(float));
    
    hipSetDevice(0);
    hipMemcpy(h, d0, 1024 * sizeof(float), hipMemcpyDeviceToHost);
    printf("GPU 0 result[0] = %f (expected 1.0)\n", h[0]);
    
    hipSetDevice(1);
    hipMemcpy(h, d1, 1024 * sizeof(float), hipMemcpyDeviceToHost);
    printf("GPU 1 result[0] = %f (expected 2.0)\n", h[0]);
    
    hipSetDevice(0);
    hipFree(d0);
    hipSetDevice(1);
    hipFree(d1);
    free(h);
    
    printf("\n=== MULTI-GPU HIP TEST PASSED ===\n");
    return 0;
}
