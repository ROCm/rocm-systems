#include <hip/hip_runtime.h>

__global__ void copy_kernel(float* out, const float* in) {
    out[threadIdx.x] = in[threadIdx.x];
}

int main() {
    void* dst = nullptr;
    void* src = nullptr;
    hipMalloc(&dst, 4096);
    hipMemcpy(dst, src, 4096, hipMemcpyHostToDevice);
    hipMemcpy(src, dst, 4096, hipMemcpyDeviceToHost);
    hipLaunchKernelGGL(copy_kernel, dim3(1), dim3(64), 0, 0, (float*)dst, (const float*)src);
    return 0;
}
