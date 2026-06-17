#include <hip/hip_runtime.h>

__global__ void sync_heavy(float* data) {
    data[threadIdx.x] = (float)threadIdx.x;
}

int main() {
    float* data = nullptr;
    hipMalloc(&data, 4096);
    hipLaunchKernelGGL(sync_heavy, dim3(1), dim3(64), 0, 0, data);
    hipDeviceSynchronize();
    hipDeviceSynchronize();
    hipDeviceSynchronize();
    hipDeviceSynchronize();
    hipDeviceSynchronize();
    hipDeviceSynchronize();
    return 0;
}
