#include <hip/hip_runtime.h>

__global__ void default_stream_only(float* out) {
    out[threadIdx.x] = (float)(threadIdx.x * 2);
}

int main() {
    float* out = nullptr;
    hipMalloc(&out, 4096);
    hipLaunchKernelGGL(default_stream_only, dim3(256), dim3(64), 0, 0, out);
    return 0;
}
