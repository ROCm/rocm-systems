#include <hip/hip_runtime.h>

extern "C" __global__ void hello_kernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from thread %d (block %d, lane %d)\n",
           tid, blockIdx.x, threadIdx.x);
}
