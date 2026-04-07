#include <hip/hip_runtime.h>

__global__ void
syncthreads_kernel(int* data)
{
    data[threadIdx.x] = threadIdx.x;
    __syncthreads();
    data[threadIdx.x] += data[threadIdx.x ^ 1];
}
