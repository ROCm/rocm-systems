#include <hip/hip_runtime.h>

__device__ void
barrier_wrapper(int* data)
{
    __syncthreads();
    data[threadIdx.x] += data[threadIdx.x ^ 1];
}

__global__ void
syncthreads_kernel(int* data)
{
    data[threadIdx.x] = threadIdx.x;
    barrier_wrapper(data);
}
