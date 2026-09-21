#include <hip/hip_runtime.h>

// Builds a deterministic, deeply-nested inlined call stack for the DWARF
// inline-annotation test: kernel -> barrier_wrapper -> accumulate -> combine.
__device__ __attribute__((always_inline)) int
combine(const int* data, int i)
{
    return data[i] + data[i ^ 1];
}

__device__ __attribute__((always_inline)) int
accumulate(const int* data, int i)
{
    return combine(data, i) + combine(data, i ^ 1);
}

__device__ __attribute__((always_inline)) void
barrier_wrapper(int* data, int i)
{
    __syncthreads();
    data[i] += accumulate(data, i);
}

__global__ void
syncthreads_kernel(int* data)
{
    int i   = threadIdx.x;
    data[i] = i;
    barrier_wrapper(data, i);
}
