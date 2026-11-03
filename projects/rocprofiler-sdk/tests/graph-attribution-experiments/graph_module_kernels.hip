#include <hip/hip_runtime.h>
extern "C" __global__ void module_add(int* x, int value) { if(threadIdx.x == 0) *x += value; }
extern "C" __global__ void module_mul(int* x, int value) { if(threadIdx.x == 0) *x *= value; }
