#include <hip/hip_runtime.h>

extern "C" {

__global__ void MultiplyKernel(float* x, float* y, float* z) {
  size_t i = threadIdx.x;
  z[i] = x[i] * y[i];
}

}

//  hipcc --genco MultiplyKernel.cpp -o MultiplyKernel.code


