// Simple test to see how function pointer tables look in GPU assembly/ELF
#include <hip/hip_runtime.h>
#include <stdio.h>

// Two simple device functions
__device__ void func_a(int* out) {
    *out = 42;
}

__device__ void func_b(int* out) {
    *out = 99;
}

// Function pointer type
typedef void (*DevFunc)(int*);

// Global table of function pointers
__device__ DevFunc funcTable[2] = { func_a, func_b };

// Kernel that uses the table
__global__ void testKernel(int* out, int funcId) {
    funcTable[funcId](out);
}

int main() {
    int* d_out;
    hipMalloc(&d_out, sizeof(int));
    
    testKernel<<<1, 1>>>(d_out, 0);
    
    int result;
    hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost);
    printf("Result: %d (expected 42)\n", result);
    
    testKernel<<<1, 1>>>(d_out, 1);
    hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost);
    printf("Result: %d (expected 99)\n", result);
    
    hipFree(d_out);
    return 0;
}
