#include <hip/hip_runtime.h>
#include <stdio.h>

// Declare the kernel from librccl
extern "C" void ncclDevKernel_Generic_1(void* args);

int main() {
    printf("Test: Direct kernel launch without NCCL\n"); fflush(stdout);
    
    // Initialize both devices
    printf("Init device 0...\n"); fflush(stdout);
    hipSetDevice(0);
    float *d0;
    hipMalloc(&d0, 4096);
    printf("Init device 1...\n"); fflush(stdout);
    hipSetDevice(1);
    float *d1;
    hipMalloc(&d1, 4096);
    printf("Both devices ready\n"); fflush(stdout);
    
    // Now try to get the kernel function address
    printf("Getting kernel function...\n"); fflush(stdout);
    hipFunction_t fn;
    hipError_t err = hipGetFuncBySymbol(&fn, (void*)ncclDevKernel_Generic_1);
    printf("hipGetFuncBySymbol returned: %d (%s)\n", err, hipGetErrorString(err)); fflush(stdout);
    
    printf("Done\n");
    return 0;
}
