#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Test kernel launch\n");
    
    ncclComm_t comms[2];
    int devList[2] = {0, 1};
    ncclCommInitAll(comms, 2, devList);
    printf("Comms OK\n");
    
    float *d[2];
    hipStream_t s[2];
    for (int i = 0; i < 2; i++) {
        hipSetDevice(i);
        hipMalloc(&d[i], 1024*sizeof(float));
        hipStreamCreate(&s[i]);
    }
    
    printf("Launching kernels...\n");
    fflush(stdout);
    
    // Launch on both GPUs simultaneously
    ncclGroupStart();
    printf("  ncclAllReduce GPU 0...\n"); fflush(stdout);
    ncclAllReduce(d[0], d[0], 1024, ncclFloat, ncclSum, comms[0], s[0]);
    printf("  ncclAllReduce GPU 1...\n"); fflush(stdout);
    ncclAllReduce(d[1], d[1], 1024, ncclFloat, ncclSum, comms[1], s[1]);
    printf("  ncclGroupEnd...\n"); fflush(stdout);
    ncclGroupEnd();
    printf("Kernels launched\n"); fflush(stdout);
    
    // Check stream status without blocking
    hipSetDevice(0);
    hipError_t e0 = hipStreamQuery(s[0]);
    hipSetDevice(1);
    hipError_t e1 = hipStreamQuery(s[1]);
    printf("Stream status: GPU0=%d GPU1=%d\n", (int)e0, (int)e1);
    
    return 0;
}
