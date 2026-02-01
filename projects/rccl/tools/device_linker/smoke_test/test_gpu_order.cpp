#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Test: Init on GPU 1 first\n"); fflush(stdout);
    
    // Create comm on GPU 1 first
    hipSetDevice(1);
    ncclComm_t comm1;
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    printf("Creating comm on GPU 1...\n"); fflush(stdout);
    ncclCommInitRank(&comm1, 2, id, 1);
    printf("GPU 1 comm created!\n"); fflush(stdout);
    
    // Now GPU 0
    hipSetDevice(0);
    ncclComm_t comm0;
    printf("Creating comm on GPU 0...\n"); fflush(stdout);
    ncclCommInitRank(&comm0, 2, id, 0);
    printf("GPU 0 comm created!\n"); fflush(stdout);
    
    printf("Both comms created!\n");
    
    // Try AllReduce
    float *d0, *d1;
    hipSetDevice(0); hipMalloc(&d0, 4096);
    hipSetDevice(1); hipMalloc(&d1, 4096);
    
    hipStream_t s0, s1;
    hipSetDevice(0); hipStreamCreate(&s0);
    hipSetDevice(1); hipStreamCreate(&s1);
    
    printf("Running AllReduce...\n"); fflush(stdout);
    ncclGroupStart();
    ncclAllReduce(d0, d0, 1024, ncclFloat, ncclSum, comm0, s0);
    ncclAllReduce(d1, d1, 1024, ncclFloat, ncclSum, comm1, s1);
    ncclGroupEnd();
    printf("Done!\n");
    
    return 0;
}
