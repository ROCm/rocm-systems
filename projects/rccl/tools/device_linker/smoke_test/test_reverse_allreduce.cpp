#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Test with reversed device order\n"); fflush(stdout);
    
    int devList[2] = {1, 0};  // GPU 1 first, GPU 0 second
    ncclComm_t comms[2];
    
    printf("Init...\n"); fflush(stdout);
    ncclCommInitAll(comms, 2, devList);
    printf("Init done!\n"); fflush(stdout);
    
    // Allocate: comms[0] is on device 1, comms[1] is on device 0
    float *d0, *d1;
    hipSetDevice(1); hipMalloc(&d1, 4096);
    hipSetDevice(0); hipMalloc(&d0, 4096);
    
    hipStream_t s0, s1;
    hipSetDevice(1); hipStreamCreate(&s1);
    hipSetDevice(0); hipStreamCreate(&s0);
    
    printf("AllReduce...\n"); fflush(stdout);
    ncclGroupStart();
    ncclAllReduce(d1, d1, 1024, ncclFloat, ncclSum, comms[0], s1);
    ncclAllReduce(d0, d0, 1024, ncclFloat, ncclSum, comms[1], s0);
    ncclGroupEnd();
    printf("GroupEnd done!\n"); fflush(stdout);
    
    hipSetDevice(1); hipStreamSynchronize(s1);
    hipSetDevice(0); hipStreamSynchronize(s0);
    printf("PASSED!\n");
    
    return 0;
}
