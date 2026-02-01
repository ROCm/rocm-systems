#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Start\n"); fflush(stdout);
    
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    
    printf("Init...\n"); fflush(stdout);
    ncclCommInitAll(comms, 2, devList);
    printf("Init done\n"); fflush(stdout);
    
    float *d0, *d1;
    hipSetDevice(0); hipMalloc(&d0, 4096);
    hipSetDevice(1); hipMalloc(&d1, 4096);
    
    hipStream_t s0, s1;
    hipSetDevice(0); hipStreamCreate(&s0);
    hipSetDevice(1); hipStreamCreate(&s1);
    
    printf("About to AllReduce\n"); fflush(stdout);
    ncclGroupStart();
    ncclAllReduce(d0, d0, 1024, ncclFloat, ncclSum, comms[0], s0);
    ncclAllReduce(d1, d1, 1024, ncclFloat, ncclSum, comms[1], s1);
    ncclGroupEnd();
    printf("GroupEnd done\n"); fflush(stdout);
    
    hipSetDevice(0); hipStreamSynchronize(s0);
    hipSetDevice(1); hipStreamSynchronize(s1);
    
    printf("PASSED\n");
    return 0;
}
