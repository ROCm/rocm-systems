#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Init...\n");
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    ncclCommInitAll(comms, 2, devList);
    printf("Initialized!\n");
    
    float *d0_send, *d0_recv, *d1_send, *d1_recv;
    hipSetDevice(0); hipMalloc(&d0_send, 1024*4); hipMalloc(&d0_recv, 1024*4);
    hipSetDevice(1); hipMalloc(&d1_send, 1024*4); hipMalloc(&d1_recv, 1024*4);
    
    hipStream_t s0, s1;
    hipSetDevice(0); hipStreamCreate(&s0);
    hipSetDevice(1); hipStreamCreate(&s1);
    
    printf("Before GroupStart\n"); fflush(stdout);
    ncclGroupStart();
    printf("After GroupStart, before AllReduce 0\n"); fflush(stdout);
    ncclAllReduce(d0_send, d0_recv, 1024, ncclFloat, ncclSum, comms[0], s0);
    printf("After AllReduce 0, before AllReduce 1\n"); fflush(stdout);
    ncclAllReduce(d1_send, d1_recv, 1024, ncclFloat, ncclSum, comms[1], s1);
    printf("After AllReduce 1, before GroupEnd\n"); fflush(stdout);
    ncclGroupEnd();
    printf("After GroupEnd\n"); fflush(stdout);
    
    hipSetDevice(0); hipStreamSynchronize(s0);
    hipSetDevice(1); hipStreamSynchronize(s1);
    printf("Synced!\n");
    
    return 0;
}
