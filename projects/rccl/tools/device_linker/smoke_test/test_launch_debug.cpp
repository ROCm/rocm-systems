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
    
    // Try single AllReduce on GPU 0 only
    printf("Test 1: Single AllReduce on GPU 0...\n");
    ncclGroupStart();
    ncclAllReduce(d0_send, d0_recv, 1024, ncclFloat, ncclSum, comms[0], s0);
    ncclGroupEnd();
    hipSetDevice(0);
    hipStreamSynchronize(s0);
    printf("GPU 0 AllReduce worked!\n");
    
    // Try single AllReduce on GPU 1 only
    printf("Test 2: Single AllReduce on GPU 1...\n");
    ncclGroupStart();
    ncclAllReduce(d1_send, d1_recv, 1024, ncclFloat, ncclSum, comms[1], s1);
    ncclGroupEnd();
    hipSetDevice(1);
    hipStreamSynchronize(s1);
    printf("GPU 1 AllReduce worked!\n");
    
    // Try both
    printf("Test 3: Both GPUs together...\n");
    ncclGroupStart();
    ncclAllReduce(d0_send, d0_recv, 1024, ncclFloat, ncclSum, comms[0], s0);
    ncclAllReduce(d1_send, d1_recv, 1024, ncclFloat, ncclSum, comms[1], s1);
    ncclGroupEnd();
    printf("Both GPUs GroupEnd worked!\n");
    
    return 0;
}
