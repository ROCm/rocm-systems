#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("=== Minimal AllReduce Test ===\n");
    
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    
    printf("Init...\n");
    ncclCommInitAll(comms, 2, devList);
    printf("Initialized!\n");
    
    printf("Allocating...\n");
    float *d0_send, *d0_recv, *d1_send, *d1_recv;
    hipSetDevice(0);
    hipMalloc(&d0_send, 1024 * sizeof(float));
    hipMalloc(&d0_recv, 1024 * sizeof(float));
    hipSetDevice(1);
    hipMalloc(&d1_send, 1024 * sizeof(float));
    hipMalloc(&d1_recv, 1024 * sizeof(float));
    printf("Allocated!\n");
    
    printf("Creating streams...\n");
    hipStream_t s0, s1;
    hipSetDevice(0);
    hipStreamCreate(&s0);
    hipSetDevice(1);
    hipStreamCreate(&s1);
    printf("Streams created!\n");
    
    printf("Running AllReduce...\n");
    ncclGroupStart();
    ncclAllReduce(d0_send, d0_recv, 1024, ncclFloat, ncclSum, comms[0], s0);
    ncclAllReduce(d1_send, d1_recv, 1024, ncclFloat, ncclSum, comms[1], s1);
    ncclGroupEnd();
    printf("GroupEnd done!\n");
    
    printf("Synchronizing...\n");
    hipSetDevice(0);
    hipStreamSynchronize(s0);
    hipSetDevice(1);
    hipStreamSynchronize(s1);
    printf("Synced!\n");
    
    printf("Cleanup...\n");
    hipSetDevice(0);
    hipFree(d0_send); hipFree(d0_recv);
    hipSetDevice(1);
    hipFree(d1_send); hipFree(d1_recv);
    ncclCommDestroy(comms[0]);
    ncclCommDestroy(comms[1]);
    
    printf("=== PASSED ===\n");
    return 0;
}
