#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Test: Init both, then launch one at a time\n"); fflush(stdout);
    
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    ncclCommInitAll(comms, 2, devList);
    printf("Init done!\n"); fflush(stdout);
    
    float *d0, *d1;
    hipSetDevice(0); hipMalloc(&d0, 4096);
    hipSetDevice(1); hipMalloc(&d1, 4096);
    
    hipStream_t s0, s1;
    hipSetDevice(0); hipStreamCreate(&s0);
    hipSetDevice(1); hipStreamCreate(&s1);
    
    // Try just GPU 0
    printf("AllReduce GPU 0 only (comm[0])...\n"); fflush(stdout);
    ncclAllReduce(d0, d0, 1024, ncclFloat, ncclSum, comms[0], s0);
    hipSetDevice(0); hipStreamSynchronize(s0);
    printf("GPU 0 ok!\n"); fflush(stdout);
    
    // Try just GPU 1
    printf("AllReduce GPU 1 only (comm[1])...\n"); fflush(stdout);
    ncclAllReduce(d1, d1, 1024, ncclFloat, ncclSum, comms[1], s1);
    hipSetDevice(1); hipStreamSynchronize(s1);
    printf("GPU 1 ok!\n"); fflush(stdout);
    
    printf("PASSED!\n");
    return 0;
}
