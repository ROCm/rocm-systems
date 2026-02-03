#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("Test\n"); fflush(stdout);
    
    ncclComm_t comms[2];
    int devList[2] = {0, 1};
    ncclCommInitAll(comms, 2, devList);
    printf("Comms OK\n"); fflush(stdout);
    
    float *d[2];
    hipStream_t s[2];
    for (int i = 0; i < 2; i++) {
        hipSetDevice(i);
        hipMalloc(&d[i], 1024*sizeof(float));
        hipStreamCreate(&s[i]);
    }
    
    printf("GroupStart\n"); fflush(stdout);
    ncclGroupStart();
    printf("AllReduce0\n"); fflush(stdout);
    ncclAllReduce(d[0], d[0], 1024, ncclFloat, ncclSum, comms[0], s[0]);
    printf("AllReduce1\n"); fflush(stdout);
    ncclAllReduce(d[1], d[1], 1024, ncclFloat, ncclSum, comms[1], s[1]);
    printf("GroupEnd\n"); fflush(stdout);
    ncclGroupEnd();
    printf("DONE\n"); fflush(stdout);
    
    return 0;
}
