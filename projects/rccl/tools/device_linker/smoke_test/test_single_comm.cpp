#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Single comm test\n"); fflush(stdout);
    
    ncclComm_t comm;
    ncclCommInitAll(&comm, 1, nullptr);
    printf("Init done\n"); fflush(stdout);
    
    float *d;
    hipMalloc(&d, 4096);
    
    hipStream_t s;
    hipStreamCreate(&s);
    
    printf("AllReduce...\n"); fflush(stdout);
    ncclAllReduce(d, d, 1024, ncclFloat, ncclSum, comm, s);
    hipStreamSynchronize(s);
    printf("Done!\n");
    
    return 0;
}
