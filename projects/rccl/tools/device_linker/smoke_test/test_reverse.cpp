#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("Test: ncclCommInitAll with reversed device order\n"); fflush(stdout);
    
    int devList[2] = {1, 0};  // GPU 1 first, GPU 0 second
    ncclComm_t comms[2];
    
    printf("Init...\n"); fflush(stdout);
    ncclCommInitAll(comms, 2, devList);
    printf("Init done!\n"); fflush(stdout);
    
    return 0;
}
