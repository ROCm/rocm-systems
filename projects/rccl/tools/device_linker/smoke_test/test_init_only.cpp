#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

int main() {
    printf("=== Init Only Test ===\n");
    
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    printf("Found %d devices\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need 2 GPUs\n");
        return 1;
    }
    
    printf("Creating communicators...\n");
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    ncclResult_t r = ncclCommInitAll(comms, 2, devList);
    if (r != ncclSuccess) {
        printf("ncclCommInitAll failed: %s\n", ncclGetErrorString(r));
        return 1;
    }
    printf("Communicators created!\n");
    
    printf("Destroying communicators...\n");
    ncclCommDestroy(comms[0]);
    ncclCommDestroy(comms[1]);
    printf("Done!\n");
    
    return 0;
}
