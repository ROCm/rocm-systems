#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <thread>

int main() {
    printf("Test: Multi-GPU AllReduce (both GPUs participate)\n");
    fflush(stdout);
    
    int nGpus = 2;
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    
    ncclCommInitAll(comms, nGpus, devList);
    printf("Init done!\n"); fflush(stdout);
    
    float *d[2];
    hipStream_t streams[2];
    
    for (int i = 0; i < nGpus; i++) {
        hipSetDevice(i);
        hipMalloc(&d[i], 4096);
        hipStreamCreate(&streams[i]);
    }
    
    printf("Running AllReduce on both GPUs...\n"); fflush(stdout);
    
    // Launch on both GPUs (they must all participate)
    ncclGroupStart();
    for (int i = 0; i < nGpus; i++) {
        ncclAllReduce(d[i], d[i], 1024, ncclFloat, ncclSum, comms[i], streams[i]);
    }
    ncclGroupEnd();
    
    // Wait for completion
    for (int i = 0; i < nGpus; i++) {
        hipSetDevice(i);
        hipError_t err = hipStreamSynchronize(streams[i]);
        if (err != hipSuccess) {
            printf("GPU %d sync failed: %s\n", i, hipGetErrorString(err));
            return 1;
        }
    }
    
    printf("AllReduce completed!\n");
    printf("PASSED!\n");
    
    // Cleanup
    for (int i = 0; i < nGpus; i++) {
        ncclCommDestroy(comms[i]);
        hipFree(d[i]);
        hipStreamDestroy(streams[i]);
    }
    
    return 0;
}
