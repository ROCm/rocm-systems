#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>

// Simple two-GPU test with small data
int main() {
    printf("=== Two GPU Simple AllReduce Test ===\n");
    
    // Use GPUs 0 and 1
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    
    printf("Init with NCCL_DEBUG...\n");
    
    ncclResult_t res = ncclCommInitAll(comms, 2, devList);
    if (res != ncclSuccess) {
        printf("ncclCommInitAll failed: %d\n", res);
        return 1;
    }
    printf("Initialized!\n");
    
    // Very small data - just 4 floats
    const int count = 4;
    printf("Allocating %d floats per GPU...\n", count);
    
    float *d_send[2], *d_recv[2];
    hipStream_t streams[2];
    
    for (int i = 0; i < 2; i++) {
        hipSetDevice(devList[i]);
        hipMalloc(&d_send[i], count * sizeof(float));
        hipMalloc(&d_recv[i], count * sizeof(float));
        hipStreamCreate(&streams[i]);
        
        // Initialize with device index
        float h_init[4] = {(float)i, (float)i, (float)i, (float)i};
        hipMemcpy(d_send[i], h_init, count * sizeof(float), hipMemcpyHostToDevice);
    }
    printf("Allocated and initialized!\n");
    
    printf("Running AllReduce (expecting crash here)...\n");
    fflush(stdout);
    
    // This is where we expect the crash
    ncclGroupStart();
    for (int i = 0; i < 2; i++) {
        ncclAllReduce(d_send[i], d_recv[i], count, ncclFloat, ncclSum, comms[i], streams[i]);
    }
    res = ncclGroupEnd();
    
    if (res != ncclSuccess) {
        printf("ncclGroupEnd failed: %d\n", res);
    } else {
        printf("GroupEnd succeeded!\n");
    }
    
    printf("Synchronizing...\n");
    fflush(stdout);
    
    for (int i = 0; i < 2; i++) {
        hipSetDevice(devList[i]);
        hipError_t err = hipStreamSynchronize(streams[i]);
        if (err != hipSuccess) {
            printf("GPU %d sync failed: %s\n", i, hipGetErrorString(err));
        }
    }
    printf("Synced!\n");
    
    // Verify results (each element should be 0+1=1)
    printf("Verifying...\n");
    int errors = 0;
    for (int i = 0; i < 2; i++) {
        hipSetDevice(devList[i]);
        float h_recv[4];
        hipMemcpy(h_recv, d_recv[i], count * sizeof(float), hipMemcpyDeviceToHost);
        for (int j = 0; j < count; j++) {
            if (h_recv[j] != 1.0f) {
                printf("GPU %d elem %d: expected 1.0, got %f\n", i, j, h_recv[j]);
                errors++;
            }
        }
    }
    
    printf("Cleanup...\n");
    for (int i = 0; i < 2; i++) {
        hipSetDevice(devList[i]);
        hipFree(d_send[i]);
        hipFree(d_recv[i]);
        hipStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }
    
    if (errors == 0) {
        printf("=== PASSED ===\n");
        return 0;
    } else {
        printf("=== FAILED with %d errors ===\n", errors);
        return 1;
    }
}
