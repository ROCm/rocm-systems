#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== Single GPU AllReduce Test ===\n");
    
    int devList[1] = {0};
    ncclComm_t comm;
    
    printf("Init...\n");
    ncclCommInitAll(&comm, 1, devList);
    printf("Initialized!\n");
    
    printf("Allocating...\n");
    float *d_send, *d_recv;
    float *h_send = (float*)malloc(1024 * sizeof(float));
    float *h_recv = (float*)malloc(1024 * sizeof(float));
    
    // Initialize host data
    for (int i = 0; i < 1024; i++) h_send[i] = 1.0f;
    
    hipSetDevice(0);
    hipMalloc(&d_send, 1024 * sizeof(float));
    hipMalloc(&d_recv, 1024 * sizeof(float));
    hipMemcpy(d_send, h_send, 1024 * sizeof(float), hipMemcpyHostToDevice);
    printf("Allocated!\n");
    
    printf("Creating stream...\n");
    hipStream_t s0;
    hipStreamCreate(&s0);
    printf("Stream created!\n");
    
    printf("Running AllReduce...\n");
    ncclAllReduce(d_send, d_recv, 1024, ncclFloat, ncclSum, comm, s0);
    printf("AllReduce submitted!\n");
    
    printf("Synchronizing...\n");
    hipStreamSynchronize(s0);
    printf("Synced!\n");
    
    // Verify result
    hipMemcpy(h_recv, d_recv, 1024 * sizeof(float), hipMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < 1024; i++) {
        if (h_recv[i] != 1.0f) errors++;
    }
    printf("Verification: %d errors\n", errors);
    
    printf("Cleanup...\n");
    hipFree(d_send); hipFree(d_recv);
    free(h_send); free(h_recv);
    ncclCommDestroy(comm);
    
    if (errors == 0) {
        printf("=== PASSED ===\n");
        return 0;
    } else {
        printf("=== FAILED ===\n");
        return 1;
    }
}
