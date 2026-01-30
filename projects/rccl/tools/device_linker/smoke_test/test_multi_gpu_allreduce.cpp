// Multi-GPU AllReduce test for device linker build
// Uses ncclCommInitAll for simpler initialization

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_NCCL(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("NCCL error %s at %s:%d\n", ncclGetErrorString(r), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    printf("=== Multi-GPU AllReduce Test ===\n\n");
    
    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    printf("Found %d HIP device(s)\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs for multi-GPU test\n");
        return 1;
    }
    
    // Use min of available GPUs or 4 for simplicity
    int nGpus = (deviceCount > 4) ? 4 : deviceCount;
    printf("Testing AllReduce with %d GPUs\n\n", nGpus);
    
    // Create device list
    int* devList = (int*)malloc(nGpus * sizeof(int));
    for (int i = 0; i < nGpus; i++) devList[i] = i;
    
    // Create communicators
    ncclComm_t* comms = (ncclComm_t*)malloc(nGpus * sizeof(ncclComm_t));
    CHECK_NCCL(ncclCommInitAll(comms, nGpus, devList));
    printf("Communicators created\n");
    
    // Allocate device memory and streams
    const int count = 1024;
    float** sendbuffs = (float**)malloc(nGpus * sizeof(float*));
    float** recvbuffs = (float**)malloc(nGpus * sizeof(float*));
    hipStream_t* streams = (hipStream_t*)malloc(nGpus * sizeof(hipStream_t));
    
    for (int i = 0; i < nGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&sendbuffs[i], count * sizeof(float)));
        CHECK_HIP(hipMalloc(&recvbuffs[i], count * sizeof(float)));
        CHECK_HIP(hipStreamCreate(&streams[i]));
        
        // Initialize send buffer with (i+1)
        float* hostbuff = (float*)malloc(count * sizeof(float));
        for (int j = 0; j < count; j++) hostbuff[j] = (float)(i + 1);
        CHECK_HIP(hipMemcpy(sendbuffs[i], hostbuff, count * sizeof(float), hipMemcpyHostToDevice));
        free(hostbuff);
    }
    
    // Run AllReduce on all GPUs
    printf("Running AllReduce...\n");
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nGpus; i++) {
        CHECK_NCCL(ncclAllReduce(sendbuffs[i], recvbuffs[i], count, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    CHECK_NCCL(ncclGroupEnd());
    
    // Synchronize all streams
    for (int i = 0; i < nGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipStreamSynchronize(streams[i]));
    }
    printf("AllReduce completed!\n");
    
    // Verify results
    float expected = (float)(nGpus * (nGpus + 1) / 2);  // sum of 1+2+...+nGpus
    int total_errors = 0;
    
    for (int i = 0; i < nGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        float* hostbuff = (float*)malloc(count * sizeof(float));
        CHECK_HIP(hipMemcpy(hostbuff, recvbuffs[i], count * sizeof(float), hipMemcpyDeviceToHost));
        
        int errors = 0;
        for (int j = 0; j < count; j++) {
            if (hostbuff[j] != expected) {
                if (errors < 3) {
                    printf("GPU %d: error at index %d: got %f, expected %f\n", 
                           i, j, hostbuff[j], expected);
                }
                errors++;
            }
        }
        if (errors > 0) {
            printf("GPU %d: %d errors\n", i, errors);
        }
        total_errors += errors;
        free(hostbuff);
    }
    
    // Cleanup
    for (int i = 0; i < nGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipStreamDestroy(streams[i]));
        CHECK_HIP(hipFree(sendbuffs[i]));
        CHECK_HIP(hipFree(recvbuffs[i]));
        CHECK_NCCL(ncclCommDestroy(comms[i]));
    }
    
    free(devList);
    free(comms);
    free(sendbuffs);
    free(recvbuffs);
    free(streams);
    
    if (total_errors == 0) {
        printf("\n=== TEST PASSED ===\n");
        return 0;
    } else {
        printf("\n=== TEST FAILED: %d total errors ===\n", total_errors);
        return 1;
    }
}
