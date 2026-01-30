// Multi-GPU Broadcast test for device linker build
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
    printf("=== Multi-GPU Broadcast Test ===\n\n");
    
    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    printf("Found %d HIP device(s)\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs for multi-GPU test\n");
        return 1;
    }
    
    // Use min of available GPUs or 4 for simplicity
    int nGpus = (deviceCount > 4) ? 4 : deviceCount;
    int root = 0;
    printf("Testing Broadcast from root=%d with %d GPUs\n\n", root, nGpus);
    
    // Create device list
    int* devList = (int*)malloc(nGpus * sizeof(int));
    for (int i = 0; i < nGpus; i++) devList[i] = i;
    
    // Create communicators
    ncclComm_t* comms = (ncclComm_t*)malloc(nGpus * sizeof(ncclComm_t));
    CHECK_NCCL(ncclCommInitAll(comms, nGpus, devList));
    printf("Communicators created\n");
    
    // Allocate device memory and streams
    const int count = 1024;
    float** buffs = (float**)malloc(nGpus * sizeof(float*));
    hipStream_t* streams = (hipStream_t*)malloc(nGpus * sizeof(hipStream_t));
    
    for (int i = 0; i < nGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&buffs[i], count * sizeof(float)));
        CHECK_HIP(hipStreamCreate(&streams[i]));
        
        // Initialize buffer: root has 42.0, others have 0.0
        float* hostbuff = (float*)malloc(count * sizeof(float));
        float init_val = (i == root) ? 42.0f : 0.0f;
        for (int j = 0; j < count; j++) hostbuff[j] = init_val;
        CHECK_HIP(hipMemcpy(buffs[i], hostbuff, count * sizeof(float), hipMemcpyHostToDevice));
        free(hostbuff);
    }
    
    // Run Broadcast on all GPUs
    printf("Running Broadcast...\n");
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nGpus; i++) {
        CHECK_NCCL(ncclBroadcast(buffs[i], buffs[i], count, ncclFloat, root, comms[i], streams[i]));
    }
    CHECK_NCCL(ncclGroupEnd());
    
    // Synchronize all streams
    for (int i = 0; i < nGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipStreamSynchronize(streams[i]));
    }
    printf("Broadcast completed!\n");
    
    // Verify results: all GPUs should have 42.0
    int total_errors = 0;
    
    for (int i = 0; i < nGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        float* hostbuff = (float*)malloc(count * sizeof(float));
        CHECK_HIP(hipMemcpy(hostbuff, buffs[i], count * sizeof(float), hipMemcpyDeviceToHost));
        
        int errors = 0;
        for (int j = 0; j < count; j++) {
            if (hostbuff[j] != 42.0f) {
                if (errors < 3) {
                    printf("GPU %d: error at index %d: got %f, expected 42.0\n", 
                           i, j, hostbuff[j]);
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
        CHECK_HIP(hipFree(buffs[i]));
        CHECK_NCCL(ncclCommDestroy(comms[i]));
    }
    
    free(devList);
    free(comms);
    free(buffs);
    free(streams);
    
    if (total_errors == 0) {
        printf("\n=== TEST PASSED ===\n");
        return 0;
    } else {
        printf("\n=== TEST FAILED: %d total errors ===\n", total_errors);
        return 1;
    }
}
