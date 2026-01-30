// Simple RCCL test to verify device linker build works
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

int main() {
    printf("=== Simple RCCL Test ===\n\n");
    
    // Initialize HIP
    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    printf("Found %d HIP device(s)\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No HIP devices available\n");
        return 1;
    }
    
    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    printf("Using device: %s\n\n", props.name);
    
    CHECK_HIP(hipSetDevice(0));
    
    // Create NCCL communicator
    ncclComm_t comm;
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_NCCL(ncclCommInitRank(&comm, 1, id, 0));
    printf("NCCL communicator created\n");
    
    // Allocate device memory
    const int count = 1024;
    float *sendbuff, *recvbuff;
    CHECK_HIP(hipMalloc(&sendbuff, count * sizeof(float)));
    CHECK_HIP(hipMalloc(&recvbuff, count * sizeof(float)));
    
    // Initialize send buffer
    float *hostbuff = (float*)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) hostbuff[i] = 1.0f;
    CHECK_HIP(hipMemcpy(sendbuff, hostbuff, count * sizeof(float), hipMemcpyHostToDevice));
    
    // Create stream
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    
    // Run AllReduce
    printf("Running AllReduce...\n");
    CHECK_NCCL(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    printf("AllReduce completed!\n");
    
    // Verify result
    CHECK_HIP(hipMemcpy(hostbuff, recvbuff, count * sizeof(float), hipMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < count; i++) {
        if (hostbuff[i] != 1.0f) errors++;
    }
    
    if (errors == 0) {
        printf("\n=== TEST PASSED ===\n");
    } else {
        printf("\n=== TEST FAILED: %d errors ===\n", errors);
    }
    
    // Cleanup
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(sendbuff));
    CHECK_HIP(hipFree(recvbuff));
    CHECK_NCCL(ncclCommDestroy(comm));
    free(hostbuff);
    
    return errors > 0 ? 1 : 0;
}
