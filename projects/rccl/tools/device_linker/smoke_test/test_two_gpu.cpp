// Two-GPU test to debug multi-GPU issues
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
    printf("=== Two-GPU Test ===\n\n");
    
    int deviceCount = 0;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    printf("Found %d HIP device(s)\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs\n");
        return 1;
    }
    
    printf("Step 1: Creating device list\n");
    int nGpus = 2;
    int devList[2] = {0, 1};
    
    printf("Step 2: Creating communicators with ncclCommInitAll\n");
    ncclComm_t comms[2];
    CHECK_NCCL(ncclCommInitAll(comms, nGpus, devList));
    printf("Communicators created!\n");
    
    printf("Step 3: Allocating memory\n");
    const int count = 1024;
    float *sendbuff0, *sendbuff1, *recvbuff0, *recvbuff1;
    
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&sendbuff0, count * sizeof(float)));
    CHECK_HIP(hipMalloc(&recvbuff0, count * sizeof(float)));
    
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipMalloc(&sendbuff1, count * sizeof(float)));
    CHECK_HIP(hipMalloc(&recvbuff1, count * sizeof(float)));
    printf("Memory allocated\n");
    
    printf("Step 4: Initializing data\n");
    float *hostbuff = (float*)malloc(count * sizeof(float));
    
    for (int i = 0; i < count; i++) hostbuff[i] = 1.0f;
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMemcpy(sendbuff0, hostbuff, count * sizeof(float), hipMemcpyHostToDevice));
    
    for (int i = 0; i < count; i++) hostbuff[i] = 2.0f;
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipMemcpy(sendbuff1, hostbuff, count * sizeof(float), hipMemcpyHostToDevice));
    printf("Data initialized\n");
    
    printf("Step 5: Creating streams\n");
    hipStream_t stream0, stream1;
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipStreamCreate(&stream0));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipStreamCreate(&stream1));
    printf("Streams created\n");
    
    printf("Step 6: Running AllReduce\n");
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclAllReduce(sendbuff0, recvbuff0, count, ncclFloat, ncclSum, comms[0], stream0));
    CHECK_NCCL(ncclAllReduce(sendbuff1, recvbuff1, count, ncclFloat, ncclSum, comms[1], stream1));
    CHECK_NCCL(ncclGroupEnd());
    
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipStreamSynchronize(stream0));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipStreamSynchronize(stream1));
    printf("AllReduce completed!\n");
    
    printf("Step 7: Verifying results\n");
    float expected = 3.0f;  // 1 + 2
    int errors = 0;
    
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMemcpy(hostbuff, recvbuff0, count * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < count; i++) {
        if (hostbuff[i] != expected) errors++;
    }
    
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipMemcpy(hostbuff, recvbuff1, count * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < count; i++) {
        if (hostbuff[i] != expected) errors++;
    }
    
    printf("Step 8: Cleanup\n");
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipStreamDestroy(stream0));
    CHECK_HIP(hipFree(sendbuff0));
    CHECK_HIP(hipFree(recvbuff0));
    
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipStreamDestroy(stream1));
    CHECK_HIP(hipFree(sendbuff1));
    CHECK_HIP(hipFree(recvbuff1));
    
    CHECK_NCCL(ncclCommDestroy(comms[0]));
    CHECK_NCCL(ncclCommDestroy(comms[1]));
    
    free(hostbuff);
    
    if (errors == 0) {
        printf("\n=== TEST PASSED ===\n");
        return 0;
    } else {
        printf("\n=== TEST FAILED: %d errors ===\n", errors);
        return 1;
    }
}
