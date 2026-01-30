// Two-GPU test without ncclGroupStart/End
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

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

ncclUniqueId g_id;
ncclComm_t g_comms[2];
volatile int g_errors = 0;

void* thread_func(void* arg) {
    int rank = *(int*)arg;
    printf("Thread %d starting\n", rank);
    
    CHECK_HIP(hipSetDevice(rank));
    
    printf("Thread %d: Creating comm\n", rank);
    CHECK_NCCL(ncclCommInitRank(&g_comms[rank], 2, g_id, rank));
    printf("Thread %d: Comm created\n", rank);
    
    // Allocate memory
    const int count = 1024;
    float *sendbuff, *recvbuff;
    CHECK_HIP(hipMalloc(&sendbuff, count * sizeof(float)));
    CHECK_HIP(hipMalloc(&recvbuff, count * sizeof(float)));
    
    // Initialize
    float* hostbuff = (float*)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) hostbuff[i] = (float)(rank + 1);
    CHECK_HIP(hipMemcpy(sendbuff, hostbuff, count * sizeof(float), hipMemcpyHostToDevice));
    
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    
    printf("Thread %d: Running AllReduce\n", rank);
    CHECK_NCCL(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, g_comms[rank], stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    printf("Thread %d: AllReduce done\n", rank);
    
    // Verify
    float expected = 3.0f;  // 1 + 2
    CHECK_HIP(hipMemcpy(hostbuff, recvbuff, count * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < count; i++) {
        if (hostbuff[i] != expected) {
            printf("Thread %d: error at %d: got %f expected %f\n", rank, i, hostbuff[i], expected);
            g_errors++;
            break;
        }
    }
    
    // Cleanup
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(sendbuff));
    CHECK_HIP(hipFree(recvbuff));
    CHECK_NCCL(ncclCommDestroy(g_comms[rank]));
    free(hostbuff);
    
    printf("Thread %d done\n", rank);
    return NULL;
}

int main() {
    printf("=== Two-GPU Test (no group API) ===\n\n");
    
    int deviceCount;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    printf("Found %d devices\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need 2+ GPUs\n");
        return 1;
    }
    
    CHECK_NCCL(ncclGetUniqueId(&g_id));
    
    pthread_t threads[2];
    int ranks[2] = {0, 1};
    
    for (int i = 0; i < 2; i++) {
        pthread_create(&threads[i], NULL, thread_func, &ranks[i]);
    }
    
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }
    
    if (g_errors == 0) {
        printf("\n=== TEST PASSED ===\n");
        return 0;
    } else {
        printf("\n=== TEST FAILED ===\n");
        return 1;
    }
}
