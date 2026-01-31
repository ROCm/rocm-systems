#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(cmd) do { hipError_t e = cmd; if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)
#define CHECK_NCCL(cmd) do { ncclResult_t r = cmd; if (r != ncclSuccess) { printf("NCCL error %s at %s:%d\n", ncclGetErrorString(r), __FILE__, __LINE__); exit(1); } } while(0)

int main() {
    printf("=== Sync Test ===\n");
    CHECK_HIP(hipSetDevice(0));
    
    ncclComm_t comm;
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_NCCL(ncclCommInitRank(&comm, 1, id, 0));
    
    const int count = 16;
    float *sendbuff, *recvbuff;
    CHECK_HIP(hipMalloc(&sendbuff, count * sizeof(float)));
    CHECK_HIP(hipMalloc(&recvbuff, count * sizeof(float)));
    
    float *hostbuff = (float*)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) hostbuff[i] = (float)(i + 1);
    CHECK_HIP(hipMemcpy(sendbuff, hostbuff, count * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(recvbuff, 0, count * sizeof(float)));
    CHECK_HIP(hipDeviceSynchronize());
    
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    
    CHECK_NCCL(ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    
    CHECK_HIP(hipMemcpy(hostbuff, recvbuff, count * sizeof(float), hipMemcpyDeviceToHost));
    
    int errors = 0;
    for (int i = 0; i < count; i++) {
        if (hostbuff[i] != (float)(i+1)) errors++;
    }
    printf("Results: %.1f %.1f %.1f %.1f\n", hostbuff[0], hostbuff[1], hostbuff[2], hostbuff[3]);
    printf("%s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(sendbuff));
    CHECK_HIP(hipFree(recvbuff));
    CHECK_NCCL(ncclCommDestroy(comm));
    free(hostbuff);
    return errors;
}
