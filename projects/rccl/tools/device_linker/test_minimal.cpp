#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(cmd) do { hipError_t e = cmd; if (e != hipSuccess) { printf("HIP error %s\n", hipGetErrorString(e)); exit(1); } } while(0)
#define CHECK_NCCL(cmd) do { ncclResult_t r = cmd; if (r != ncclSuccess) { printf("NCCL error %s\n", ncclGetErrorString(r)); exit(1); } } while(0)

int main() {
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
    CHECK_HIP(hipMemset(recvbuff, 0xCC, count * sizeof(float)));
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
    printf("Recv: %.1f %.1f %.1f %.1f - %s (%d errors)\n", 
           hostbuff[0], hostbuff[1], hostbuff[2], hostbuff[3],
           errors == 0 ? "PASS" : "FAIL", errors);
    
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(sendbuff));
    CHECK_HIP(hipFree(recvbuff));
    CHECK_NCCL(ncclCommDestroy(comm));
    free(hostbuff);
    return errors;
}
