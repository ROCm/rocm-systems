#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>

#define CHECK_HIP(cmd) { hipError_t e = cmd; if (e != hipSuccess) { printf("HIP error %s\n", hipGetErrorString(e)); return 1; } }
#define CHECK_NCCL(cmd) { ncclResult_t r = cmd; if (r != ncclSuccess) { printf("NCCL error %s\n", ncclGetErrorString(r)); return 1; } }

int main() {
    printf("Single GPU Test\n"); fflush(stdout);
    
    ncclComm_t comm;
    int devList[1] = {0};
    
    printf("Init...\n"); fflush(stdout);
    CHECK_NCCL(ncclCommInitAll(&comm, 1, devList));
    printf("OK\n"); fflush(stdout);
    
    float *d;
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&d, 1024*sizeof(float)));
    
    hipStream_t s;
    CHECK_HIP(hipStreamCreate(&s));
    
    printf("AllReduce...\n"); fflush(stdout);
    CHECK_NCCL(ncclAllReduce(d, d, 1024, ncclFloat, ncclSum, comm, s));
    printf("Enqueued\n"); fflush(stdout);
    
    CHECK_HIP(hipStreamSynchronize(s));
    printf("PASSED\n");
    return 0;
}
