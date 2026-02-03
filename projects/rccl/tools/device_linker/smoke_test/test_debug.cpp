#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <unistd.h>

#define CHECK_HIP(cmd) { hipError_t e = cmd; if (e != hipSuccess) { printf("HIP error %s\n", hipGetErrorString(e)); return 1; } }
#define CHECK_NCCL(cmd) { ncclResult_t r = cmd; if (r != ncclSuccess) { printf("NCCL error %s\n", ncclGetErrorString(r)); return 1; } }

int main() {
    printf("Start\n"); fflush(stdout);
    
    ncclComm_t comms[2];
    int devList[2] = {0, 1};
    
    printf("Creating comms...\n"); fflush(stdout);
    CHECK_NCCL(ncclCommInitAll(comms, 2, devList));
    printf("Comms created!\n"); fflush(stdout);
    
    float *d0, *d1;
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&d0, 1024*sizeof(float)));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipMalloc(&d1, 1024*sizeof(float)));
    
    hipStream_t s0, s1;
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipStreamCreate(&s0));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipStreamCreate(&s1));
    printf("Ready\n"); fflush(stdout);
    
    printf("AllReduce...\n"); fflush(stdout);
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclAllReduce(d0, d0, 1024, ncclFloat, ncclSum, comms[0], s0));
    CHECK_NCCL(ncclAllReduce(d1, d1, 1024, ncclFloat, ncclSum, comms[1], s1));
    CHECK_NCCL(ncclGroupEnd());
    printf("Enqueued\n"); fflush(stdout);
    
    // Poll for completion with timeout
    for (int i = 0; i < 100; i++) {
        hipError_t e0, e1;
        CHECK_HIP(hipSetDevice(0));
        e0 = hipStreamQuery(s0);
        CHECK_HIP(hipSetDevice(1));
        e1 = hipStreamQuery(s1);
        
        printf("Poll %d: GPU0=%d GPU1=%d\n", i, (int)e0, (int)e1);
        fflush(stdout);
        
        if (e0 == hipSuccess && e1 == hipSuccess) {
            printf("PASSED\n");
            return 0;
        }
        usleep(100000);  // 100ms
    }
    
    printf("TIMEOUT\n");
    return 1;
}
