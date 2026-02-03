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
    printf("Streams created\n"); fflush(stdout);
    
    // Synchronize GPUs before AllReduce
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipDeviceSynchronize());
    printf("GPUs synchronized\n"); fflush(stdout);
    
    printf("Starting AllReduce...\n"); fflush(stdout);
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclAllReduce(d0, d0, 1024, ncclFloat, ncclSum, comms[0], s0));
    CHECK_NCCL(ncclAllReduce(d1, d1, 1024, ncclFloat, ncclSum, comms[1], s1));
    CHECK_NCCL(ncclGroupEnd());
    printf("AllReduce enqueued\n"); fflush(stdout);
    
    printf("Synchronizing streams...\n"); fflush(stdout);
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipStreamSynchronize(s0));
    printf("GPU 0 synced\n"); fflush(stdout);
    
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipStreamSynchronize(s1));
    printf("GPU 1 synced\n"); fflush(stdout);
    
    printf("TEST PASSED\n");
    return 0;
}
