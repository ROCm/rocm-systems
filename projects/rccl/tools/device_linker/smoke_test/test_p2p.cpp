#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <unistd.h>

#define CHECK_HIP(cmd) { hipError_t e = cmd; if (e != hipSuccess) { printf("HIP error %s\n", hipGetErrorString(e)); return 1; } }
#define CHECK_NCCL(cmd) { ncclResult_t r = cmd; if (r != ncclSuccess) { printf("NCCL error %s\n", ncclGetErrorString(r)); return 1; } }

int main() {
    printf("P2P Test\n"); fflush(stdout);
    
    ncclComm_t comms[2];
    int devList[2] = {0, 1};
    CHECK_NCCL(ncclCommInitAll(comms, 2, devList));
    printf("Comms OK\n"); fflush(stdout);
    
    float *send_buf, *recv_buf;
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipMalloc(&send_buf, 1024*sizeof(float)));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipMalloc(&recv_buf, 1024*sizeof(float)));
    
    hipStream_t s0, s1;
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipStreamCreate(&s0));
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipStreamCreate(&s1));
    
    printf("Send/Recv...\n"); fflush(stdout);
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclSend(send_buf, 1024, ncclFloat, 1, comms[0], s0));
    CHECK_NCCL(ncclRecv(recv_buf, 1024, ncclFloat, 0, comms[1], s1));
    CHECK_NCCL(ncclGroupEnd());
    printf("Enqueued\n"); fflush(stdout);
    
    CHECK_HIP(hipSetDevice(0));
    CHECK_HIP(hipStreamSynchronize(s0));
    printf("GPU0 done\n"); fflush(stdout);
    
    CHECK_HIP(hipSetDevice(1));
    CHECK_HIP(hipStreamSynchronize(s1));
    printf("GPU1 done\n"); fflush(stdout);
    
    printf("PASSED\n");
    return 0;
}
