#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

ncclComm_t comms[2];
float *d[2];
hipStream_t streams[2];
int done[2] = {0, 0};

void* gpu_thread(void* arg) {
    int gpu = *(int*)arg;
    printf("GPU %d: Setting device\n", gpu); fflush(stdout);
    hipSetDevice(gpu);
    
    printf("GPU %d: Enqueuing AllReduce\n", gpu); fflush(stdout);
    ncclResult_t r = ncclAllReduce(d[gpu], d[gpu], 1024, ncclFloat, ncclSum, comms[gpu], streams[gpu]);
    if (r != ncclSuccess) {
        printf("GPU %d: AllReduce FAILED: %s\n", gpu, ncclGetErrorString(r));
        return NULL;
    }
    printf("GPU %d: AllReduce enqueued OK\n", gpu); fflush(stdout);
    
    printf("GPU %d: Synchronizing...\n", gpu); fflush(stdout);
    hipError_t e = hipStreamSynchronize(streams[gpu]);
    if (e != hipSuccess) {
        printf("GPU %d: Sync FAILED: %s\n", gpu, hipGetErrorString(e));
    } else {
        printf("GPU %d: Sync OK\n", gpu);
    }
    done[gpu] = 1;
    return NULL;
}

int main() {
    printf("Parallel GPU Test\n"); fflush(stdout);
    
    int devList[2] = {0, 1};
    ncclCommInitAll(comms, 2, devList);
    printf("Comms OK\n"); fflush(stdout);
    
    for (int i = 0; i < 2; i++) {
        hipSetDevice(i);
        hipMalloc(&d[i], 1024*sizeof(float));
        hipStreamCreate(&streams[i]);
    }
    printf("Alloc OK\n"); fflush(stdout);
    
    // Use ncclGroupStart/End to ensure both operations are launched together
    printf("Using ncclGroupStart/End\n"); fflush(stdout);
    ncclGroupStart();
    ncclAllReduce(d[0], d[0], 1024, ncclFloat, ncclSum, comms[0], streams[0]);
    ncclAllReduce(d[1], d[1], 1024, ncclFloat, ncclSum, comms[1], streams[1]);
    ncclGroupEnd();
    printf("AllReduces enqueued\n"); fflush(stdout);
    
    // Poll for completion
    for (int i = 0; i < 50; i++) {
        hipError_t e0, e1;
        hipSetDevice(0);
        e0 = hipStreamQuery(streams[0]);
        hipSetDevice(1);
        e1 = hipStreamQuery(streams[1]);
        printf("Poll %d: GPU0=%d GPU1=%d\n", i, (int)e0, (int)e1);
        fflush(stdout);
        if (e0 == hipSuccess && e1 == hipSuccess) {
            printf("PASSED\n");
            return 0;
        }
        usleep(200000);
    }
    printf("TIMEOUT\n");
    return 1;
}
