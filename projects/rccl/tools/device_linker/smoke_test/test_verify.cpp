#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <vector>

int main() {
    printf("Multi-GPU AllReduce verification\n");
    
    int devList[2] = {0, 1};
    ncclComm_t comms[2];
    ncclCommInitAll(comms, 2, devList);
    
    const int N = 1024;
    float *d0, *d1;
    hipSetDevice(0); hipMalloc(&d0, N * sizeof(float));
    hipSetDevice(1); hipMalloc(&d1, N * sizeof(float));
    
    // Initialize: GPU0 has 1.0, GPU1 has 2.0
    std::vector<float> init0(N, 1.0f), init1(N, 2.0f);
    hipSetDevice(0); hipMemcpy(d0, init0.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipSetDevice(1); hipMemcpy(d1, init1.data(), N * sizeof(float), hipMemcpyHostToDevice);
    
    hipStream_t s0, s1;
    hipSetDevice(0); hipStreamCreate(&s0);
    hipSetDevice(1); hipStreamCreate(&s1);
    
    // AllReduce Sum: result should be 3.0 on both GPUs
    ncclGroupStart();
    ncclAllReduce(d0, d0, N, ncclFloat, ncclSum, comms[0], s0);
    ncclAllReduce(d1, d1, N, ncclFloat, ncclSum, comms[1], s1);
    ncclGroupEnd();
    
    hipSetDevice(0); hipStreamSynchronize(s0);
    hipSetDevice(1); hipStreamSynchronize(s1);
    
    // Verify
    std::vector<float> result0(N), result1(N);
    hipSetDevice(0); hipMemcpy(result0.data(), d0, N * sizeof(float), hipMemcpyDeviceToHost);
    hipSetDevice(1); hipMemcpy(result1.data(), d1, N * sizeof(float), hipMemcpyDeviceToHost);
    
    bool pass = true;
    float expected = 3.0f;
    for (int i = 0; i < N && pass; i++) {
        if (result0[i] != expected || result1[i] != expected) {
            printf("FAIL at index %d: gpu0=%f, gpu1=%f (expected %f)\n", 
                   i, result0[i], result1[i], expected);
            pass = false;
        }
    }
    
    if (pass) {
        printf("SUCCESS: AllReduce produced correct results!\n");
        printf("  GPU0[0] = %f, GPU1[0] = %f (expected %f)\n", result0[0], result1[0], expected);
    }
    
    // Cleanup
    ncclCommDestroy(comms[0]);
    ncclCommDestroy(comms[1]);
    hipFree(d0);
    hipFree(d1);
    
    return pass ? 0 : 1;
}
