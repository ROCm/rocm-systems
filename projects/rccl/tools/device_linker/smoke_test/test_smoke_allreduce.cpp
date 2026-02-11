/*
 * Smoke test 1: AllReduce, out-of-place, SUM, float32.
 * Single-process only. Verifies result on host.
 *
 * Usage: test_smoke_allreduce [num_gpus] [array_length]
 *   num_gpus    Number of GPUs (default 2)
 *   array_length  Elements per rank (default 1024)
 *
 * Each rank i fills its input with (i+1). Expected result on all ranks: sum(1..num_gpus).
 *
 * Build (no script required):
 *   System ROCm:  hipcc -O2 -I/opt/rocm/include test_smoke_allreduce.cpp -L/opt/rocm/lib -lrccl -Wl,-rpath,/opt/rocm/lib -o test_smoke_allreduce
 *   RCCL build:  hipcc -O2 -I$BUILD_DIR/include test_smoke_allreduce.cpp -L$BUILD_DIR -lrccl -Wl,-rpath,$BUILD_DIR -o test_smoke_allreduce
 * Run:           ./test_smoke_allreduce [num_gpus] [array_length]
 */

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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

static int getIntArg(int argc, char** argv, int idx, int defaultVal) {
    if (idx < argc) {
        int v = atoi(argv[idx]);
        if (v > 0) return v;
    }
    return defaultVal;
}

int main(int argc, char** argv) {
    int numGpus = getIntArg(argc, argv, 1, 2);
    int N       = getIntArg(argc, argv, 2, 1024);

    if (numGpus < 2) {
        printf("Skip: requires at least 2 GPUs (got %d). Single-GPU run is not a real collective.\n", numGpus);
        return 0;
    }

    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
        printf("Skip: hipGetDeviceCount failed (no GPUs or driver issue).\n");
        return 0;
    }
    if (deviceCount < numGpus) {
        printf("Skip: need %d GPUs but only %d available.\n", numGpus, deviceCount);
        return 0;
    }

    printf("Smoke test: AllReduce out-of-place SUM float32\n");
    printf("  GPUs: %d, elements per rank: %d\n", numGpus, N);

    std::vector<int> devList(numGpus);
    for (int i = 0; i < numGpus; i++) devList[i] = i;

    ncclComm_t* comms = (ncclComm_t*)malloc(numGpus * sizeof(ncclComm_t));
    CHECK_NCCL(ncclCommInitAll(comms, numGpus, devList.data()));

    size_t bytes = (size_t)N * sizeof(float);
    std::vector<float*> d_in(numGpus, nullptr);
    std::vector<float*> d_out(numGpus, nullptr);
    std::vector<hipStream_t> streams(numGpus, 0);

    for (int i = 0; i < numGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&d_in[i], bytes));
        CHECK_HIP(hipMalloc(&d_out[i], bytes));
        CHECK_HIP(hipStreamCreate(&streams[i]));
    }
    for (int i = 0; i < numGpus; i++) {
        printf("  [host] GPU %d: d_in=%p d_out=%p bytes=%zu (N=%d)\n", i, (void*)d_in[i], (void*)d_out[i], bytes, N);
    }

    // Host buffers: fill input for each rank (rank i has value (i+1) everywhere)
    std::vector<std::vector<float>> h_in(numGpus);
    for (int i = 0; i < numGpus; i++) {
        h_in[i].resize(N, (float)(i + 1));
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMemcpy(d_in[i], h_in[i].data(), bytes, hipMemcpyHostToDevice));
    }

    // AllReduce SUM out-of-place
    ncclGroupStart();
    for (int i = 0; i < numGpus; i++) {
        CHECK_NCCL(ncclAllReduce(d_in[i], d_out[i], N, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    ncclGroupEnd();

    for (int i = 0; i < numGpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipStreamSynchronize(streams[i]));
    }

    // Expected: sum(1..numGpus) = numGpus*(numGpus+1)/2
    float expected = (float)(numGpus * (numGpus + 1) / 2);
    bool pass = true;
    std::vector<float> h_out(N);

    for (int g = 0; g < numGpus && pass; g++) {
        CHECK_HIP(hipSetDevice(g));
        CHECK_HIP(hipMemcpy(h_out.data(), d_out[g], bytes, hipMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) {
            if (h_out[i] != expected) {
                printf("FAIL: GPU %d index %d: got %f expected %f\n", g, i, h_out[i], expected);
                pass = false;
                break;
            }
        }
    }

    for (int i = 0; i < numGpus; i++) {
        CHECK_NCCL(ncclCommDestroy(comms[i]));
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipFree(d_in[i]));
        CHECK_HIP(hipFree(d_out[i]));
        CHECK_HIP(hipStreamDestroy(streams[i]));
    }
    free(comms);

    if (pass) {
        printf("PASS: AllReduce SUM result correct (%.0f on all ranks)\n", expected);
        return 0;
    }
    return 1;
}
