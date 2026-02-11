/*
 * Smoke test 2: Batched AllReduce, out-of-place, SUM, float32.
 * Single-process only. Submits multiple AllReduce ops in one group; verifies all.
 *
 * Usage: test_smoke_allreduce_batched [num_gpus] [array_length] [num_batches]
 *   num_gpus     Number of GPUs (default 2)
 *   array_length Elements per rank per batch (default 1024)
 *   num_batches  Number of AllReduce ops in the group (default 16)
 *
 * Each batch b: rank i fills input with (i+1) + b*1000. Expected result for batch b:
 *   sum over ranks = num_gpus*(num_gpus+1)/2 + num_gpus*b*1000.
 */

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static int getIntArg(int argc, char** argv, int idx, int defaultVal) {
    if (idx < argc) {
        int v = atoi(argv[idx]);
        if (v > 0) return v;
    }
    return defaultVal;
}

int main(int argc, char** argv) {
    int numGpus   = getIntArg(argc, argv, 1, 2);
    int N         = getIntArg(argc, argv, 2, 1024);
    int numBatches = getIntArg(argc, argv, 3, 16);

    if (numGpus < 2) {
        printf("Skip: requires at least 2 GPUs (got %d). Single-GPU run is not a real collective.\n", numGpus);
        return 0;
    }

    printf("Smoke test: Batched AllReduce out-of-place SUM float32\n");
    printf("  GPUs: %d, elements per rank: %d, batches: %d\n", numGpus, N, numBatches);

    std::vector<int> devList(numGpus);
    for (int i = 0; i < numGpus; i++) devList[i] = i;

    ncclComm_t* comms = (ncclComm_t*)malloc(numGpus * sizeof(ncclComm_t));
    if (ncclCommInitAll(comms, numGpus, devList.data()) != ncclSuccess) {
        printf("FAIL: ncclCommInitAll\n");
        return 1;
    }

    size_t bytes = (size_t)N * sizeof(float);
    std::vector<std::vector<float*>> d_in(numGpus);
    std::vector<std::vector<float*>> d_out(numGpus);
    std::vector<hipStream_t> streams(numGpus, 0);

    for (int i = 0; i < numGpus; i++) {
        d_in[i].resize(numBatches, nullptr);
        d_out[i].resize(numBatches, nullptr);
        hipSetDevice(i);
        for (int b = 0; b < numBatches; b++) {
            if (hipMalloc(&d_in[i][b], bytes) != hipSuccess || hipMalloc(&d_out[i][b], bytes) != hipSuccess) {
                printf("FAIL: hipMalloc GPU %d batch %d\n", i, b);
                return 1;
            }
        }
        if (hipStreamCreate(&streams[i]) != hipSuccess) {
            printf("FAIL: hipStreamCreate on GPU %d\n", i);
            return 1;
        }
    }

    // Fill inputs: rank i batch b has value (i+1) + b*1000
    for (int i = 0; i < numGpus; i++) {
        hipSetDevice(i);
        for (int b = 0; b < numBatches; b++) {
            float val = (float)((i + 1) + b * 1000);
            std::vector<float> h_in(N, val);
            hipMemcpy(d_in[i][b], h_in.data(), bytes, hipMemcpyHostToDevice);
        }
    }

    // One group: all batches of AllReduce
    ncclGroupStart();
    for (int b = 0; b < numBatches; b++) {
        for (int i = 0; i < numGpus; i++) {
            ncclAllReduce(d_in[i][b], d_out[i][b], N, ncclFloat, ncclSum, comms[i], streams[i]);
        }
    }
    ncclGroupEnd();

    for (int i = 0; i < numGpus; i++) {
        hipSetDevice(i);
        hipStreamSynchronize(streams[i]);
    }

    // Expected for batch b: numGpus*(numGpus+1)/2 + numGpus*b*1000
    bool pass = true;
    std::vector<float> h_out(N);

    for (int g = 0; g < numGpus && pass; g++) {
        hipSetDevice(g);
        for (int b = 0; b < numBatches; b++) {
            float expected = (float)(numGpus * (numGpus + 1) / 2 + numGpus * b * 1000);
            hipMemcpy(h_out.data(), d_out[g][b], bytes, hipMemcpyDeviceToHost);
            for (int i = 0; i < N; i++) {
                if (h_out[i] != expected) {
                    printf("FAIL: GPU %d batch %d index %d: got %f expected %f\n", g, b, i, h_out[i], expected);
                    pass = false;
                    break;
                }
            }
            if (!pass) break;
        }
    }

    for (int i = 0; i < numGpus; i++) {
        ncclCommDestroy(comms[i]);
        hipSetDevice(i);
        for (int b = 0; b < numBatches; b++) {
            hipFree(d_in[i][b]);
            hipFree(d_out[i][b]);
        }
        hipStreamDestroy(streams[i]);
    }
    free(comms);

    if (pass) {
        printf("PASS: %d batched AllReduce SUM results correct\n", numBatches);
        return 0;
    }
    return 1;
}
