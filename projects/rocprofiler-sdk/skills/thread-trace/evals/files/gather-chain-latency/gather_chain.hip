// Gathers values through an index table (see kernel_api.h).
// Build: hipcc -O3 -gline-tables-only --offload-arch=gfx942 harness.cpp gather_chain.hip -o gather_chain
// Run:   ./gather_chain           (checks the result and prints the time of gather_launch)
//        ./gather_chain --once    (one call, for a profiler capture)
// harness.cpp and kernel_api.h are fixed; change only this file.
#include "kernel_api.h"

__global__ void gather(const int* __restrict__ index, const float* __restrict__ data,
                       float* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
#pragma unroll 1
    for (int j = 0; j < STEPS; ++j) {
        int i = index[size_t(j) * THREADS + t];
        acc += data[i];
    }
    out[t] = acc;
}

void gather_launch(const int* index, const float* data, float* out) {
    gather<<<THREADS / 64, 64>>>(index, data, out);
}
