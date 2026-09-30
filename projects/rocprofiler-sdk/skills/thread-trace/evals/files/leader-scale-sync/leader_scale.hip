// Scales each round's slice of every block by a per-round factor (see kernel_api.h).
// Build: hipcc -O3 -gline-tables-only --offload-arch=gfx942 harness.cpp leader_scale.hip -o leader_scale
// Run:   ./leader_scale           (checks the result and prints the time of leader_scale_launch)
//        ./leader_scale --once    (one call, for a profiler capture)
// harness.cpp and kernel_api.h are fixed; change only this file.
#include "kernel_api.h"

__device__ float round_scale(int round, float decay) {
    // A slowly converging series.
    float s = 0.0f, term = 1.0f;
    for (int k = 1; k <= SERIES; ++k) {
        term *= decay;
        s = fmaf(term, 1.0f / float(k + round), s);
    }
    return 1.0f + 1e-3f * s;
}

__global__ void leader_scale(const float* __restrict__ in, float* __restrict__ out, float decay) {
    __shared__ float scale;
    const float* src = in + size_t(blockIdx.x) * ROUNDS * 256;
    float* dst = out + size_t(blockIdx.x) * ROUNDS * 256;
    for (int r = 0; r < ROUNDS; ++r) {
        if (threadIdx.x == 0) scale = round_scale(r, decay);
        __syncthreads();
        dst[r * 256 + threadIdx.x] = src[r * 256 + threadIdx.x] * scale;
        __syncthreads();
    }
}

void leader_scale_launch(const float* in, float* out, float decay) {
    leader_scale<<<BLOCKS, 256>>>(in, out, decay);
}
