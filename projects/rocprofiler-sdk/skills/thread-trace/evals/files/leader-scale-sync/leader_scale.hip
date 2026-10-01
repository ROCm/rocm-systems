// Scales 64 rounds of 256 floats in each of 1216 blocks: round r of every block is
// multiplied by scale(r) = 1 + 1e-3 * sum over k = 1..512 of decay^k / (k + r).
// Build: hipcc -O3 -gline-tables-only --offload-arch=gfx942 leader_scale.hip -o leader_scale
// Run:   ./leader_scale           (checks the result and prints the kernel time)
//        ./leader_scale --once    (one launch, for a profiler capture)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(x)                                                                         \
    do {                                                                                     \
        hipError_t e_ = (x);                                                                 \
        if (e_ != hipSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_));   \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

constexpr int BLOCKS = 304 * 4;
constexpr int ROUNDS = 64;
constexpr int SERIES = 512;  // terms in the per-round scale series

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

int main(int argc, char** argv) {
    bool once = argc > 1 && std::strcmp(argv[1], "--once") == 0;
    const float decay = 0.999f;
    std::vector<float> host(size_t(BLOCKS) * ROUNDS * 256);
    for (size_t i = 0; i < host.size(); ++i) host[i] = float(int(i % 101) - 50);
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&out, host.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(in, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
    leader_scale<<<BLOCKS, 256>>>(in, out, decay);
    HIP_CHECK(hipDeviceSynchronize());
    if (once) return 0;

    std::vector<float> result(host.size());
    HIP_CHECK(hipMemcpy(result.data(), out, host.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (int r = 0; r < ROUNDS; ++r) {
        double s = 0, term = 1;
        for (int k = 1; k <= SERIES; ++k) term *= double(decay), s += term / (k + r);
        double scale = 1 + 1e-3 * s;
        for (int b = 0; b < BLOCKS; ++b) {
            for (int x = 0; x < 256; ++x) {
                size_t i = (size_t(b) * ROUNDS + r) * 256 + x;
                if (std::fabs(result[i] - host[i] * scale) > 1e-4 * (1 + std::fabs(host[i]))) {
                    std::printf("FAIL output %zu: %f vs %f\n", i, result[i], host[i] * scale);
                    return 1;
                }
            }
        }
    }

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < 20; ++i) leader_scale<<<BLOCKS, 256>>>(in, out, decay);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::printf("PASS  %.4f ms\n", ms / 20);
    return 0;
}
