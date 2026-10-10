// Applies an iterated smooth clipping function y = x / (1 + exp(-4 * y)) (32 rounds) to
// 16M floats.
// Build: hipcc -O3 -gline-tables-only --offload-arch=gfx942 soft_clip.hip -o soft_clip
// Run:   ./soft_clip           (checks the result and prints the kernel time)
//        ./soft_clip --once    (one launch, for a profiler capture)
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

constexpr int N = 1 << 24;
constexpr int ROUNDS = 32;

__global__ void soft_clip(const float* __restrict__ in, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[i], y = x;
    for (int r = 0; r < ROUNDS; ++r) y = x / (1.0f + expf(-4.0f * y));
    out[i] = y;
}

int main(int argc, char** argv) {
    bool once = argc > 1 && std::strcmp(argv[1], "--once") == 0;
    std::vector<float> host(N);
    for (int i = 0; i < N; ++i) host[i] = float((i * 37) % 2001 - 1000) / 250.0f;
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&out, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(in, host.data(), N * sizeof(float), hipMemcpyHostToDevice));
    soft_clip<<<N / 256, 256>>>(in, out);
    HIP_CHECK(hipDeviceSynchronize());
    if (once) return 0;

    std::vector<float> result(N);
    HIP_CHECK(hipMemcpy(result.data(), out, N * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < N; i += 997) {
        double x = host[i], y = x;
        for (int r = 0; r < ROUNDS; ++r) y = x / (1.0 + std::exp(-4.0 * y));
        if (std::fabs(result[i] - y) > 1e-4 * (1 + std::fabs(y))) {
            std::printf("FAIL at %d: %f vs %f\n", i, result[i], y);
            return 1;
        }
    }
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < 20; ++i) soft_clip<<<N / 256, 256>>>(in, out);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::printf("PASS  %.4f ms\n", ms / 20);
    return 0;
}
