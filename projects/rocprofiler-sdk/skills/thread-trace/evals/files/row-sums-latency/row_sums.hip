// Sums each row of a 1216 x 16384 float matrix: one 64-lane wave per row.
// Build: hipcc -O3 -gline-tables-only --offload-arch=gfx942 row_sums.hip -o row_sums
// Run:   ./row_sums           (checks the result and prints the kernel time)
//        ./row_sums --once    (one launch, for a profiler capture)
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

constexpr int ROWS = 1216;
constexpr int COLS = 16384;

__global__ void row_sums(const float* __restrict__ in, float* __restrict__ out) {
    const float* row = in + size_t(blockIdx.x) * COLS;
    float acc = 0.0f;
#pragma unroll 1
    for (int c = threadIdx.x; c < COLS; c += 64) acc += row[c];
    for (int offset = 32; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    bool once = argc > 1 && std::strcmp(argv[1], "--once") == 0;
    std::vector<float> host(size_t(ROWS) * COLS);
    for (size_t i = 0; i < host.size(); ++i) host[i] = float(int(i % 7) - 3);
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, host.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&out, ROWS * sizeof(float)));
    HIP_CHECK(hipMemcpy(in, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
    row_sums<<<ROWS, 64>>>(in, out);
    HIP_CHECK(hipDeviceSynchronize());
    if (once) return 0;

    std::vector<float> result(ROWS);
    HIP_CHECK(hipMemcpy(result.data(), out, ROWS * sizeof(float), hipMemcpyDeviceToHost));
    for (int r = 0; r < ROWS; ++r) {
        double expect = 0;
        for (int c = 0; c < COLS; ++c) expect += host[size_t(r) * COLS + c];
        if (std::fabs(result[r] - expect) > 1e-3 * (1 + std::fabs(expect))) {
            std::printf("FAIL row %d: %f vs %f\n", r, result[r], expect);
            return 1;
        }
    }

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < 50; ++i) row_sums<<<ROWS, 64>>>(in, out);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    ms /= 50;
    std::printf("PASS  %.4f ms  %.0f GB/s\n", ms, host.size() * sizeof(float) / (ms * 1e6));
    return 0;
}
