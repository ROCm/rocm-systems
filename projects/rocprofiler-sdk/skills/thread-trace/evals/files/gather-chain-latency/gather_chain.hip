// Sums 256 values gathered through an index table for each of 19456 outputs:
// out[t] = sum over j of data[index[j * THREADS + t]], with random indices into a 64 MB table.
// Build: hipcc -O3 -gline-tables-only --offload-arch=gfx942 gather_chain.hip -o gather_chain
// Run:   ./gather_chain           (checks the result and prints the kernel time)
//        ./gather_chain --once    (one launch, for a profiler capture)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
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

constexpr int THREADS = 304 * 64;  // outputs
constexpr int STEPS = 256;         // gathers per output
constexpr int TABLE = 1 << 24;     // entries in the data table

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

int main(int argc, char** argv) {
    bool once = argc > 1 && std::strcmp(argv[1], "--once") == 0;
    std::vector<int> index(size_t(THREADS) * STEPS);
    std::vector<float> data(TABLE);
    uint64_t s = 7;
    auto next = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (auto& v : index) v = int(next() % TABLE);
    for (auto& v : data) v = float(int(next() % 2001) - 1000) / 1000.0f;
    int* dindex;
    float *ddata, *dout;
    HIP_CHECK(hipMalloc(&dindex, index.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&ddata, data.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, THREADS * sizeof(float)));
    HIP_CHECK(hipMemcpy(dindex, index.data(), index.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddata, data.data(), data.size() * sizeof(float), hipMemcpyHostToDevice));
    gather<<<THREADS / 64, 64>>>(dindex, ddata, dout);
    HIP_CHECK(hipDeviceSynchronize());
    if (once) return 0;

    std::vector<float> result(THREADS);
    HIP_CHECK(hipMemcpy(result.data(), dout, THREADS * sizeof(float), hipMemcpyDeviceToHost));
    for (int t = 0; t < THREADS; ++t) {
        double expect = 0, mag = 0;
        for (int j = 0; j < STEPS; ++j) {
            float v = data[index[size_t(j) * THREADS + t]];
            expect += v;
            mag += std::fabs(v);
        }
        if (std::fabs(result[t] - expect) > 1e-5 * mag + 1e-5) {
            std::printf("FAIL output %d: %f vs %f\n", t, result[t], expect);
            return 1;
        }
    }

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < 20; ++i) gather<<<THREADS / 64, 64>>>(dindex, ddata, dout);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::printf("PASS  %.4f ms\n", ms / 20);
    return 0;
}
