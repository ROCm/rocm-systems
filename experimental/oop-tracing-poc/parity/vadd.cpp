// Parity test app: a few HIP calls that exercise memory, copies, a kernel and streams.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

__global__ void vadd(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) c[i] = a[i] + b[i];
}

#define CHECK(x)                                                                          \
    do                                                                                    \
    {                                                                                     \
        hipError_t _e = (x);                                                              \
        if(_e != hipSuccess)                                                              \
        {                                                                                 \
            std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(_e));           \
            return 1;                                                                     \
        }                                                                                 \
    } while(0)

int main(int argc, char** argv)
{
    int n     = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int iters = (argc > 2) ? std::atoi(argv[2]) : 3;
    auto ha   = std::vector<float>(n, 1.0f);
    auto hb   = std::vector<float>(n, 2.0f);
    auto hc   = std::vector<float>(n, 0.0f);
    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;
    hipStream_t s{};
    CHECK(hipSetDevice(0));
    CHECK(hipStreamCreate(&s));
    CHECK(hipMalloc(&da, n * sizeof(float)));
    CHECK(hipMalloc(&db, n * sizeof(float)));
    CHECK(hipMalloc(&dc, n * sizeof(float)));
    CHECK(hipMemcpyAsync(da, ha.data(), n * sizeof(float), hipMemcpyHostToDevice, s));
    CHECK(hipMemcpyAsync(db, hb.data(), n * sizeof(float), hipMemcpyHostToDevice, s));
    for(int i = 0; i < iters; ++i)
    {
        hipLaunchKernelGGL(vadd, dim3((n + 255) / 256), dim3(256), 0, s, da, db, dc, n);
        CHECK(hipGetLastError());
    }
    CHECK(hipMemcpyAsync(hc.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
    CHECK(hipFree(da));
    CHECK(hipFree(db));
    CHECK(hipFree(dc));
    CHECK(hipStreamDestroy(s));
    std::printf("c[0]=%g c[n-1]=%g (expect 3)\n", hc[0], hc[n - 1]);
    return (hc[0] == 3.0f && hc[n - 1] == 3.0f) ? 0 : 2;
}
