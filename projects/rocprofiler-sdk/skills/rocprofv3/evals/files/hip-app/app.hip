// Small HIP workload with deliberate performance problems for profiling evals:
// pageable host buffers, a strided (uncoalesced) copy, a compute-bound loop, and
// hundreds of tiny launches per iteration.

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                               \
    do                                                                                         \
    {                                                                                          \
        hipError_t e = (x);                                                                    \
        if(e != hipSuccess)                                                                    \
        {                                                                                      \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e));          \
            exit(1);                                                                           \
        }                                                                                      \
    } while(0)

__global__ void
stream_copy(const float* __restrict__ in, float* __restrict__ out, size_t n)
{
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if(i < n) out[i] = in[i];
}

__global__ void
strided_copy(const float* __restrict__ in, float* __restrict__ out, size_t n, int stride)
{
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if(i < n) out[i] = in[(i * stride) % n];
}

__global__ void
fma_loop(float* __restrict__ data, size_t n, int iters)
{
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if(i >= n) return;
    float a = data[i], b = 1.0001f, c = 0.9999f;
    for(int k = 0; k < iters; ++k)
    {
        a = a * b + c;
        b = b * c + a * 1e-7f;
    }
    data[i] = a + b;
}

__global__ void
tiny_kernel(float* data)
{
    data[threadIdx.x] += 1.0f;
}

int
main(int argc, char** argv)
{
    size_t n     = (argc > 1) ? strtoull(argv[1], nullptr, 10) : (16ull << 20);
    int    iters = (argc > 2) ? atoi(argv[2]) : 3;
    size_t bytes = n * sizeof(float);

    std::vector<float> host(n, 1.0f);
    float *d_in = nullptr, *d_out = nullptr;
    CHECK(hipMalloc(&d_in, bytes));
    CHECK(hipMalloc(&d_out, bytes));

    roctxRangePush("h2d");
    CHECK(hipMemcpy(d_in, host.data(), bytes, hipMemcpyHostToDevice));
    roctxRangePop();

    dim3 block(256), grid((n + 255) / 256);
    for(int it = 0; it < iters; ++it)
    {
        roctxRangePush("iteration");
        stream_copy<<<grid, block>>>(d_in, d_out, n);
        strided_copy<<<grid, block>>>(d_in, d_out, n, 33);
        fma_loop<<<grid, block>>>(d_out, n, 256);
        for(int t = 0; t < 200; ++t)
            tiny_kernel<<<1, 64>>>(d_out);
        CHECK(hipDeviceSynchronize());
        roctxRangePop();
    }

    roctxRangePush("d2h");
    CHECK(hipMemcpy(host.data(), d_out, bytes, hipMemcpyDeviceToHost));
    roctxRangePop();

    printf("result[0]=%f\n", host[0]);
    CHECK(hipFree(d_in));
    CHECK(hipFree(d_out));
    return 0;
}
