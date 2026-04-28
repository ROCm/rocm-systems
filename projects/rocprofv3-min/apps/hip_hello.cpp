#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#define HIP_CHECK(expr)                                                    \
    do {                                                                   \
        hipError_t _e = (expr);                                            \
        if (_e != hipSuccess) {                                            \
            std::fprintf(stderr, "HIP error %d at %s:%d: %s\n",            \
                         (int)_e, __FILE__, __LINE__, hipGetErrorString(_e)); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

__global__ void hello_world_kernel(char* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Simple, deterministic mutation: uppercase ASCII letters.
        char c = buf[i];
        if (c >= 'a' && c <= 'z') buf[i] = c - ('a' - 'A');
    }
}

int main() {
    const char original[] = "hello world from hip";
    const int n = (int)sizeof(original); // includes terminator

    char* d_buf = nullptr;
    HIP_CHECK(hipMalloc(&d_buf, n));
    HIP_CHECK(hipMemcpy(d_buf, original, n, hipMemcpyHostToDevice));

    const int tpb = 64;
    const int blocks = (n + tpb - 1) / tpb;
    hello_world_kernel<<<blocks, tpb>>>(d_buf, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    char modified[64] = {0};
    HIP_CHECK(hipMemcpy(modified, d_buf, n, hipMemcpyDeviceToHost));

    std::printf("Original message: %s\n", original);
    std::printf("Modified message: %s\n", modified);

    HIP_CHECK(hipFree(d_buf));
    return 0;
}
