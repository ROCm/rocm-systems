#include <hip/hip_runtime.h>

#define HIP_CALL(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "%s\n", hipGetErrorString(err));                                       \
            abort();                                                                               \
        }                                                                                          \
    } while(0)

// Force the pointer through a device function to generate flat memory access
// (no address space annotation -> generic/flat)
__device__ __noinline__ float flat_load(float *p, int idx) {
    return p[idx];
}

__global__ void addr_trace_kernel(float *out, int n) {
    __shared__ float shm[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Global load
    float val = out[idx];

    // LDS store + barrier + LDS load
    shm[threadIdx.x] = val;
    __syncthreads();
    val += shm[(threadIdx.x + 1) % 256];

    // Global atomic
    atomicAdd(&out[0], val);

    // LDS atomic
    atomicAdd(&shm[0], 1.0f);
    __syncthreads();

    // Flat memory access (through device function)
    val += flat_load(out, idx);

    out[idx] = val + shm[0];
}

int main() {
    const int N = 256;
    float *d_out;
    hipMalloc(&d_out, N * sizeof(float));

    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    hipMemcpy(d_out, h_data, N * sizeof(float), hipMemcpyHostToDevice);

    addr_trace_kernel<<<1, 256>>>(d_out, N);
    hipDeviceSynchronize();

    hipFree(d_out);
    return 0;
}
