// Simple HIP kernel for testing bundled binary extraction
// This is intentionally minimal to produce small, predictable test artifacts

#include <hip/hip_runtime.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Simple scalar multiplication kernel
__global__ void scalarMultiply(float* data, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scalar;
    }
}

// Library API function for shared library builds
extern "C" void launch_kernels(float* d_a, float* d_b, float* d_c, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    scalarMultiply<<<numBlocks, blockSize>>>(d_c, 2.0f, n);
}

// Main function for executable builds
#ifdef BUILD_EXECUTABLE
int main() {
    // Minimal main that ensures kernels are compiled and linked
    // Note: We don't actually run the kernels to avoid GPU requirements
    return 0;
}
#endif
