#include "test_framework.hpp"
#include <vector>

// L2 cache miss test - large streaming access
// Targets GL2C miss counters
__global__ void cache_miss_kernel(float* output, const float* input, size_t large_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Streaming access through large array to maximize cache misses
    size_t stride = blockDim.x * gridDim.x;
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
        size_t read_idx = (idx + i * stride) % large_size;
        sum += input[read_idx];
    }

    output[idx] = sum;
}

class CacheMissTest : public test_framework::TestBase {
public:
    CacheMissTest() : TestBase("Cache Miss - L2 Cache Misses") {}

    bool setup() override {
        // Large working set that doesn't fit in L2 cache
        large_size_ = 64 * 1024 * 1024;  // 64M floats = 256 MB
        output_size_ = 1024 * 1024;  // 1M threads

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(large_size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(output_size_);

        std::cout << "  Large working set: " << large_size_ << " floats ("
                  << (large_size_ * sizeof(float) / (1024*1024)) << " MB)" << std::endl;
        std::cout << "  Output size: " << output_size_ << " elements" << std::endl;
        std::cout << "  Access pattern: Streaming (strided)" << std::endl;
        std::cout << "  Expected behavior: High L2 cache miss rate" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((output_size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("cache_miss_kernel", grid_size, block_size);
        runner.launch(cache_miss_kernel, d_output_, d_input_, large_size_);

        return true;
    }

    void cleanup() override {
        hip_utils::freeDevice(d_input_);
        hip_utils::freeDevice(d_output_);
    }

private:
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
    size_t large_size_;
    size_t output_size_;
};

int main() {
    CacheMissTest test;
    return test.run();
}
