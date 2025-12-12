#include "test_framework.hpp"
#include <vector>

// L2 cache hit test - small working set
// Targets GL2C hit counters
__global__ void cache_hit_kernel(float* output, const float* input, int small_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Repeatedly access small working set to maximize cache hits
    #pragma unroll 4
    for (int i = 0; i < 1000; i++) {
        int read_idx = (idx + i) % small_size;  // Small working set
        sum += input[read_idx];
    }

    output[idx] = sum;
}

class CacheHitTest : public test_framework::TestBase {
public:
    CacheHitTest() : TestBase("Cache Hit - L2 Cache Hits") {}

    bool setup() override {
        // Small working set that fits in L2 cache (typical L2: 4-6 MB)
        small_size_ = 256 * 1024;  // 256K floats = 1 MB
        output_size_ = 1024 * 1024;  // 1M threads

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(small_size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(output_size_);

        std::cout << "  Small working set: " << small_size_ << " floats ("
                  << (small_size_ * sizeof(float) / (1024*1024)) << " MB)" << std::endl;
        std::cout << "  Output size: " << output_size_ << " elements" << std::endl;
        std::cout << "  Accesses per thread: 1000" << std::endl;
        std::cout << "  Expected behavior: High L2 cache hit rate" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((output_size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("cache_hit_kernel", grid_size, block_size);
        runner.launch(cache_hit_kernel, d_output_, d_input_, small_size_);

        return true;
    }

    void cleanup() override {
        hip_utils::freeDevice(d_input_);
        hip_utils::freeDevice(d_output_);
    }

private:
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
    size_t small_size_;
    size_t output_size_;
};

int main() {
    CacheHitTest test;
    return test.run();
}
