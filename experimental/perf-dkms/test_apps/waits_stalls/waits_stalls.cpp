#include "test_framework.hpp"
#include <vector>

// Kernel designed to create memory wait states and stalls
// Targets GRBM_* wait/stall counters
__global__ void waits_stalls_kernel(float* output, const float* input, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;

    // Create memory dependencies that cause stalls
    for (int i = 0; i < iterations; i++) {
        // Read that creates dependency
        float temp = input[(idx + i) % (blockDim.x * gridDim.x)];

        // Immediate use creates stall waiting for memory
        val = temp * temp + temp;

        // Another dependent read
        temp = input[(idx + (int)(val * 13)) % (blockDim.x * gridDim.x)];

        // More dependency
        val = val + temp;

        // Write that depends on previous operations
        output[idx] = val;

        // Sync to create more stalls
        __syncthreads();
    }

    output[idx] = val;
}

class WaitsStallsTest : public test_framework::TestBase {
public:
    WaitsStallsTest() : TestBase("Waits/Stalls - Memory Dependencies") {}

    bool setup() override {
        size_ = 256 * 1024;  // 256K elements
        iterations_ = 100;

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  Iterations: " << iterations_ << std::endl;
        std::cout << "  Expected behavior: Memory wait states and stalls" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("waits_stalls_kernel", grid_size, block_size);
        runner.launch(waits_stalls_kernel, d_output_, d_input_, iterations_);

        return true;
    }

    void cleanup() override {
        hip_utils::freeDevice(d_input_);
        hip_utils::freeDevice(d_output_);
    }

private:
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
    size_t size_;
    int iterations_;
};

int main() {
    WaitsStallsTest test;
    return test.run();
}
