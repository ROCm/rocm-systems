#include "test_framework.hpp"
#include <vector>

// Basic GPU activity test - simple baseline
// Provides baseline activity for GRBM_GUI_ACTIVE and other activity counters
__global__ void basic_activity_kernel(float* output, const float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Simple computation
    float val = input[idx];
    val = val * 2.0f + 1.0f;
    val = val / 3.0f;

    output[idx] = val;
}

class BasicActivityTest : public test_framework::TestBase {
public:
    BasicActivityTest() : TestBase("Basic Activity - Baseline GPU Activity") {}

    bool setup() override {
        size_ = 1024 * 1024;  // 1M elements

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  Purpose: Baseline GPU activity measurement" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("basic_activity_kernel", grid_size, block_size);
        runner.launch(basic_activity_kernel, d_output_, d_input_);

        // Run multiple iterations for more consistent measurements
        std::cout << "  Running 10 iterations for stable measurements..." << std::endl;
        for (int i = 0; i < 10; i++) {
            hipLaunchKernelGGL(basic_activity_kernel, grid_size, block_size, 0, 0,
                              d_output_, d_input_);
        }
        HIP_CHECK(hipDeviceSynchronize());

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
};

int main() {
    BasicActivityTest test;
    return test.run();
}
