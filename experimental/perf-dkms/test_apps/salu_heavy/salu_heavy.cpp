#include "test_framework.hpp"
#include <vector>

// SALU-heavy kernel: Scalar operations
// Targets SQ_INSTS_SALU counter
__global__ void salu_heavy_kernel(int* output, const int* input, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Scalar operations using __lane_id() and wave-uniform values
    int lane = __lane_id();
    int scalar_val = blockIdx.x;  // Scalar uniform across wavefront

    int result = input[idx];

    // Perform many scalar operations
    #pragma unroll 4
    for (int i = 0; i < iterations; i++) {
        // Scalar ALU operations
        scalar_val = scalar_val + 1;
        scalar_val = scalar_val << 1;
        scalar_val = scalar_val ^ 0xABCD;
        scalar_val = scalar_val >> 1;
        scalar_val = scalar_val & 0xFFFF;
        scalar_val = scalar_val | 0x1000;

        // Use scalar value with vector
        result += scalar_val;
    }

    output[idx] = result + lane;
}

class SALUHeavyTest : public test_framework::TestBase {
public:
    SALUHeavyTest() : TestBase("SALU Heavy - Scalar Operations") {}

    bool setup() override {
        size_ = 1024 * 1024;  // 1M elements
        iterations_ = 1000;

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<int>(size_, 1);
        d_output_ = hip_utils::allocateDevice<int>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  Iterations per thread: " << iterations_ << std::endl;
        std::cout << "  Total SALU operations: " << (size_ * iterations_ * 6) << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("salu_heavy_kernel", grid_size, block_size);
        runner.launch(salu_heavy_kernel, d_output_, d_input_, iterations_);

        return true;
    }

    void cleanup() override {
        hip_utils::freeDevice(d_input_);
        hip_utils::freeDevice(d_output_);
    }

private:
    int* d_input_ = nullptr;
    int* d_output_ = nullptr;
    size_t size_;
    int iterations_;
};

int main() {
    SALUHeavyTest test;
    return test.run();
}
