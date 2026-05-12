#include "test_framework.hpp"
#include <vector>

// VALU-heavy kernel: FMA (Fused Multiply-Add) operations
// Targets SQ_INSTS_VALU counter
__global__ void valu_heavy_kernel(float* output, const float* input, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float a = input[idx];
    float b = 1.00001f;
    float c = 0.99999f;

    // Perform many VALU operations (FMA)
    #pragma unroll 8
    for (int i = 0; i < iterations; i++) {
        a = a * b + c;      // FMA
        a = a * c + b;      // FMA
        a = a * b + c;      // FMA
        a = a * c + b;      // FMA
    }

    output[idx] = a;
}

class VALUHeavyTest : public test_framework::TestBase {
public:
    VALUHeavyTest() : TestBase("VALU Heavy - FMA Operations") {}

    bool setup() override {
        size_ = 1024 * 1024;  // 1M elements
        iterations_ = 1000;

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  Iterations per thread: " << iterations_ << std::endl;
        std::cout << "  Total VALU operations: " << (size_ * iterations_ * 4) << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("valu_heavy_kernel", grid_size, block_size);
        runner.launch(valu_heavy_kernel, d_output_, d_input_, iterations_);

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
    VALUHeavyTest test;
    return test.run();
}
