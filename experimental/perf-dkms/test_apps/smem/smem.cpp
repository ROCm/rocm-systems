#include "test_framework.hpp"
#include <vector>

// Constant memory for scalar memory operations
__constant__ float const_data[1024];

// Scalar memory operations - constant memory reads
// Targets SQ_INSTS_SMEM counter
__global__ void smem_kernel(float* output, const float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = input[idx];

    // Multiple scalar memory reads from constant memory
    #pragma unroll 8
    for (int i = 0; i < 100; i++) {
        int const_idx = (idx + i) % 1024;
        val += const_data[const_idx];
        val *= const_data[(const_idx + 1) % 1024];
    }

    output[idx] = val;
}

class SMEMTest : public test_framework::TestBase {
public:
    SMEMTest() : TestBase("SMEM - Scalar Memory (Constant Memory)") {}

    bool setup() override {
        size_ = 1024 * 1024;  // 1M elements

        // Initialize constant memory
        std::vector<float> h_const(1024);
        for (int i = 0; i < 1024; i++) {
            h_const[i] = 1.0f + i * 0.001f;
        }
        HIP_CHECK(hipMemcpyToSymbol(const_data, h_const.data(),
                                     1024 * sizeof(float), 0, hipMemcpyHostToDevice));

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  Constant memory size: 1024 floats" << std::endl;
        std::cout << "  Scalar memory reads per thread: 200" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("smem_kernel", grid_size, block_size);
        runner.launch(smem_kernel, d_output_, d_input_);

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
    SMEMTest test;
    return test.run();
}
