#include "test_framework.hpp"
#include <vector>

// Coalesced buffer write operations
// Targets memory write counters
__global__ void buffer_write_kernel(float* output, const float* input) {
    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = input[base_idx];

    // Perform multiple coalesced writes
    #pragma unroll 4
    for (int i = 0; i < 100; i++) {
        int write_idx = base_idx + i * blockDim.x * gridDim.x;
        output[write_idx] = val * (1.0f + i * 0.01f);
    }
}

class BufferWriteTest : public test_framework::TestBase {
public:
    BufferWriteTest() : TestBase("Buffer Write - Coalesced Memory Writes") {}

    bool setup() override {
        input_size_ = 1024 * 1024;  // 1M input elements
        output_size_ = 100 * 1024 * 1024;  // 100M output elements

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(input_size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(output_size_);

        std::cout << "  Input array size: " << input_size_ << " elements" << std::endl;
        std::cout << "  Output array size: " << output_size_ << " elements ("
                  << (output_size_ * sizeof(float) / (1024*1024)) << " MB)" << std::endl;
        std::cout << "  Writes per thread: 100" << std::endl;
        std::cout << "  Total memory writes: " << (output_size_ * sizeof(float) / (1024*1024))
                  << " MB" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((input_size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("buffer_write_kernel", grid_size, block_size);
        runner.launch(buffer_write_kernel, d_output_, d_input_);

        return true;
    }

    void cleanup() override {
        hip_utils::freeDevice(d_input_);
        hip_utils::freeDevice(d_output_);
    }

private:
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;
    size_t input_size_;
    size_t output_size_;
};

int main() {
    BufferWriteTest test;
    return test.run();
}
