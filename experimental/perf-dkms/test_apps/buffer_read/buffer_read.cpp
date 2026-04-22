#include "test_framework.hpp"
#include <vector>

// Coalesced buffer read operations
// Targets TA_* counters (Texture Addresser - memory fetches)
__global__ void buffer_read_kernel(float* output, const float* input, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Perform multiple coalesced reads
    #pragma unroll 4
    for (int i = 0; i < 100; i++) {
        int read_idx = idx + i * stride;
        sum += input[read_idx];
    }

    output[idx] = sum;
}

class BufferReadTest : public test_framework::TestBase {
public:
    BufferReadTest() : TestBase("Buffer Read - Coalesced Memory Reads") {}

    bool setup() override {
        size_ = 16 * 1024 * 1024;  // 16M elements for reading
        output_size_ = 1024 * 1024;  // 1M output elements
        stride_ = 16;

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(output_size_);

        std::cout << "  Input array size: " << size_ << " elements ("
                  << (size_ * sizeof(float) / (1024*1024)) << " MB)" << std::endl;
        std::cout << "  Output array size: " << output_size_ << " elements" << std::endl;
        std::cout << "  Reads per thread: 100" << std::endl;
        std::cout << "  Total memory reads: " << (output_size_ * 100 * sizeof(float) / (1024*1024))
                  << " MB" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((output_size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("buffer_read_kernel", grid_size, block_size);
        runner.launch(buffer_read_kernel, d_output_, d_input_, stride_);

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
    size_t output_size_;
    int stride_;
};

int main() {
    BufferReadTest test;
    return test.run();
}
