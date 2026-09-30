#include "test_framework.hpp"
#include <vector>

// FLAT memory addressing (generic pointers)
// Targets SQ_INSTS_FLAT counter
__global__ void flat_memory_kernel(float* output, float* input, int use_flat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use FLAT addressing by going through generic pointer
    // Cast through void* to force FLAT addressing instead of buffer operations
    float* generic_ptr = use_flat ? (float*)((void*)input) : input;
    float val = generic_ptr[idx];

    // Multiple FLAT operations
    for (int i = 0; i < 100; i++) {
        // FLAT load
        float tmp = generic_ptr[(idx + i) % (blockDim.x * gridDim.x)];
        val = val * 0.99f + tmp * 0.01f;

        // FLAT store
        generic_ptr[idx] = val;
    }

    output[idx] = val;
}

class FlatMemoryTest : public test_framework::TestBase {
public:
    FlatMemoryTest() : TestBase("FLAT Memory - Generic Pointer Operations") {}

    bool setup() override {
        size_ = 1024 * 1024;  // 1M elements

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  FLAT operations per thread: 200 (100 loads + 100 stores)" << std::endl;
        std::cout << "  Total FLAT operations: " << (size_ * 200) << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        test_framework::KernelRunner runner("flat_memory_kernel", grid_size, block_size);
        runner.launch(flat_memory_kernel, d_output_, d_input_, 1);

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
    FlatMemoryTest test;
    return test.run();
}
