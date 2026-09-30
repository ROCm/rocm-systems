#include "test_framework.hpp"
#include <vector>

// Test various memory transaction sizes (32B, 64B, 128B)
// Targets GL2C transaction counters
__global__ void mem_sizes_32b_kernel(int* output, const int* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 32-byte transactions (8 x 4-byte ints)
    int val = input[idx];
    output[idx] = val + 1;
}

__global__ void mem_sizes_64b_kernel(int2* output, const int2* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 64-byte transactions (8 x 8-byte int2)
    int2 val = input[idx];
    val.x += 1;
    val.y += 1;
    output[idx] = val;
}

__global__ void mem_sizes_128b_kernel(int4* output, const int4* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 128-byte transactions (8 x 16-byte int4)
    int4 val = input[idx];
    val.x += 1;
    val.y += 1;
    val.z += 1;
    val.w += 1;
    output[idx] = val;
}

class MemSizesTest : public test_framework::TestBase {
public:
    MemSizesTest() : TestBase("Memory Sizes - 32B/64B/128B Transactions") {}

    bool setup() override {
        size_ = 1024 * 1024;  // 1M elements

        // Allocate for different sizes
        d_input_32_ = hip_utils::allocateDeviceWithValue<int>(size_, 1);
        d_output_32_ = hip_utils::allocateDevice<int>(size_);

        d_input_64_ = hip_utils::allocateDeviceWithValue<int2>(size_, make_int2(1, 1));
        d_output_64_ = hip_utils::allocateDevice<int2>(size_);

        d_input_128_ = hip_utils::allocateDeviceWithValue<int4>(size_, make_int4(1, 1, 1, 1));
        d_output_128_ = hip_utils::allocateDevice<int4>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  Testing 32-byte, 64-byte, and 128-byte transactions" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        // Test 32-byte transactions
        std::cout << "  Running 32-byte transaction test..." << std::endl;
        test_framework::KernelRunner runner32("mem_sizes_32b_kernel", grid_size, block_size);
        runner32.launch(mem_sizes_32b_kernel, d_output_32_, d_input_32_);

        // Test 64-byte transactions
        std::cout << "  Running 64-byte transaction test..." << std::endl;
        test_framework::KernelRunner runner64("mem_sizes_64b_kernel", grid_size, block_size);
        runner64.launch(mem_sizes_64b_kernel, d_output_64_, d_input_64_);

        // Test 128-byte transactions
        std::cout << "  Running 128-byte transaction test..." << std::endl;
        test_framework::KernelRunner runner128("mem_sizes_128b_kernel", grid_size, block_size);
        runner128.launch(mem_sizes_128b_kernel, d_output_128_, d_input_128_);

        return true;
    }

    void cleanup() override {
        hip_utils::freeDevice(d_input_32_);
        hip_utils::freeDevice(d_output_32_);
        hip_utils::freeDevice(d_input_64_);
        hip_utils::freeDevice(d_output_64_);
        hip_utils::freeDevice(d_input_128_);
        hip_utils::freeDevice(d_output_128_);
    }

private:
    int* d_input_32_ = nullptr;
    int* d_output_32_ = nullptr;
    int2* d_input_64_ = nullptr;
    int2* d_output_64_ = nullptr;
    int4* d_input_128_ = nullptr;
    int4* d_output_128_ = nullptr;
    size_t size_;
};

int main() {
    MemSizesTest test;
    return test.run();
}
