#include "test_framework.hpp"
#include <vector>

// Wave32 kernel - compile with -mwavefrontsize32 for wave32 mode
__global__ void wave32_kernel(float* output, const float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = input[idx];

    // Wave-level operations
    int lane = __lane_id();

    // Wave shuffle operations (different behavior in wave32 vs wave64)
    #pragma unroll 4
    for (int i = 0; i < 100; i++) {
        val += __shfl_xor(val, 1);
        val += __shfl_xor(val, 2);
        val += __shfl_xor(val, 4);
        val += __shfl_xor(val, 8);
    }

    output[idx] = val + lane;
}

// Wave64 kernel
__global__ void wave64_kernel(float* output, const float* input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = input[idx];

    int lane = __lane_id();

    // Wave shuffle operations
    #pragma unroll 4
    for (int i = 0; i < 100; i++) {
        val += __shfl_xor(val, 1);
        val += __shfl_xor(val, 2);
        val += __shfl_xor(val, 4);
        val += __shfl_xor(val, 8);
        val += __shfl_xor(val, 16);
        val += __shfl_xor(val, 32);
    }

    output[idx] = val + lane;
}

class WaveModesTest : public test_framework::TestBase {
public:
    WaveModesTest() : TestBase("Wave Modes - Wave32/Wave64 Operations") {}

    bool setup() override {
        size_ = 1024 * 1024;  // 1M elements

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        // Get device properties to determine native wave size
        int device;
        HIP_CHECK(hipGetDevice(&device));
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device));

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  Native warp size: " << prop.warpSize << std::endl;
        std::cout << "  Testing wave shuffle operations" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size((size_ + block_size.x - 1) / block_size.x);

        // Test with wave32-style kernel
        std::cout << "  Running wave32-style kernel..." << std::endl;
        test_framework::KernelRunner runner32("wave32_kernel", grid_size, block_size);
        runner32.launch(wave32_kernel, d_output_, d_input_);

        // Test with wave64-style kernel
        std::cout << "  Running wave64-style kernel..." << std::endl;
        test_framework::KernelRunner runner64("wave64_kernel", grid_size, block_size);
        runner64.launch(wave64_kernel, d_output_, d_input_);

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
    WaveModesTest test;
    return test.run();
}
