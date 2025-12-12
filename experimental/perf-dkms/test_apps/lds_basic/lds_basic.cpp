#include "test_framework.hpp"
#include <vector>

// Basic LDS operations without conflicts
// Targets SQ_LDS_* counters
__global__ void lds_basic_kernel(float* output, const float* input) {
    __shared__ float shared_mem[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Sequential LDS write (no bank conflicts)
    shared_mem[tid] = input[gid];
    __syncthreads();

    // Sequential LDS read (no bank conflicts)
    float val = shared_mem[tid];

    // Some computation
    val = val * 2.0f + 1.0f;

    // Another round of LDS operations
    shared_mem[tid] = val;
    __syncthreads();

    val = shared_mem[tid];

    output[gid] = val;
}

class LDSBasicTest : public test_framework::TestBase {
public:
    LDSBasicTest() : TestBase("LDS Basic - Sequential Access") {}

    bool setup() override {
        size_ = 256 * 1024;  // 256K elements (1024 blocks * 256 threads)

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  LDS size per block: 256 floats (1 KB)" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size(size_ / block_size.x);

        test_framework::KernelRunner runner("lds_basic_kernel", grid_size, block_size);
        runner.launch(lds_basic_kernel, d_output_, d_input_);

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
    LDSBasicTest test;
    return test.run();
}
