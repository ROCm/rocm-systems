#include "test_framework.hpp"
#include <vector>

// LDS bank conflict test - stride-32 access pattern
// Targets SQ_LDS_BANK_CONFLICT counter
__global__ void lds_conflicts_kernel(float* output, const float* input) {
    __shared__ float shared_mem[1024];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Write with stride to cause bank conflicts
    // On AMD GPUs, LDS has 32 banks
    int write_idx = (tid * 32) % 1024;
    shared_mem[write_idx] = input[gid];
    __syncthreads();

    // Read with stride to cause bank conflicts
    int read_idx = (tid * 32 + 1) % 1024;
    float val = shared_mem[read_idx];

    // Multiple rounds of conflicting access
    for (int i = 0; i < 10; i++) {
        write_idx = ((tid + i) * 32) % 1024;
        shared_mem[write_idx] = val * 1.01f;
        __syncthreads();

        read_idx = ((tid + i + 1) * 32) % 1024;
        val = shared_mem[read_idx];
        __syncthreads();
    }

    output[gid] = val;
}

class LDSConflictsTest : public test_framework::TestBase {
public:
    LDSConflictsTest() : TestBase("LDS Conflicts - Bank Conflicts") {}

    bool setup() override {
        size_ = 256 * 1024;  // 256K elements

        // Allocate device memory
        d_input_ = hip_utils::allocateDeviceWithValue<float>(size_, 1.0f);
        d_output_ = hip_utils::allocateDevice<float>(size_);

        std::cout << "  Array size: " << size_ << " elements" << std::endl;
        std::cout << "  LDS size per block: 1024 floats (4 KB)" << std::endl;
        std::cout << "  Access pattern: stride-32 (causes bank conflicts)" << std::endl;

        return true;
    }

    bool execute() override {
        dim3 block_size(256);
        dim3 grid_size(size_ / block_size.x);

        test_framework::KernelRunner runner("lds_conflicts_kernel", grid_size, block_size);
        runner.launch(lds_conflicts_kernel, d_output_, d_input_);

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
    LDSConflictsTest test;
    return test.run();
}
