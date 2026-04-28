#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

static constexpr auto kernel_src{R"(
extern "C" __global__ void test_kernel(int* data) {
    int idx = threadIdx.x;
    data[idx] = idx * 2;
    __syncthreads();
    data[idx] += 1;
}
)"};

int main() {
    std::cout << "Test: hipRTC + __syncthreads() + amdgcnspirv" << std::endl;

    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, kernel_src, "test.hip", 0, nullptr, nullptr);

    const char* options[] = {"-xhip", "--offload-arch=amdgcnspirv"};
    hiprtcResult compileResult = hiprtcCompileProgram(prog, 2, options);

    size_t logSize;
    hiprtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::string log(logSize, 0);
        hiprtcGetProgramLog(prog, &log[0]);
        std::cout << "Log: " << log << std::endl;
    }

    if (compileResult != HIPRTC_SUCCESS) {
        std::cerr << "COMPILE FAILED" << std::endl;
        return 1;
    }
    std::cout << "Compile OK" << std::endl;

    size_t bcSize;
    hiprtcGetBitcodeSize(prog, &bcSize);
    std::cout << "Bitcode: " << bcSize << " bytes" << std::endl;

    std::vector<char> bc(bcSize);
    hiprtcGetBitcode(prog, bc.data());
    hiprtcDestroyProgram(&prog);

    hipLinkState_t state;
    hipLinkCreate(0, nullptr, nullptr, &state);

    hipError_t err = hipLinkAddData(state, hipJitInputSpirv, bc.data(), bc.size(), "t.spv", 0, nullptr, nullptr);
    if (err != hipSuccess) {
        std::cerr << "hipLinkAddData FAILED: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    void* bin = nullptr;
    size_t binSize = 0;
    err = hipLinkComplete(state, &bin, &binSize);
    if (err != hipSuccess) {
        std::cerr << "hipLinkComplete FAILED: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    hipModule_t module;
    err = hipModuleLoadData(&module, bin);
    if (err != hipSuccess) {
        std::cerr << "hipModuleLoadData FAILED: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "SUCCESS" << std::endl;
    return 0;
}
