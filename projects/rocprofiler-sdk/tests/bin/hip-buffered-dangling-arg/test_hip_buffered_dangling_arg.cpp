// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file tests/hip-buffered-dangling-arg/test_hip_buffered_dangling_arg.cpp
 *
 * @brief Workload that calls hipModuleGetFunction with kernel-name strings whose
 * storage is freed before the buffered tracing record is flushed.
 *
 * hipModuleGetFunction takes a `const char* kname` argument. The rocprofiler-sdk
 * HIP tracer records this argument into a buffered (`*_EXT`) record and only
 * stringifies it later, when the buffer is flushed on the rocprofiler callback
 * thread. This test deliberately passes each kernel name via a short-lived
 * std::string temporary whose backing buffer is released as soon as the
 * enclosing scope exits -- i.e. long before the flush. Unless the tracer interns
 * the argument (copies it into stable storage), the flush dereferences a
 * dangling pointer and segfaults.
 *
 * Paired with client.cpp, which enables buffered HIP API tracing and iterates
 * each record's arguments (forcing the stringification / dereference).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (expr);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            std::cerr << "HIP error " << hipGetErrorString(_err) << " at " << __FILE__ << ":"      \
                      << __LINE__ << "\n";                                                         \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
// Kernel source compiled at runtime via hiprtc, then loaded as a module.
const char* kernel_code = R"(
extern "C" __global__ void dynamic_kernel_a(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = idx * 2;
}

extern "C" __global__ void dynamic_kernel_b(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = idx * 3.14f;
}

extern "C" __global__ void dynamic_kernel_c(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = idx * idx;
}
)";

hipModule_t
build_and_load_module()
{
    hiprtcProgram prog{};
    if(hiprtcCreateProgram(&prog, kernel_code, "dynamic_kernels.cu", 0, nullptr, nullptr) !=
       HIPRTC_SUCCESS)
    {
        std::cerr << "hiprtcCreateProgram failed\n";
        std::abort();
    }

    if(hiprtcCompileProgram(prog, 0, nullptr) != HIPRTC_SUCCESS)
    {
        size_t log_size{0};
        hiprtcGetProgramLogSize(prog, &log_size);
        if(log_size > 1)
        {
            auto log = std::vector<char>(log_size);
            hiprtcGetProgramLog(prog, log.data());
            std::cerr << "Compilation failed:\n" << log.data() << "\n";
        }
        std::abort();
    }

    size_t code_size{0};
    hiprtcGetCodeSize(prog, &code_size);
    std::vector<char> code(code_size);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    hipModule_t module{};
    HIP_CHECK(hipModuleLoadData(&module, code.data()));
    return module;
}

// Resolve one kernel, passing its name via a std::string temporary that is
// destroyed the instant this function returns. The const char* handed to
// hipModuleGetFunction is therefore invalid by the time the buffered record is
// flushed.
hipFunction_t
get_function_with_transient_name(hipModule_t module, const std::string& suffix)
{
    hipFunction_t func{};
    std::string   kname = std::string("dynamic_kernel_") + suffix;
    HIP_CHECK(hipModuleGetFunction(&func, module, kname.c_str()));
    return func;
}
}  // namespace

int
main()
{
    std::cout << "HIP buffered dangling-arg test\n";

    int num_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    if(num_gpus == 0)
    {
        std::cerr << "No GPUs found. Test requires at least one GPU.\n";
        return 1;
    }

    HIP_CHECK(hipSetDevice(0));

    // Repeat a few times so the buffer is likely to flush mid-run (not only at
    // finalize), broadening coverage of the deferred-stringify path.
    constexpr int iterations = 8;
    for(int i = 0; i < iterations; ++i)
    {
        hipModule_t module = build_and_load_module();

        hipFunction_t func_a = get_function_with_transient_name(module, "a");
        hipFunction_t func_b = get_function_with_transient_name(module, "b");
        hipFunction_t func_c = get_function_with_transient_name(module, "c");

        const int n = 1024;
        int*      d_int_data{nullptr};
        float*    d_float_data{nullptr};
        HIP_CHECK(hipMalloc(&d_int_data, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_float_data, n * sizeof(float)));

        int   n_arg    = n;
        void* args_a[] = {&d_int_data, &n_arg};
        HIP_CHECK(
            hipModuleLaunchKernel(func_a, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_a, nullptr));
        void* args_b[] = {&d_float_data, &n_arg};
        HIP_CHECK(
            hipModuleLaunchKernel(func_b, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_b, nullptr));
        void* args_c[] = {&d_int_data, &n_arg};
        HIP_CHECK(
            hipModuleLaunchKernel(func_c, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_c, nullptr));

        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipFree(d_int_data));
        HIP_CHECK(hipFree(d_float_data));
        HIP_CHECK(hipModuleUnload(module));
    }

    std::cout << "Test completed successfully\n";
    return 0;
}
