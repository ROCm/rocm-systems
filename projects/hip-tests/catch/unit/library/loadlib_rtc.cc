/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <string>
#include <vector>

static std::vector<char> compile_using_hiprtc(const std::string& code, std::string gpu_arch) {
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, code.c_str(), "code.cu", 0, NULL, NULL));
  #ifdef __HIP_PLATFORM_AMD__
    std::string offload_arch = "--offload-arch=" + gpu_arch;
  #else
    std::string offload_arch = "--fmad=false";
  #endif
  const char* opts[] = {offload_arch.c_str()};
  HIPRTC_CHECK(hiprtcCompileProgram(prog, 1, opts));
  size_t size;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &size));
  std::vector<char> res(size, 0);
  HIPRTC_CHECK(hiprtcGetCode(prog, res.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return res;
}

TEST_CASE("Unit_hip_library_load_rtc") {
  constexpr size_t size = 32;
  const std::string kernel1 =
      "extern \"C\" __global__ void add_kernel(float* out, float*a, float*b) { size_t i = "
      "threadIdx.x; out[i] = a[i] + b[i]; }\n";
  const std::string kernel2 =
      "extern \"C\" __global__ void sub_kernel(float* out, float*a, float*b) { size_t i = "
      "threadIdx.x; out[i] = a[i] - b[i]; }\n";
  const std::string kernel3 =
      "extern \"C\" __global__ void mul_kernel(float* out, float*a, float*b) { size_t i = "
      "threadIdx.x; out[i] = a[i] * b[i]; }\n";

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string gpu_arch = prop.gcnArchName;

  std::vector<float> input1, input2;
  input1.reserve(size);
  input2.reserve(size);
  for (size_t i = 0; i < size; i++) {
    input1[i] = (i + 1) * 2;
    input2[i] = i;
  }

  float *d_in1, *d_in2, *d_out;
  HIP_CHECK(hipMalloc(&d_in1, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_in2, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));

  HIP_CHECK(hipMemset(d_out, 0, sizeof(float) * size));
  HIP_CHECK(hipMemcpy(d_in1, input1.data(), sizeof(float) * size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_in2, input2.data(), sizeof(float) * size, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("One Kernel") {
    auto kernel = kernel1;
    auto code = compile_using_hiprtc(kernel, gpu_arch);

    hipLibrary_t library;
    hipKernel_t function;

    HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    HIP_CHECK(hipLibraryGetKernel(&function, library, "add_kernel"));

    unsigned int count = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&count, library));
    REQUIRE(count == 1);

    void* args[] = {&d_out, &d_in1, &d_in2};

    HIP_CHECK(hipLaunchKernel(function, 1, size, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLibraryUnload(library));


    std::vector<float> out(size, 0);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float tmp = input1[i] + input2[i];
      INFO("Index: " << i << " cpu res: " << tmp << " gpu res: " << out[i]);
      REQUIRE(out[i] == tmp);
    }
  }

  SECTION("Two Kernel") {
    auto kernel = kernel1 + kernel2;
    auto code = compile_using_hiprtc(kernel, gpu_arch);

    hipLibrary_t library;
    hipKernel_t function;

    HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    HIP_CHECK(hipLibraryGetKernel(&function, library, "sub_kernel"));

    unsigned int count = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&count, library));
    REQUIRE(count == 2);

    void* args[] = {&d_out, &d_in1, &d_in2};

    HIP_CHECK(hipLaunchKernel(function, 1, size, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLibraryUnload(library));


    std::vector<float> out(size, 0);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float tmp = input1[i] - input2[i];
      INFO("Index: " << i << " cpu res: " << tmp << " gpu res: " << out[i]);
      REQUIRE(out[i] == tmp);
    }
  }

  SECTION("Three Kernel") {
    auto kernel = kernel1 + kernel2 + kernel3;
    auto code = compile_using_hiprtc(kernel, gpu_arch);

    hipLibrary_t library;
    hipKernel_t function;

    HIP_CHECK(hipLibraryLoadData(&library, code.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    HIP_CHECK(hipLibraryGetKernel(&function, library, "mul_kernel"));

    unsigned int count = 0;
    HIP_CHECK(hipLibraryGetKernelCount(&count, library));
    REQUIRE(count == 3);

    void* args[] = {&d_out, &d_in1, &d_in2};

    HIP_CHECK(hipLaunchKernel(function, 1, size, args, 0, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipLibraryUnload(library));


    std::vector<float> out(size, 0);
    HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(float) * size, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
      float tmp = input1[i] * input2[i];
      INFO("Index: " << i << " cpu res: " << tmp << " gpu res: " << out[i]);
      REQUIRE(out[i] == tmp);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_in1));
  HIP_CHECK(hipFree(d_in2));
  HIP_CHECK(hipFree(d_out));
}