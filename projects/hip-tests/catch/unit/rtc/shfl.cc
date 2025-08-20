/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>
#include "shfl.hh"

static constexpr auto shfl{
    R"(
template <typename T>
__global__ void shflUpSum(T* a, int size) {
  T val = a[threadIdx.x];
  for (unsigned int i = size / 2; i > 0; i /= 2) {
    val += __shfl_up(val, i, size);
  }
  a[threadIdx.x] = val;
}

template <typename T>
__global__ void shflDownSum(T* a, int size) {
  T val = a[threadIdx.x];
  for (int i = size / 2; i > 0; i /= 2) {
    val += __shfl_down(val, i, size);
  }
  a[threadIdx.x] = val;
}

template <typename T>
__global__ void shflXorSum(T* a, int size) {
  T val = a[threadIdx.x];
  for (int i = size/2; i > 0; i /= 2)
    val += __shfl_xor(val, i, size);
  a[threadIdx.x] = val;
}
)"};

template <typename T> void runTestShfl(int option) {
  using namespace std;
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog,      // prog
                      shfl,       // buffer
                      "shfl.cu",  // name
                      0, nullptr, nullptr);

  string str;
  switch (option) {
    case 1:
      str = "shflUpSum<__half>";
      break;
    case 2:
      str = "shflDownSum<__half>";
      break;
    case 3:
      str = "shflXorSum<__half>";
      break;
    default:
      INFO("Options 1,2,3 are supported, but the passed option is: " << option);
      REQUIRE(false);
  }

  hiprtcAddNameExpression(prog, str.c_str());

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 0, 0)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

  // Do hip malloc first so that we donot need to do a cuInit manually before calling hipModule APIs
  size_t bufferSize = n * sizeof(T);

  T a[n];
  T cpuSum = sum(a);
  T* d_a;
  HIP_CHECK(hipMalloc(&d_a, bufferSize));

  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  const char* name;
  hiprtcGetLoweredName(prog, str.c_str(), &name);
  HIP_CHECK(hipModuleGetFunction(&kernel, module, name));

  HIP_CHECK(hipMemcpy(d_a, &a, bufferSize, hipMemcpyDefault));

  struct {
    T* a_;
    int b_;
  } args{d_a, n};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, n, 1, 1, 0, nullptr, nullptr, config));

  HIP_CHECK(hipMemcpy(&a, d_a, bufferSize, hipMemcpyDefault));
  bool result;
  switch (option) {
    case 1:  // shflUpSum
      result = compare(a[n - 1], cpuSum);
      break;
    case 2:  // shflDownSum
    case 3:  // shflXorSum
      result = compare(a[0], cpuSum);
      break;
  }

  if (result) {
    HIP_CHECK(hipFree(d_a));
    REQUIRE(false);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}

TEST_CASE("Unit_hiprtc_half_shuffle") {
  runTestShfl<__half>(1);
  runTestShfl<__half>(2);
  runTestShfl<__half>(3);
}
