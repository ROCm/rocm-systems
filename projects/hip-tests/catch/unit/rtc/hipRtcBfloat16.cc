/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// This test verifies the accuracy of hip_bfloat16 and its usage with hiprtc

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>
#include <cmath>

const char* kernelname = "test_hip_bfloat16";

static constexpr auto code{
    R"(
extern "C"
__global__
void test_hip_bfloat16(float* f, bool* result)
{
    float &f_a = *f;
    hip_bfloat16 bf_a(f_a);
    float f_c = float(bf_a);
    // float relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    if (fabs(f_c - f_a) / f_a <= 1.0 / 128) {
     *result = true;
    } else {
     *result = false;
    }
}
)"};

TEST_CASE(Unit_hiprtc_test_hip_bfloat16) {
  using namespace std;
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, code, "code.cu", 0, nullptr, nullptr));
  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
#else
  std::string sarg = std::string("--gpu-architecture=compute_") + std::to_string(props.major) +
                     std::to_string(props.minor);
#endif
  vector<const char*> opts;
  opts.push_back(sarg.c_str());
  hiprtcResult compileResult{hiprtcCompileProgram(prog, opts.size(), opts.data())};
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
  vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  float h_a = 10.0f;
  float* f_a;
  bool* d_result;
  HIP_CHECK(hipMalloc(&f_a, sizeof(float)));
  HIP_CHECK(hipMalloc(&d_result, sizeof(bool)));
  HIP_CHECK(hipMemcpy(f_a, &h_a, sizeof(float), hipMemcpyHostToDevice));
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kernelname));
  struct {
    float* a_;
    bool* b_;
  } args{f_a, d_result};
  auto sizeofargs = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &sizeofargs, HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());
  bool h_result;
  HIP_CHECK(hipMemcpyDtoH(&h_result, reinterpret_cast<hipDeviceptr_t>(d_result), sizeof(bool)));
  HIP_CHECK(hipFree(d_result));
  HIP_CHECK(hipFree(f_a));
  HIP_CHECK(hipModuleUnload(module));
  // Result returned is true if the hip_bfloat16 accuracy is as expected
  REQUIRE(h_result == true);
}
