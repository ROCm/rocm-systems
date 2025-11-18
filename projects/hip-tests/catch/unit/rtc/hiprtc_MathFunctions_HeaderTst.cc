/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hiprtc_MathFunctions_HeaderTst hiprtc_MathFunctions_HeaderTst
 * @{
 * @ingroup hiprtcHeaders
 * `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
 *                                    int numOptions,
 *                                    const char** options);` -
 * These test cases are target including various header file in kernel
 * string and compile using the api mentioned above.
 */

#include <hip_test_common.hh>
#include "hip_test_rtc.hh"
#include <hip/hiprtc.h>
#include <hip/math_functions.h>

static constexpr auto mathFuntn_string{
    R"(
extern "C"
__global__ void mathFuntn(float *res) {
  res[0] = (amd_mixed_dot(short2{10, 10}, short2{20, 20}, (int)30, 1) == 430);
  res[1] = (amd_mixed_dot(ushort2{10, 10}, ushort2{20, 20}, (uint)30, 1) == 430);
  res[2] = (amd_mixed_dot((char)10, (char)20, (int)30, 1) == 6);
  res[3] = (amd_mixed_dot(uchar4{'a', 'a', 'a', 'a'}, uchar4{'b', 'b', 'b', 'b'}, (uint)30, 1) == 38054);
  res[4] = (amd_mixed_dot(10, 20, 30, 1) == 6);
  res[5] = (amd_mixed_dot((uint)10, (uint)20, (uint)30, 1) == 70);
}
)"};

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hiprtcCompileProgram
 *    1) To test working of "hip/math_functions.h" header inside kernel string
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtc_MathFunctions_HeaderTst.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_Rtc_MathFunctions_header") {
  std::string kernel_name = "mathFuntn";
  const char* kername = kernel_name.c_str();
  float* result_h;
  float* result_d;
  int n = 6;
  int Nbytes = n * sizeof(float);
  result_h = new float[n];
  for (int i = 0; i < n; i++) {
    result_h[i] = 0;
  }
  HIP_CHECK(hipMalloc(&result_d, Nbytes));
  HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string architecture = prop.gcnArchName;
  std::string complete_CO = "--gpu-architecture=" + architecture;
  const char* compiler_option = complete_CO.c_str();

  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, mathFuntn_string, kername, 0, NULL, NULL));
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
  if (!(compileResult == HIPRTC_SUCCESS)) {
    WARN("hiprtcCompileProgram() api failed!!");
    size_t logSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    WARN(log);
    REQUIRE(false);
  }

  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  std::vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  void* kernelParam[] = {result_d};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kername));
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(result_h, result_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    if (result_h[i] != 1.0f) {
      WARN("FAIL for " << i << " iteration");
      WARN(result_h[i]);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  HIP_CHECK(hipFree(result_d));
  delete[] result_h;
  REQUIRE(true);
}

/**
 * End doxygen group hiprtcHeaders.
 * @}
 */
