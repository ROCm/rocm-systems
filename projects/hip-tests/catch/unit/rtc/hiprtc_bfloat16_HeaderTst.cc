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
 * @addtogroup hiprtc_bfloat16_HeaderTst hiprtc_bfloat16_HeaderTst
 * @{
 * @ingroup hiprtcHeaders
 * `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
 *                                  int numOptions,
 *                                  const char** options);` -
 * These test cases are target including various header file in kernel
 * string and compile using the api mentioned above.
 */

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
static constexpr auto bfloat16_string{
    R"(
extern "C"
__global__ void bfloat16(float *res) {
  res[0] = (hip_bfloat16::round_to_bfloat16((float) 10) == 10)? 1 : 0;
  res[1] = (operator+((hip_bfloat16) 10) == 10)? 1 : 0;
  res[2] = (operator-((hip_bfloat16) 10) == -10)? 1 : 0;
  res[3] = (operator+((hip_bfloat16) 10, (hip_bfloat16) 1) == 11)? 1 : 0;
  res[4] = (operator-((hip_bfloat16) 10, (hip_bfloat16) 1) == 9)? 1 : 0;
  res[5] = (operator*((hip_bfloat16) 10, (hip_bfloat16) 1) == 10)? 1 : 0;
  res[6] = (operator/((hip_bfloat16) 10, (hip_bfloat16) 1) == 10)? 1 : 0;
  res[7] = (operator<((hip_bfloat16) 1, (hip_bfloat16) 10) == 1)? 1 : 0;
  res[8] = (operator==((hip_bfloat16) 10, (hip_bfloat16) 10) == 1)? 1 : 0;
  res[9] = (operator>((hip_bfloat16) 10, (hip_bfloat16) 1) == 1)? 1 : 0;
  res[10] = (operator<=((hip_bfloat16) 10, (hip_bfloat16) 10) == 1)? 1 : 0;
  res[11] = (operator!=((hip_bfloat16) 1, (hip_bfloat16) 10) == 1)? 1 : 0;
  res[12] = (operator>=((hip_bfloat16) 10, (hip_bfloat16) 10) == 1)? 1 : 0;
  hip_bfloat16 a = (hip_bfloat16)10;
  res[13] = (operator+=(a, (hip_bfloat16) 10) == 20)? 1 : 0;
  res[14] = (operator-=(a, (hip_bfloat16) 10) == 10)? 1 : 0;
  res[15] = (operator*=(a, (hip_bfloat16) 10) == 100)? 1 : 0;
  res[16] = (operator/=(a, (hip_bfloat16) 10) == 10)? 1 : 0;
  res[17] = (operator++(a) == 11)? 1 : 0;
  res[18] = (operator--(a) == 10)? 1 : 0;
  res[19] = (operator++(a, 0) == 10)? 1 : 0;
  res[20] = (hip_bfloat16::round_to_bfloat16((float) 10, hip_bfloat16::truncate) == 10)? 1 : 0;
  hip_bfloat16 x(11.3);
  res[21] = (x.data == 16693)? 1 : 0;
  hip_bfloat16 y(11.3, hip_bfloat16::truncate);
  res[22] = (y.data == 16692)? 1 : 0;
  x.operator=(11.6);
  res[23] = (x.data == 16698)? 1 : 0;
  hip_bfloat16 z(11.75);
  res[24] = ((z.operator float ()) == 11.75)? 1 : 0;
  res[25] = ((std::iszero((hip_bfloat16)(0*0))) == 1)? 1 : 0;
}
)"};

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hiprtcCompileProgram
 *    1) To test working of "hip/hip_bfloat16.h"  header inside kernel string
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtc_bfloat16_HeaderTst.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_Rtc_bfloat16_header") {
  std::string kernel_name = "bfloat16";
  const char* kername = kernel_name.c_str();
  float* result_h;
  float* result_d;
  int n = 26;
  float Nbytes = n * sizeof(float);
  result_h = new float[n];
  for (int i = 0; i < n; i++) {
    result_h[i] = 0.0f;
  }
  HIP_CHECK(hipMalloc(&result_d, Nbytes));
  HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string architecture = prop.gcnArchName;
  std::string complete_CO = "--gpu-architecture=" + architecture;
  const char* compiler_option = complete_CO.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, bfloat16_string, kername, 0, NULL, NULL));
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
  HIP_CHECK(hipFree(result_d));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  delete[] result_h;
}
