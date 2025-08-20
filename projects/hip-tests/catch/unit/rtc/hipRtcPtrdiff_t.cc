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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_process.hh>
#include <hip/hiprtc.h>
/**
 * @addtogroup ptrdiff_t ptrdiff_t
 * @{
 * @ingroup hiprtcHeadersTest
 * `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
 *                                  int numOptions,
 *                                  const char** options);` -
 * This test case verifies the data type of ptrdiff_t via hipRTC API's
 */
static constexpr auto ptrdiff_Kernel_String{
    R"(
extern "C"
__global__ void ptrdiff_Kernel(unsigned int *res, int platformVar) {
  #if __HIPRTC_PTRDIFF_T_IS_LONG_LONG__
    *res = __hip_internal::is_same<ptrdiff_t, long long>::value;
  #else
    if(platformVar == 1)
      *res = __hip_internal::is_same<ptrdiff_t, long>::value;
    else
      *res = __hip_internal::is_same<ptrdiff_t, long long>::value;
  #endif
}
)"};
/**
 * Test Description
 * ------------------------
 *  - Functional Test to verify the data type of ptrdiff_t
 *  - If the macro __HIPRTC_PTRDIFF_T_IS_LONG_LONG__ is
 *  - defined and equal to 1 then the data type of ptrdiff_t
 *  - must be long long on both platforms linux and windows.
 * Test source
 * ------------------------
 *  - unit/rtc/hipRtcPtrdiff_t.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipRTC_Ptrdiff_t_Check") {
  std::string kernel_name = "ptrdiff_Kernel";
  const char* kername = kernel_name.c_str();
  unsigned int* result_h;
  unsigned int* result_d;
  unsigned int Nbytes = sizeof(unsigned int);
  result_h = new unsigned int;
  *result_h = 0;
  HIP_CHECK(hipMalloc(&result_d, Nbytes));
  HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string architecture = prop.gcnArchName;
  std::string complete_CO = "--gpu-architecture=" + architecture;
  hiprtcProgram prog;
  const char** compiler_options = new const char*[2];
  compiler_options[0] = complete_CO.c_str();
  int platformVar = 0;
  SECTION("Test for Macro -D__HIPRTC_PTRDIFF_T_IS_LONG_LONG__ = 0") {
    compiler_options[1] = "-D__HIPRTC_PTRDIFF_T_IS_LONG_LONG__=0";
#ifdef __linux__
    platformVar = 1;
#endif
  }
  SECTION("Test for Macro -D__HIPRTC_PTRDIFF_T_IS_LONG_LONG__ = 1") {
    compiler_options[1] = "-D__HIPRTC_PTRDIFF_T_IS_LONG_LONG__=1";
  }

  HIPRTC_CHECK(hiprtcCreateProgram(&prog, ptrdiff_Kernel_String, kername, 0, NULL, NULL));
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, compiler_options)};
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
  void* kernelParam[] = {result_d, reinterpret_cast<void*>(platformVar)};
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
  if (*result_h != 1) {
    REQUIRE(false);
  }
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  HIP_CHECK(hipFree(result_d));
  delete result_h;
  delete[] compiler_options;
}
/**
 * Test Description
 * ------------------------
 *  - Functional Test to verify the data type of ptrdiff_t from the child process.
 *  - If the macro __HIPRTC_PTRDIFF_T_IS_LONG_LONG__ is
 *  - defined and equal to 1 then the data type of ptrdiff_t
 *  - must be long long on both platforms linux and windows.
 * Test source
 * ------------------------
 *  - unit/rtc/hipRtcPtrdiff_t.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipRTC_Check_Ptrdiff_t_FrmChildProcess") {
  // Spawn a process
  hip::SpawnProc proc("ChkPtrdiff_t_Exe", true);
  if ((proc.run("HIPRTC_PTRDIFF_T_IS_LONG_LONG 1") == 1) &&
      (proc.run("HIPRTC_PTRDIFF_T_IS_LONG_LONG 0") == 1)) {
    WARN("Test pass\n");
    REQUIRE(true);
  } else {
    WARN("Test Fail\n");
    REQUIRE(false);
  }
}
