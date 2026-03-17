/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <cassert>
#include <vector>
#define HIP_CHECK(error)                                                                           \
  {                                                                                                \
    hipError_t localError = error;                                                                 \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {          \
      printf("error: '%s'(%d) from %s at %s:%d\n", hipGetErrorString(localError), localError,      \
             #error, __FUNCTION__, __LINE__);                                                      \
      exit(0);                                                                                     \
    }                                                                                              \
  }
#define HIPRTC_CHECK(error)                                                                        \
  {                                                                                                \
    auto localError = error;                                                                       \
    if (localError != HIPRTC_SUCCESS) {                                                            \
      printf("error: '%s'(%d) from %s at %s:%d\n", hiprtcGetErrorString(localError), localError,   \
             #error, __FUNCTION__, __LINE__);                                                      \
      exit(0);                                                                                     \
    }                                                                                              \
  }
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

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Invalid number of args passed.\n"
              << "argc : " << argc << std::endl;
    return -1;
  }
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
  const char* compiler_option = complete_CO.c_str();
  hiprtcProgram prog;
  const char** compiler_options = new const char*[2];
  compiler_options[0] = compiler_option;
  std::string macro = argv[1];
  int macroVal = std::stoi(argv[2]);
  int platformVar = 0;
  if ((macro == "HIPRTC_PTRDIFF_T_IS_LONG_LONG") && (macroVal == 1)) {
    compiler_options[1] = "-D__HIPRTC_PTRDIFF_T_IS_LONG_LONG__=1";
  } else if ((macro == "HIPRTC_PTRDIFF_T_IS_LONG_LONG") && (macroVal == 0)) {
    compiler_options[1] = "-D__HIPRTC_PTRDIFF_T_IS_LONG_LONG__=0";
#ifdef __linux__
    platformVar = 1;
#endif
  }
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, ptrdiff_Kernel_String, kername, 0, NULL, NULL));
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, compiler_options)};

  if (!(compileResult == HIPRTC_SUCCESS)) {
    printf("hiprtcCompileProgram() api failed!!");
    size_t logSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    assert(false);
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
    return 0;
  }
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  HIP_CHECK(hipFree(result_d));
  delete[] compiler_options;
  delete result_h;
  return 1;
}
