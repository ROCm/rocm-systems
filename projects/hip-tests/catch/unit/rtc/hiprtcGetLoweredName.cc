/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <vector>
#include <string>


/**
 * @addtogroup hiprtcGetLoweredName hiprtcGetLoweredName
 * @{
 * @ingroup hiprtc
 * `hiprtcResult hiprtcGetLoweredName(prog,
 *               kernel_name_vec[i].c_str(), // name expression
 *               &name                       // lowered name
 *               ));` -
 * These test cases tests working hiprtcGetLoweredName() api
 */

static const char* const gpuProgram1 = R"(
template <int N, typename T>
__global__ void my_kernel(T* data) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  T data0 = data[0];
  for( int i=0; i<N-1; ++i ) {
    data[0] *= data0;
  }
})";

static const char* const gpuProgram2 = R"(
template <int N, typename T, typename K>
__global__ void my_kernel(T* data, K* Arr) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  T data0 = data[0];
  for( int i=0; i<N-1; ++i ) {
    data[0] = data0 + Arr[i];
  }
})";

static const char* const gpuProgram3 = R"(
template <int N>
__global__ void nontype_kernel() {}
)";

bool Test(int CaseNum, const char* GpuProgram) {
  bool IfTestPassed = true;
  // Create an instance of hiprtcProgram
  hiprtcProgram prog;

  HIPRTC_CHECK(hiprtcCreateProgram(&prog,       // prog
                                   GpuProgram,  // buffer
                                   "prog.cu",   // name
                                   0,           // numHeaders
                                   NULL,        // headers
                                   NULL));      // includeNames

  // add all name expressions for kernels
  std::vector<std::string> kernel_name_vec, kernelNameExpectdOutput;
  if (CaseNum == 1) {
    kernel_name_vec.push_back("my_kernel<(int)3, float >");
    kernel_name_vec.push_back("my_kernel<(int)5, double >");
    kernel_name_vec.push_back("my_kernel<(int)6, int >");
    kernel_name_vec.push_back("my_kernel<(int)10, long >");
    kernel_name_vec.push_back("my_kernel<(int)11, long long >");
    kernel_name_vec.push_back("my_kernel<(int)123, unsigned int >");
    kernel_name_vec.push_back("my_kernel<(int)1234, char >");
    kernel_name_vec.push_back("my_kernel<(int)12345, unsigned char >");

    kernelNameExpectdOutput.push_back("_Z9my_kernelILi3EfEvPT0_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi5EdEvPT0_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi6EiEvPT0_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi10ElEvPT0_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi11ExEvPT0_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi123EjEvPT0_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi1234EcEvPT0_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi12345EhEvPT0_");
  } else if (CaseNum == 2) {
    kernel_name_vec.push_back("my_kernel<(int)3, float, float >");
    kernel_name_vec.push_back("my_kernel<(int)66, float, int >");
    kernel_name_vec.push_back("my_kernel<(int)(4 + 6), long, double >");
    kernel_name_vec.push_back("my_kernel<(int)5, double, double >");
    kernel_name_vec.push_back("my_kernel<(int)6, int, int >");
    kernel_name_vec.push_back("my_kernel<(int)10, long, long >");
    kernel_name_vec.push_back("my_kernel<(int)11, long long, long long >");
    kernel_name_vec.push_back("my_kernel<(int)123, unsigned int, unsigned int >");  // NOLINT
    kernel_name_vec.push_back("my_kernel<(int)1234, char, char >");
    kernel_name_vec.push_back("my_kernel<(int)12345, unsigned char, unsigned char >");  // NOLINT

    kernelNameExpectdOutput.push_back("_Z9my_kernelILi3EffEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi66EfiEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi10EldEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi5EddEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi6EiiEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi10EllEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi11ExxEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi123EjjEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi1234EccEvPT0_PT1_");
    kernelNameExpectdOutput.push_back("_Z9my_kernelILi12345EhhEvPT0_PT1_");
  } else if (CaseNum == 3) {
    kernel_name_vec.push_back("nontype_kernel<3>");
    kernel_name_vec.push_back("nontype_kernel<(2 + 3)>");
    kernel_name_vec.push_back("nontype_kernel<(-2 + 6)>");

    kernelNameExpectdOutput.push_back("_Z14nontype_kernelILi3EEvv");
    kernelNameExpectdOutput.push_back("_Z14nontype_kernelILi5EEvv");
    kernelNameExpectdOutput.push_back("_Z14nontype_kernelILi4EEvv");
  }

  // add kernel name expressions to HIPRTC
  for (size_t i = 0; i < kernel_name_vec.size(); ++i)
    HIPRTC_CHECK(hiprtcAddNameExpression(prog, kernel_name_vec[i].c_str()));

  hiprtcResult compileResult = hiprtcCompileProgram(prog,   // prog
                                                    0,      // numOptions
                                                    NULL);  // options
  // Obtain compilation log from the program.
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));

  if (logSize > 0) {
    char* log = new char[logSize];
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, log));
    INFO(log);
    delete[] log;
    REQUIRE(compileResult == HIPRTC_SUCCESS);
  }
  // Extract lowered names
  for (size_t i = 0; i < kernel_name_vec.size(); ++i) {
    const char* name;
    HIPRTC_CHECK(hiprtcGetLoweredName(prog,
                                      kernel_name_vec[i].c_str(),  // name expression
                                      &name));                     // lowered name
    if (name != kernelNameExpectdOutput[i]) {
      IfTestPassed = false;
    }
  }
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return IfTestPassed;
}


TEST_CASE(Unit_hiprtcGetLoweredName_templateKrnls) {
  REQUIRE(Test(1, gpuProgram1));
  REQUIRE(Test(2, gpuProgram2));
  REQUIRE(Test(3, gpuProgram3));
}

/**
 * End doxygen group hiprtc.
 * @}
 */
