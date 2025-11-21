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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
This file contains functions for idividual HIPRTC supported compiler options
validation. For PASS senario the function returns 1 or 0 otherwise.
*/

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_fp16.h>
#include <picojson.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include "headers/RtcUtility.h"
#include "headers/RtcFunctions.h"
#include "headers/RtcKernels.h"
#include <hip_test_common.hh>
#include "headers/printf_common.h"

#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"

bool check_architecture(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                        int fast_math_present) {
  std::string block_name = "architecture";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string actual_architecture = prop.gcnArchName;
  std::string complete_CO = retrieved_CO + actual_architecture;
  const char* compiler_option = complete_CO.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, max_thread_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  return 1;
}

bool check_rdc(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
               int fast_math_present) {
  std::string block_name = "rdc";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string CO = get_string_parameters("compiler_option", block_name);
  if (CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  const char* compiler_opt = CO.c_str();
  float *A_d, *B_d, *C_d;
  float *A_h, *B_h, *C_h, *result;
  float Nbytes = sizeof(float);
  A_h = new float[1];
  B_h = new float[1];
  C_h = new float[1];
  result = new float[1];
  for (int i = 0; i < 1; i++) {
    A_h[i] = 4;
    B_h[i] = 4;
    result[i] = 16;
  }
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, rdc_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_opt);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_opt)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_opt);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  void* kernelParam[] = {A_d, B_d, C_d};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetBitcodeSize(prog, &codeSize));
  std::vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetBitcode(prog, codec.data()));
  float wall_time;
  int reg_count = 2;
  int max_thread = 1;
  unsigned int log_size = 5120;
  char error_log[5120];
  char info_log[5120];
  std::vector<hiprtcJIT_option> jit_options = {HIPRTC_JIT_MAX_REGISTERS,
                                               HIPRTC_JIT_THREADS_PER_BLOCK,
                                               HIPRTC_JIT_WALL_TIME,
                                               HIPRTC_JIT_INFO_LOG_BUFFER,
                                               HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                               HIPRTC_JIT_ERROR_LOG_BUFFER,
                                               HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                               HIPRTC_JIT_LOG_VERBOSE};
  const void* lopts[] = {reinterpret_cast<void*>(&reg_count), reinterpret_cast<void*>(&max_thread),
                         reinterpret_cast<void*>(&wall_time), info_log,
                         reinterpret_cast<void*>(log_size),   error_log,
                         reinterpret_cast<void*>(log_size),   reinterpret_cast<void*>(1)};
  hiprtcLinkState rtc_link_state;
  void* binary;
  size_t binarySize;
  int pass_count = 0;
  hipModule_t module;
  hipFunction_t function;
  for (int i = 0; i < 2; i++) {
    switch (i) {
      case 0:
        HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &rtc_link_state));
        HIPRTC_CHECK(hiprtcLinkAddData(rtc_link_state, HIPRTC_JIT_INPUT_LLVM_BITCODE, codec.data(),
                                       codeSize, 0, 0, 0, 0));
        HIPRTC_CHECK(hiprtcLinkComplete(rtc_link_state, &binary, &binarySize));
        HIP_CHECK(hipModuleLoadData(&module, binary));
        HIP_CHECK(hipModuleGetFunction(&function, module, kername));
        HIP_CHECK(
            hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
        pass_count++;
        break;
      case 1:
        HIPRTC_CHECK(hiprtcLinkCreate(8, jit_options.data(), reinterpret_cast<void**>(&lopts),
                                      &rtc_link_state));
        HIPRTC_CHECK(hiprtcLinkAddData(rtc_link_state, HIPRTC_JIT_INPUT_LLVM_BITCODE, codec.data(),
                                       codeSize, 0, 0, 0, 0));
        HIPRTC_CHECK(hiprtcLinkComplete(rtc_link_state, &binary, &binarySize));
        HIP_CHECK(hipModuleLoadData(&module, binary));
        HIP_CHECK(hipModuleGetFunction(&function, module, kername));
        HIP_CHECK(
            hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
        pass_count++;
        break;
      default:
        WARN(" NOT VALID INPUT ");
        break;
    }
  }
  HIP_CHECK(hipMemcpy(result, C_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < 1; i++) {
    if (result[i] != ((A_h[i] * B_h[i]))) {
      WARN("Compiler Option : " << compiler_opt);
      WARN("EXPECTED RESULT DOES NOT MATCH ");
      WARN("INPUT A & B : " << A_h[i] << " , " << B_h[i]);
      WARN("EXPECTED RES : " << (A_h[i] * B_h[i]));
      WARN("OBTAINED RES : " << result[i]);
      return 0;
    }
  }
  if (pass_count == 2) {
    return 1;
  } else {
    WARN(" pass_count IS NOT MATCHING ");
    return 0;
  }
}

bool check_denormals_enabled(const char** Combination_CO, int Combination_CO_size,
                             int max_thread_pos, int fast_math_present) {
  std::string block_name = "denormals";
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  picojson::array Input_Vals = get_array_parameters("Input_Vals", block_name);
  picojson::array Expected_Results = get_array_parameters("Expected_Results", block_name);
  const char* kername = kernel_name.c_str();
  const char* compiler_option = retrieved_CO.c_str();
  std::vector<double> double_vec_input;
  for (auto& indx : Input_Vals) {
    double_vec_input.push_back(indx.get<double>());
  }
  std::vector<int> Input_Vals_int;
  for (auto& indx : double_vec_input) {
    Input_Vals_int.push_back(static_cast<int>(indx));
  }
  std::vector<double> double_vec_expected;
  for (auto& indx : Expected_Results) {
    double_vec_expected.push_back(indx.get<double>());
  }
  std::vector<int> Expected_Results_int;
  for (auto& indx : double_vec_expected) {
    Expected_Results_int.push_back(static_cast<int>(indx));
  }
  int test_case, res_inc;
  for (test_case = 0, res_inc = 0;
       test_case < Input_Vals_int.size() && res_inc < Expected_Results_int.size();
       test_case += 2, res_inc++) {
    double *base_h, *power_h, *result_h;
    double *base_d, *power_d, *result_d;
    double Nbytes = sizeof(double);
    base_h = new double[1];
    power_h = new double[1];
    result_h = new double[1];
    *base_h = Input_Vals_int[test_case];
    *power_h = Input_Vals_int[test_case + 1];
    *result_h = 1;
    HIP_CHECK(hipMalloc(&base_d, Nbytes));
    HIP_CHECK(hipMalloc(&power_d, Nbytes));
    HIP_CHECK(hipMalloc(&result_d, Nbytes));
    HIP_CHECK(hipMemcpy(base_d, base_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(power_d, power_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));
    hiprtcProgram program;
    HIPRTC_CHECK(hiprtcCreateProgram(&program, denormals_string, "denormals", 0, NULL, NULL));
    if (Combination_CO_size != -1) {
      hiprtcResult compileResult{
          hiprtcCompileProgram(program, Combination_CO_size, Combination_CO)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << compiler_option);
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(program, &log[0]));
          WARN(log);
        }
        return 0;
      }
    } else {
      hiprtcResult compileResult{hiprtcCompileProgram(program, 1, &compiler_option)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << compiler_option);
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(program, &log[0]));
          WARN(log);
        }
        return 0;
      }
    }
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(program, &codeSize));
    std::vector<char> codec(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(program, codec.data()));
    void* kernelParam[] = {base_d, power_d, result_d};
    auto size = sizeof(kernelParam);
    void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                                HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    hipModule_t module;
    hipFunction_t function;
    HIP_CHECK(hipModuleLoadData(&module, codec.data()));
    HIP_CHECK(hipModuleGetFunction(&function, module, kername));
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
    HIP_CHECK(hipMemcpy(result_h, result_d, sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipModuleUnload(module));
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    if (*result_h != Expected_Results_int[res_inc]) {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("EXPECTED RESULT DOES NOT MATCH FOR " << res_inc);
      WARN("th ITERATION (start iteration is 0 ) ");
      WARN("INPUT : pow(2, " << *power_h << ") ");
      WARN("EXPECTED OP: " << Expected_Results_int[res_inc]);
      WARN("OBTAINED OP: " << *result_h);
      return 0;
    }
  }
  return 1;
}

bool check_denormals_disabled(const char** Combination_CO, int Combination_CO_size,
                              int max_thread_pos, int fast_math_present) {
  std::string block_name = "denormals";
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  picojson::array Input_Vals = get_array_parameters("Input_Vals", block_name);
  picojson::array Expected_Results_for_no =
      get_array_parameters("Expected_Results_for_no", block_name);
  const char* kername = kernel_name.c_str();
  const char* compiler_option = retrieved_CO.c_str();
  std::vector<double> double_vec_input;
  for (auto& indx : Input_Vals) {
    double_vec_input.push_back(indx.get<double>());
  }
  std::vector<int> Input_Vals_int;
  for (auto& indx : double_vec_input) {
    Input_Vals_int.push_back(static_cast<int>(indx));
  }
  std::vector<double> double_vec_expected_for_no;
  for (auto& indx : Expected_Results_for_no) {
    double_vec_expected_for_no.push_back(indx.get<double>());
  }
  std::vector<int> Expected_Results_for_no_int;
  for (auto& indx : double_vec_expected_for_no) {
    Expected_Results_for_no_int.push_back(static_cast<int>(indx));
  }
  int test_case, res_inc;
  for (test_case = 0, res_inc = 0;
       test_case < Input_Vals_int.size() && res_inc < Expected_Results_for_no_int.size();
       test_case += 2, res_inc++) {
    double *base_h, *power_h, *result_h;
    double *base_d, *power_d, *result_d;
    double Nbytes = sizeof(double);
    base_h = new double[1];
    power_h = new double[1];
    result_h = new double[1];
    *base_h = Input_Vals_int[test_case];
    *power_h = Input_Vals_int[test_case + 1];
    *result_h = 0;
    HIP_CHECK(hipMalloc(&base_d, Nbytes));
    HIP_CHECK(hipMalloc(&power_d, Nbytes));
    HIP_CHECK(hipMalloc(&result_d, Nbytes));
    HIP_CHECK(hipMemcpy(base_d, base_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(power_d, power_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));
    hiprtcProgram program;
    HIPRTC_CHECK(hiprtcCreateProgram(&program, denormals_string, "denormals", 0, NULL, NULL));
    if (Combination_CO_size != -1) {
      hiprtcResult compileResult{
          hiprtcCompileProgram(program, Combination_CO_size, Combination_CO)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << compiler_option);
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(program, &log[0]));
          WARN(log);
        }
        return 0;
      }
    } else {
      hiprtcResult compileResult{hiprtcCompileProgram(program, 1, &compiler_option)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << compiler_option);
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(program, &log[0]));
          WARN(log);
        }
        return 0;
      }
    }
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(program, &codeSize));
    std::vector<char> codec(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(program, codec.data()));
    void* kernelParam[] = {base_d, power_d, result_d};
    auto size = sizeof(kernelParam);
    void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                                HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    hipModule_t module;
    hipFunction_t function;
    HIP_CHECK(hipModuleLoadData(&module, codec.data()));
    HIP_CHECK(hipModuleGetFunction(&function, module, kername));
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
    HIP_CHECK(hipMemcpy(result_h, result_d, sizeof(double), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipModuleUnload(module));
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
    if (*result_h != Expected_Results_for_no_int[res_inc]) {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("EXPECTED RESULT DOES NOT MATCH FOR " << res_inc);
      WARN("th ITERATION (start iteration is 0 ) ");
      WARN("INPUT : pow(2, " << *power_h << ") ");
      WARN("EXPECTED OP: " << Expected_Results_for_no_int[res_inc]);
      WARN("OBTAINED OP: " << *result_h);
      return 0;
    }
  }
  return 1;
}

bool check_ffp_contract_off(const char** Combination_CO, int Combination_CO_size,
                            int max_thread_pos, int fast_math_present) {
  std::string block_name = "ffp_contract";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  picojson::array retrieved_CO = get_array_parameters("compiler_option", block_name);
  if (retrieved_CO.size() < 3) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::vector<std::string> CO_vec;
  for (auto& indx : retrieved_CO) {
    CO_vec.push_back(indx.get<std::string>());
  }
  int CO_IRadded_size = 3;
  const char** CO_IRadded = new const char*[3];
  std::string hold = CO_vec[0];
  CO_IRadded[0] = hold.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO[0]);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (data.find("fmul contract") != -1 && data.find("@llvm.fmuladd.f32") != -1) {
    WARN("Compiler option : " << retrieved_CO[0]);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR CONTAIN EITHER");
    WARN("'fmul contract' or '@llvm.fmuladd.f32' or both ");
    WARN("WHICH IS NOT EXPECTED");
    return 0;
  } else {
    return 1;
  }
}

bool check_ffp_contract_on(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                           int fast_math_present) {
  std::string block_name = "ffp_contract";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  picojson::array retrieved_CO = get_array_parameters("compiler_option", block_name);
  if (retrieved_CO.size() < 3) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::vector<std::string> CO_vec;
  for (auto& indx : retrieved_CO) {
    CO_vec.push_back(indx.get<std::string>());
  }
  int CO_IRadded_size = 3;
  const char** CO_IRadded = new const char*[3];
  std::string hold = CO_vec[1];
  CO_IRadded[0] = hold.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO[1]);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 0 && data.find("@llvm.fmuladd.f32") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO[1]);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN '@llvm.fmuladd.f32' ");
      return 0;
    }
  } else {
    if (data.find("@llvm.fmuladd.f32") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO[1]);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN '@llvm.fmuladd.f32' ");
      return 0;
    }
  }
}

bool check_ffp_contract_fast(const char** Combination_CO, int Combination_CO_size,
                             int max_thread_pos, int fast_math_present) {
  std::string block_name = "ffp_contract";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  picojson::array retrieved_CO = get_array_parameters("compiler_option", block_name);
  if (retrieved_CO.size() < 3) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::vector<std::string> CO_vec;
  for (auto& indx : retrieved_CO) {
    CO_vec.push_back(indx.get<std::string>());
  }
  int CO_IRadded_size = 3;
  const char** CO_IRadded = new const char*[3];
  std::string hold = CO_vec[2];
  CO_IRadded[0] = hold.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO[2]);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 1 && data.find("contract") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO[2]);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'fmul contract' ");
      return 0;
    }
  } else {
    if (data.find("fmul contract") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO[2]);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'fmul contract' ");
      return 0;
    }
  }
}

bool check_fast_math_enabled(const char** Combination_CO, int Combination_CO_size,
                             int max_thread_pos, int fast_math_present) {
  std::string block_name = "fast_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (data.find("fmul fast") != -1) {
    return 1;
  } else {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR DOESN'T CONTAIN 'fmul fast' ");
    return 0;
  }
}

bool check_fast_math_disabled(const char** Combination_CO, int Combination_CO_size,
                              int max_thread_pos, int fast_math_present) {
  std::string block_name = "fast_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (data.find("fmul fast") != -1) {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR DOESN'T CONTAIN 'fmul fast' ");
    return 0;
  } else {
    return 1;
  }
}

bool check_slp_vectorize_enabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present) {
  std::string block_name = "slp_vectorize";
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  int CO_IRadded_size = 3;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  __half2 *a_d, *x_d, *y_d;
  __half2 a_h, x_h;
  a_h.data.x = 1.5;
  x_h.data.y = 3.0;
  CaptureStream capture(stderr);
  HIP_CHECK(hipMalloc(&a_d, sizeof(__half2)));
  HIP_CHECK(hipMalloc(&x_d, sizeof(__half2)));
  HIP_CHECK(hipMalloc(&y_d, sizeof(__half2)));
  HIP_CHECK(hipMemcpy(a_d, &a_h, sizeof(__half2), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(x_d, &x_h, sizeof(__half2), hipMemcpyHostToDevice));
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, slp_vectorize_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    int Combination_CO_IRadded_size = Combination_CO_size + 3;
    int b = 0;
    std::vector<std::string> add_ir_forcombi(Combination_CO_size + 3, "");
    const char** Combination_CO_IRadded = new const char*[Combination_CO_size + 3];
    for (int i = 0; i < Combination_CO_size + 3; ++i) {
      if (i == Combination_CO_size) {
        Combination_CO_IRadded[i] = "-fno-signed-zeros";
        Combination_CO_IRadded[i + 1] = "-mllvm";
        Combination_CO_IRadded[i + 2] = "-print-after=constmerge";
        break;
      }
      add_ir_forcombi[i] = Combination_CO[b];
      Combination_CO_IRadded[i] = add_ir_forcombi[i].c_str();
      b++;
    }
    capture.Begin();
    hiprtcResult compileResult{
        hiprtcCompileProgram(prog, Combination_CO_IRadded_size, Combination_CO_IRadded)};
    capture.End();
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler option : " << retrieved_CO);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size + 3; i++) {
        WARN(Combination_CO_IRadded[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    capture.Begin();
    hiprtcResult compileResult{hiprtcCompileProgram(prog, CO_IRadded_size, CO_IRadded)};
    capture.End();
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler option : " << retrieved_CO);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  std::string data = capture.getData();
  std::stringstream dataStream;
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  std::vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  void* kernelParam[] = {reinterpret_cast<void*>(a_d), reinterpret_cast<void*>(x_d),
                         reinterpret_cast<void*>(y_d)};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kername));
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  int times = 0;
  if (data.find("contract <2 x half>", 0) != -1) {
    times++;
  }
  int start = data.find("contract <2 x half>", 0) + 1;
  while (data.find("contract <2 x half>", start) != -1) {
    times++;
    start = data.find("contract <2 x half>", start) + 1;
  }
  if (times == 1) {
    return 1;
  } else if (times == 0) {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR DOESN'T CONTAIN 'fadd contract <2 x half>' ");
    return 0;
  } else {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR CONTAIN 'fadd contract <2 x half>' " << times << "times");
    WARN(" WHICH IS NOT EXPECTED (IT SHOULD BE PRESENT ONCE)");
    return 0;
  }
}

bool check_slp_vectorize_disabled(const char** Combination_CO, int Combination_CO_size,
                                  int max_thread_pos, int fast_math_present) {
  std::string block_name = "slp_vectorize";
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  int CO_IRadded_size = 3;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  __half2 *a_d, *x_d, *y_d;
  __half2 a_h, x_h;
  a_h.data.x = 1.5;
  x_h.data.y = 3.0;
  CaptureStream capture(stderr);
  HIP_CHECK(hipMalloc(&a_d, sizeof(__half2)));
  HIP_CHECK(hipMalloc(&x_d, sizeof(__half2)));
  HIP_CHECK(hipMalloc(&y_d, sizeof(__half2)));
  HIP_CHECK(hipMemcpy(a_d, &a_h, sizeof(__half2), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(x_d, &x_h, sizeof(__half2), hipMemcpyHostToDevice));
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, slp_vectorize_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    int Combination_CO_IRadded_size = Combination_CO_size + 3;
    int b = 0;
    std::vector<std::string> add_ir_forcombi(Combination_CO_size + 3, "");
    const char** Combination_CO_IRadded = new const char*[Combination_CO_size + 3];
    for (int i = 0; i < Combination_CO_size + 3; ++i) {
      if (i == Combination_CO_size) {
        Combination_CO_IRadded[i] = "-fno-signed-zeros";
        Combination_CO_IRadded[i + 1] = "-mllvm";
        Combination_CO_IRadded[i + 2] = "-print-after=constmerge";
        break;
      }
      add_ir_forcombi[i] = Combination_CO[b];
      Combination_CO_IRadded[i] = add_ir_forcombi[i].c_str();
      b++;
    }
    capture.Begin();
    hiprtcResult compileResult{
        hiprtcCompileProgram(prog, Combination_CO_IRadded_size, Combination_CO_IRadded)};
    capture.End();
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler option : " << retrieved_CO);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size + 3; i++) {
        WARN(Combination_CO_IRadded[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    capture.Begin();
    hiprtcResult compileResult{hiprtcCompileProgram(prog, CO_IRadded_size, CO_IRadded)};
    capture.End();
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler option : " << retrieved_CO);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  std::string data = capture.getData();
  std::stringstream dataStream;
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  std::vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  void* kernelParam[] = {reinterpret_cast<void*>(a_d), reinterpret_cast<void*>(x_d),
                         reinterpret_cast<void*>(y_d)};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kername));
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  int times = 0;
  if (data.find("contract <2 x half>", 0) != -1) {
    times++;
  }
  int start = data.find("contract <2 x half>", 0) + 1;
  while (data.find("contract <2 x half>", start) != -1) {
    times++;
    start = data.find("contract <2 x half>", start) + 1;
  }
  if (times == 2) {
    return 1;
  } else if (times < 2) {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR CONTAIN 'fadd contract <2 x half>' " << times << "times");
    WARN(" WHICH IS NOT EXPECTED(IT SHOULD BE PRESENT TWICE)");
    return 0;
  } else {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR CONTAIN 'fadd contract <2 x half>' " << times << "times");
    WARN(" WHICH IS NOT EXPECTED(IT SHOULD BE PRESENT TWICE)");
    return 0;
  }
}

bool check_macro(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                 int fast_math_present) {
  std::string block_name = "macro";
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  picojson::array Expected_Results = get_array_parameters("Expected_Results", block_name);
  const char* kername = kernel_name.c_str();
  std::vector<double> double_vec_expected;
  for (auto& indx : Expected_Results) {
    double_vec_expected.push_back(indx.get<double>());
  }
  std::vector<int> Expected_Results_int;
  for (auto& indx : double_vec_expected) {
    Expected_Results_int.push_back(static_cast<int>(indx));
  }
  const char* compiler_option = retrieved_CO.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, macro_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  int* macro_value_h;
  int* macro_value_d;
  macro_value_h = new int[1];
  HIP_CHECK(hipMalloc(&macro_value_d, sizeof(int)));
  *macro_value_h = 0;
  HIP_CHECK(hipMemcpy(macro_value_d, macro_value_h, sizeof(int), hipMemcpyHostToDevice));
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  std::vector<char> codec(codeSize);
  hiprtcGetCode(prog, codec.data());
  void* kernelParam[] = {macro_value_d};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kername));
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
  HIP_CHECK(hipMemcpy(macro_value_h, macro_value_d, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  if (*macro_value_h != Expected_Results_int[0]) {
    WARN("Compiler Option : " << compiler_option);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("EXPECTED RESULT DOES NOT MATCH");
    WARN("INPUT: " << compiler_option);
    WARN("EXPECTED OP : " << Expected_Results_int[0]);
    WARN("OBTAINED OP: " << *macro_value_h);
    return 0;
  } else {
    return 1;
  }
}

bool check_undef_macro(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                       int fast_math_present) {
  std::string block_name = "undef_macro";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  picojson::array comp_opt = get_array_parameters("compiler_option", block_name);
  if (comp_opt.size() < 2) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::vector<std::string> compiler_option;
  for (auto& indx : comp_opt) {
    compiler_option.push_back(indx.get<std::string>());
  }
  std::vector<std::string> variable(compiler_option.size(), "");
  const char** appended_compiler_options = new const char*[compiler_option.size()];
  for (int i = 0; i < compiler_option.size(); ++i) {
    variable[i] = compiler_option[i];
    appended_compiler_options[i] = variable[i].c_str();
  }
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, undef_macro_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        if (log.find("undeclared identifier")) {
          return 1;
        }
      } else {
        WARN("Compiler Option : " << appended_compiler_options[1]);
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
        WARN("Expected error : 'undeclared identifier' NOT GENERATED");
        return 0;
      }
    }
  } else {
    hiprtcResult compileResult{
        hiprtcCompileProgram(prog, compiler_option.size(), appended_compiler_options)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        if (log.find("undeclared identifier")) {
          return 1;
        }
      } else {
        WARN("Compiler Option : " << appended_compiler_options[0]);
        if (Combination_CO_size != -1) {
          WARN("FAILED IN COMBINATION :");
          for (int i = 0; i < Combination_CO_size; i++) {
            WARN(Combination_CO[i]);
          }
        }
        WARN("Expected error : 'undeclared identifier' NOT GENERATED");
        return 0;
      }
    }
  }
  WARN("Compiler Option : " << appended_compiler_options[0]);
  if (Combination_CO_size != -1) {
    WARN("FAILED IN COMBINATION :");
    for (int i = 0; i < Combination_CO_size; i++) {
      WARN(Combination_CO[i]);
    }
  }
  WARN("EXPECTED ERROR WAS NOT GENERATED");
  return 0;
}

bool check_header_dir(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                      int fast_math_present) {
  std::string block_name = "header_dir";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string compiler_option = get_string_parameters("compiler_option", block_name);
  if (compiler_option == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  picojson::array Headers = get_array_parameters("Headers", block_name);
  picojson::array depending_comp_optn = get_array_parameters("depending_comp_optn", block_name);
  picojson::array Src_headers = get_array_parameters("Src_headers", block_name);
  picojson::array Input_Thrd_Vals = get_array_parameters("Input_Vals", block_name);
  picojson::array Expected_Results = get_array_parameters("Expected_Results", block_name);
  std::string str = "pwd";
  const char* cmd = str.c_str();
  CaptureStream capture(stdout);
  capture.Begin();
  system(cmd);
  capture.End();
  std::string wor_dir = capture.getData();
  std::string break_dir = wor_dir.substr(0, wor_dir.find("build"));
  std::string append_str = "catch/unit/rtc/headers";
  std::string CO = compiler_option + " " + break_dir + append_str;
  const char* appended_CO = CO.c_str();
  std::vector<std::string> Headers_list;
  for (auto& indx : Headers) {
    Headers_list.push_back(indx.get<std::string>());
  }
  std::vector<std::string> Src_headers_list;
  for (auto& indx : Src_headers) {
    Src_headers_list.push_back(indx.get<std::string>());
  }
  std::vector<std::string> depending_co_list;
  for (auto& indx : depending_comp_optn) {
    depending_co_list.push_back(indx.get<std::string>());
  }
  std::vector<double> double_vec_target;
  for (auto& indx : Input_Thrd_Vals) {
    double_vec_target.push_back(indx.get<double>());
  }
  std::vector<int> Input_Thrd_Vals_int;
  for (auto& indx : double_vec_target) {
    Input_Thrd_Vals_int.push_back(static_cast<int>(indx));
  }
  std::vector<double> double_vec_expected;
  for (auto& indx : Expected_Results) {
    double_vec_expected.push_back(indx.get<double>());
  }
  std::vector<int> Expected_Results_int;
  for (auto& indx : double_vec_expected) {
    Expected_Results_int.push_back(static_cast<int>(indx));
  }
  std::vector<std::string> src_var_hdr_lst(Src_headers_list.size(), "");
  const char** src_hder_lst = new const char*[Src_headers_list.size()];
  for (int i = 0; i < Src_headers_list.size(); ++i) {
    src_var_hdr_lst[i] = Src_headers_list[i];
    src_hder_lst[i] = src_var_hdr_lst[i].c_str();
  }
  std::vector<std::string> var_hdr_lst(Headers_list.size(), "");
  const char** hder_lst = new const char*[Headers_list.size()];
  for (int i = 0; i < Headers_list.size(); ++i) {
    var_hdr_lst[i] = Headers_list[i];
    hder_lst[i] = var_hdr_lst[i].c_str();
  }
  for (int senario = 0; senario < Input_Thrd_Vals_int.size(); senario++) {
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, header_dir_string, kername, Headers_list.size(),
                                     src_hder_lst, hder_lst));
    if (Combination_CO_size != -1) {
      hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << appended_CO);
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
          WARN(log);
        }
        return 0;
      }
    } else {
      hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &appended_CO)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << appended_CO);
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
          WARN(log);
        }
        return 0;
      }
    }
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    std::vector<char> codec(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
    int value_h = 0;
    int* ptr_value_h = &value_h;
    int input_h = Input_Thrd_Vals_int[senario];
    int* ptr_input_h = &input_h;
    int* value_d;
    int* input_d;
    HIP_CHECK(hipMalloc(&value_d, sizeof(int)));
    HIP_CHECK(hipMalloc(&input_d, sizeof(int)));
    HIP_CHECK(hipMemcpy(value_d, ptr_value_h, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(input_d, ptr_input_h, sizeof(int), hipMemcpyHostToDevice));
    void* kernelParam[] = {value_d, input_d};
    auto size = sizeof(kernelParam);
    void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                                HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    hipModule_t module;
    hipFunction_t function;
    HIP_CHECK(hipModuleLoadData(&module, codec.data()));
    HIP_CHECK(hipModuleGetFunction(&function, module, kername));
    HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
    HIP_CHECK(hipMemcpy(ptr_value_h, value_d, sizeof(int), hipMemcpyDeviceToHost));
    if (*ptr_value_h != Expected_Results_int[senario]) {
      WARN("Compiler Option : " << appended_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN(" EXPECTED RESULT DOES NOT MATCH FOR " << senario);
      WARN("th ITERATION (start iteration is 0 ) ");
      WARN(" INPUT: " << Input_Thrd_Vals_int[senario]);
      WARN(" EXPECTED OP: " << Expected_Results_int[senario]);
      WARN(" OBTAINED OP: " << *ptr_value_h);
      return 0;
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipModuleUnload(module));
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  }
  return 1;
}

bool check_warning(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                   int fast_math_present) {
  std::string block_name = "warning";
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  const char* compiler_option = retrieved_CO.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, warning_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    if (-1 != log.find("#warning")) {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN(" WARNING MESSAGE IS PRINTING WHICH IS NOT SUPRESSED ");
      return 0;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

bool check_Rpass_inline(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                        int fast_math_present) {
  std::string block_name = "Rpass_inline";
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  const char* compiler_option = retrieved_CO.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, max_thread_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    if (log.find("inlined into")) {
      return 1;
    } else {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("EXPECTED STRING 'inlined into' IS NOT PRESENT IN LOG ");
      return 0;
    }
  } else {
    WARN("Compiler Option : " << compiler_option);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN(" LOG WITH EXPECTED STRING 'inlined into' IS NOT PRESENT ");
    return 0;
  }
}

bool check_conversionerror_enabled(const char** Combination_CO, int Combination_CO_size,
                                   int max_thread_pos, int fast_math_present) {
  std::string block_name = "error";
  picojson::array retrieved_CO = get_array_parameters("compiler_option", block_name);
  if (retrieved_CO.size() < 4) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::vector<std::string> CO_vec;
  for (auto& indx : retrieved_CO) {
    CO_vec.push_back(indx.get<std::string>());
  }
  std::string variable = CO_vec[0];
  const char* compiler_option = variable.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, error_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
  }
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::string variable = "error";
    if (-1 != log.find(variable)) {
      return 1;
    } else {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("ERROR MSG : '" << variable << "' NOT FOUND");
      return 0;
    }
  } else {
    WARN("Compiler Option : " << compiler_option);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("LOG IS NOT GENERATED");
    WARN("maybe due to presence of '-w' compiler option");
    return 0;
  }
}

bool check_conversionerror_disabled(const char** Combination_CO, int Combination_CO_size,
                                    int max_thread_pos, int fast_math_present) {
  std::string block_name = "error";
  picojson::array retrieved_CO = get_array_parameters("compiler_option", block_name);
  if (retrieved_CO.size() < 4) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::vector<std::string> CO_vec;
  for (auto& indx : retrieved_CO) {
    CO_vec.push_back(indx.get<std::string>());
  }
  std::string variable = CO_vec[1];
  const char* compiler_option = variable.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, error_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
  }
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    if (-1 != log.find("error")) {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("LOG IS PRESENT WITH ERROR WHICH IS NOT EXPECTED : ");
      WARN("maybe due to presence of '-w' compiler option");
      return 0;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

bool check_conversionwarning_enabled(const char** Combination_CO, int Combination_CO_size,
                                     int max_thread_pos, int fast_math_present) {
  std::string block_name = "error";
  picojson::array retrieved_CO = get_array_parameters("compiler_option", block_name);
  if (retrieved_CO.size() < 4) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::vector<std::string> CO_vec;
  for (auto& indx : retrieved_CO) {
    CO_vec.push_back(indx.get<std::string>());
  }
  std::string variable = CO_vec[2];
  const char* compiler_option = variable.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, error_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
  }
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::string variable = "warning";
    if (-1 != log.find(variable)) {
      return 1;
    } else {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("LOG DOESN'T CONTAIN WARNING AS EXP : " << compiler_option);
      return 0;
    }
  } else {
    WARN("Compiler Option : " << compiler_option);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("LOG IS NOT GENERATED");
    return 0;
  }
}

bool check_conversionwarning_disabled(const char** Combination_CO, int Combination_CO_size,
                                      int max_thread_pos, int fast_math_present) {
  std::string block_name = "error";
  picojson::array retrieved_CO = get_array_parameters("compiler_option", block_name);
  if (retrieved_CO.size() < 4) {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::vector<std::string> CO_vec;
  for (auto& indx : retrieved_CO) {
    CO_vec.push_back(indx.get<std::string>());
  }
  std::string variable = CO_vec[3];
  const char* compiler_option = variable.c_str();
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, error_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
  }
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    if (-1 != log.find("warning")) {
      WARN("Compiler Option : " << compiler_option);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("WARNING IS GENERATED WHICH IS NOT EXPECTED");
      WARN(compiler_option);
      return 0;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

bool check_max_thread(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                      int fast_math_present) {
  std::string block_name = "max_thread";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  std::string default_CO = get_string_parameters("kernel_name", block_name);
  picojson::array Target_Thrd_Vals = get_array_parameters("Target_Vals", block_name);
  picojson::array Input_Thrd_Vals = get_array_parameters("Input_Vals", block_name);
  picojson::array Expected_Results = get_array_parameters("Expected_Results", block_name);
  const char* kername = kernel_name.c_str();
  std::string compiler_option = get_string_parameters("compiler_option", block_name);
  if (compiler_option == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::vector<double> double_vec_target;
  for (auto& indx : Target_Thrd_Vals) {
    double_vec_target.push_back(indx.get<double>());
  }
  std::vector<int> Target_Thrd_Vals_int;
  for (auto& indx : double_vec_target) {
    Target_Thrd_Vals_int.push_back(static_cast<int>(indx));
  }
  int a = 0;
  std::vector<std::string> variable(Target_Thrd_Vals_int.size(), "");
  const char** appended_compiler_options = new const char*[Target_Thrd_Vals_int.size()];
  for (int i = 0; i < Target_Thrd_Vals_int.size(); i++) {
    variable[i] = compiler_option + std::to_string(Target_Thrd_Vals_int[i]);
    appended_compiler_options[i] = variable[i].c_str();
  }
  std::vector<double> double_vec_input;
  for (auto& indx : Input_Thrd_Vals) {
    double_vec_input.push_back(indx.get<double>());
  }
  std::vector<int> Input_Thrd_Vals_int;
  for (auto& indx : double_vec_input) {
    Input_Thrd_Vals_int.push_back(static_cast<int>(indx));
  }
  std::vector<double> double_vec_expected;
  for (auto& indx : Expected_Results) {
    double_vec_expected.push_back(indx.get<double>());
  }
  std::vector<int> Expected_Results_int;
  for (auto& indx : double_vec_expected) {
    Expected_Results_int.push_back(static_cast<int>(indx));
  }
  int pass_count = 0;
  int inc = (Input_Thrd_Vals_int.size() / Target_Thrd_Vals_int.size());
  int start = 0;
  int check, test_case;
  for (int senario = 0; senario < Target_Thrd_Vals_int.size(); senario++) {
    if (Target_Thrd_Vals_int[senario] == 0) {
      check = 0;
      for (test_case = start; test_case < (start + inc); test_case++) {
        if (check == Expected_Results_int[test_case]) {
          pass_count++;
        }
      }
      start += inc;
      continue;
    }
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, max_thread_string, kername, 0, NULL, NULL));
    if (Combination_CO_size != -1) {
      std::string max_thread_string = variable[senario];
      Combination_CO[max_thread_pos] = max_thread_string.c_str();
      hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << appended_compiler_options[senario]);
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
          WARN(log);
        }
        return 0;
      }
    } else {
      hiprtcResult compileResult{
          hiprtcCompileProgram(prog, 1, &appended_compiler_options[senario])};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << appended_compiler_options[senario]);
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
          WARN(log);
        }
        return 0;
      }
    }
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    std::vector<char> codec(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
    for (test_case = start; test_case < (start + inc); test_case++) {
      int num_threads_h = 0;
      int* ptr_num_threads_h = &num_threads_h;
      int* Thread_count_d;
      HIP_CHECK(hipMalloc(&Thread_count_d, sizeof(int)));
      HIP_CHECK(hipMemcpy(Thread_count_d, ptr_num_threads_h, sizeof(int), hipMemcpyHostToDevice));
      void* kernelParam[] = {Thread_count_d};
      auto size = sizeof(kernelParam);
      void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                                  HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
      hipModule_t module;
      hipFunction_t function;
      HIP_CHECK(hipModuleLoadData(&module, codec.data()));
      HIP_CHECK(hipModuleGetFunction(&function, module, kername));
      hipError_t status = hipModuleLaunchKernel(function, 1, 1, 1, Input_Thrd_Vals_int[test_case],
                                                1, 1, 0, 0, nullptr, kernel_parameter);
      HIP_CHECK(hipMemcpy(ptr_num_threads_h, Thread_count_d, sizeof(int), hipMemcpyDeviceToHost));
      if ((status == hipSuccess) && (num_threads_h <= Target_Thrd_Vals_int[senario])) {
        check = 1;
      } else {
        check = 0;
      }
      if (check != Expected_Results_int[test_case]) {
        WARN("Compiler Option : " << appended_compiler_options[senario]);
        if (Combination_CO_size != -1) {
          WARN("FAILED IN COMBINATION :");
          std::string max_thread_string = variable[senario];
          Combination_CO[max_thread_pos] = max_thread_string.c_str();
          for (int i = 0; i < Combination_CO_size; i++) {
            WARN(Combination_CO[i]);
          }
        }
        WARN("EXPECTED RESULT DOES NOT MATCH FOR " << test_case);
        WARN("th ITERATION (start iteration is 0 ) ");
        WARN("IP THREAD VAL: " << Input_Thrd_Vals_int[test_case]);
        WARN("EXPECTED OP: " << Expected_Results_int[test_case]);
        WARN("OBTAINED OP: " << check);
        return 0;
      }
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipModuleUnload(module));
    }
    start += inc;
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  }
  return 1;
}

bool check_unsafe_atomic_enabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present) {
  std::string block_name = "unsafe_atomic";
  std::string compiler_option = get_string_parameters("compiler_option", block_name);
  if (compiler_option == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  const char* compiler_option_cstr = compiler_option.c_str();
  float* A_d;
  const int N = 1000;
  float A_h[N];
  float Nbytes = N * sizeof(float);
  double sum_w = 0, sum_wo = 0, sum_tocheck = 0;
  for (int i = 0; i < N; i++) {
    A_h[i] = 0.1f;
    sum_tocheck += A_h[i] + 0.2f;
  }
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  for (int senario = 0; senario < 2; senario++) {
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, unsafe_atomic_string, kername, 0, NULL, NULL));
    if (Combination_CO_size != -1) {
      hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << compiler_option);
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
          WARN(log);
        }
        return 0;
      }
    } else {
      hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option_cstr)};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("Compiler Option : " << compiler_option);
        WARN("hiprtcCompileProgram() api failed!! with error code: ");
        WARN(compileResult);
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
        if (logSize) {
          std::string log(logSize, '\0');
          HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
          WARN(log);
        }
        return 0;
      }
    }
    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
    std::vector<char> codec(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
    void* kernelParam[] = {A_d};
    auto size = sizeof(kernelParam);
    void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                                HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
    hipModule_t module;
    hipFunction_t function;
    HIP_CHECK(hipModuleLoadData(&module, codec.data()));
    HIP_CHECK(hipModuleGetFunction(&function, module, kername));
    HIP_CHECK(hipModuleLaunchKernel(function, N, 1, 1, N, 1, 1, 0, 0, nullptr, kernel_parameter));
    HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
      if (senario == 0) {
        sum_wo += A_h[i];
      } else {
        sum_w += A_h[i];
      }
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipModuleUnload(module));
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  }
  if (sum_w != sum_tocheck) {
    return 1;
  } else {
    WARN("Compiler Option : " << compiler_option);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("EXPECTED : " << sum_w << " != " << sum_tocheck);
    return 0;
  }
}

bool check_unsafe_atomic_disabled(const char** Combination_CO, int Combination_CO_size,
                                  int max_thread_pos, int fast_math_present) {
  std::string block_name = "unsafe_atomic";
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  const char* compiler_option = retrieved_CO.c_str();
  float* A_d;
  const int N = 1000;
  float A_h[N];
  float Nbytes = N * sizeof(float);
  double sum = 0, sum_tocheck = 0;
  for (int i = 0; i < N; i++) {
    A_h[i] = 0.1f;
    sum_tocheck += A_h[i] + 0.2f;
  }
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, unsafe_atomic_string, kername, 0, NULL, NULL));
  if (Combination_CO_size != -1) {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, Combination_CO_size, Combination_CO)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  } else {
    hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, &compiler_option)};
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler Option : " << compiler_option);
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return 0;
    }
  }
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  std::vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  void* kernelParam[] = {A_d};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kername));
  HIP_CHECK(hipModuleLaunchKernel(function, N, 1, 1, N, 1, 1, 0, 0, nullptr, kernel_parameter));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    sum += A_h[i];
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  if (sum == sum_tocheck) {
    return 1;
  } else {
    WARN("Compiler Option : " << compiler_option);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("EXPECTED RESULT IS NOT OBTAINED ");
    WARN("EXPECTED RESULT: " << sum_tocheck);
    WARN("OBTAINED RESULT: " << sum);
    return 0;
  }
}

bool check_infinite_num_enabled(const char** Combination_CO, int Combination_CO_size,
                                int max_thread_pos, int fast_math_present) {
  std::string block_name = "infinite_num";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 0 && data.find("contract") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'contract' ");
      return 0;
    }
  } else {
    if (data.find("ninf") != -1) {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'ninf' ");
      return 0;
    } else {
      return 1;
    }
  }
}

bool check_infinite_num_disabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present) {
  std::string block_name = "infinite_num";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 1 && data.find("fmul fast") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'fmul fast' ");
      return 0;
    }
  } else {
    if (data.find("ninf") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'ninf' ");
      return 0;
    }
  }
}

bool check_NAN_num_enabled(const char** Combination_CO, int Combination_CO_size, int max_thread_pos,
                           int fast_math_present) {
  std::string block_name = "NAN_num";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME ");
    WARN(block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 0 && data.find("contract") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'contract' ");
      return 0;
    }
  } else {
    if (data.find("nnan") != -1) {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'nnan' ");
      return 0;
    } else {
      return 1;
    }
  }
}

bool check_NAN_num_disabled(const char** Combination_CO, int Combination_CO_size,
                            int max_thread_pos, int fast_math_present) {
  std::string block_name = "NAN_num";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 1 && data.find("fmul fast") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'fmul fast' ");
      return 0;
    }
  } else {
    if (data.find("nnan") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'nnan' ");
      return 0;
    }
  }
}

bool check_finite_math_enabled(const char** Combination_CO, int Combination_CO_size,
                               int max_thread_pos, int fast_math_present) {
  std::string block_name = "finite_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 1 && data.find("fmul fast") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'fmul fast'");
      return 0;
    }
  } else {
    if (data.find("nnan") != -1 && (data.find("ninf") != -1)) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'nnan' or 'ninf' or both ");
      return 0;
    }
  }
}

bool check_finite_math_disabled(const char** Combination_CO, int Combination_CO_size,
                                int max_thread_pos, int fast_math_present) {
  std::string block_name = "finite_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 0 && data.find("contract") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'contract'");
      return 0;
    }
  } else {
    if (data.find("nnan") != -1 && (data.find("ninf") != -1)) {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR CONTAIN 'nnan' or 'ninf' or both WHICH IS NOT EXPECTED ");
      return 0;
    } else {
      return 1;
    }
  }
}

bool check_associative_math_enabled(const char** Combination_CO, int Combination_CO_size,
                                    int max_thread_pos, int fast_math_present) {
  std::string block_name = "associative_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 4, a = 0;
  const char** CO_IRadded = new const char*[4];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-fno-signed-zeros";
  CO_IRadded[2] = "-mllvm";
  CO_IRadded[3] = "-print-after=constmerge";
  std::string data;
  if (Combination_CO_size != -1) {
    int Combination_CO_IRadded_size = Combination_CO_size + 1;
    int b = 0;
    std::vector<std::string> add_ir_forcombi(Combination_CO_size + 1, "");
    const char** Combination_CO_IRadded = new const char*[Combination_CO_size + 1];
    for (int i = 0; i < Combination_CO_size + 1; ++i) {
      if (i == Combination_CO_size) {
        Combination_CO_IRadded[i] = "-fno-signed-zeros";
        break;
      }
      add_ir_forcombi[i] = Combination_CO[b];
      Combination_CO_IRadded[i] = add_ir_forcombi[i].c_str();
      b++;
    }
    data = checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO_IRadded,
                       Combination_CO_IRadded_size);
  } else {
    data = checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  }
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 1 && data.find("fmul fast") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'fmul fast' ");
      return 0;
    }
  } else {
    if (data.find("reassoc") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'reassoc' ");
      WARN(data);
      return 0;
    }
  }
}

bool check_associative_math_disabled(const char** Combination_CO, int Combination_CO_size,
                                     int max_thread_pos, int fast_math_present) {
  std::string block_name = "associative_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 4, a = 0;
  const char** CO_IRadded = new const char*[4];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-fno-signed-zeros";
  CO_IRadded[2] = "-mllvm";
  CO_IRadded[3] = "-print-after=constmerge";
  std::string data;
  if (Combination_CO_size != -1) {
    int Combination_CO_IRadded_size = Combination_CO_size + 1;
    int b = 0;
    std::vector<std::string> add_ir_forcombi(Combination_CO_size + 1, "");
    const char** Combination_CO_IRadded = new const char*[Combination_CO_size + 1];
    for (int i = 0; i < Combination_CO_size + 1; ++i) {
      if (i == Combination_CO_size) {
        Combination_CO_IRadded[i] = "-fno-signed-zeros";
        break;
      }
      add_ir_forcombi[i] = Combination_CO[b];
      Combination_CO_IRadded[i] = add_ir_forcombi[i].c_str();
      b++;
    }
    data = checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO_IRadded,
                       Combination_CO_IRadded_size);
  } else {
    data = checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  }
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 0 && data.find("contract") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'contract' ");
      return 0;
    }
  } else {
    if (data.find("reassoc") != -1) {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR CONTAIN 'reassoc' WHICH IS NOT EXPECTED ");
      return 0;
    } else {
      return 1;
    }
  }
}

bool check_signed_zeros_enabled(const char** Combination_CO, int Combination_CO_size,
                                int max_thread_pos, int fast_math_present) {
  std::string block_name = "signed_zeros";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 0 && data.find("contract") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'contract' ");
      return 0;
    }
  } else {
    if (data.find("nsz") != -1) {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR CONTAIN 'nsz' WHICH IS NOT EXPECTED ");
      return 0;
    } else {
      return 1;
    }
  }
}

bool check_signed_zeros_disabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present) {
  std::string block_name = "signed_zeros";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (fast_math_present != -1) {
    if (fast_math_present == 1 && data.find("fmul fast") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'fmul fast' ");
      return 0;
    }
  } else {
    if (data.find("nsz") != -1) {
      return 1;
    } else {
      WARN("Compiler option : " << retrieved_CO);
      if (Combination_CO_size != -1) {
        WARN("FAILED IN COMBINATION :");
        for (int i = 0; i < Combination_CO_size; i++) {
          WARN(Combination_CO[i]);
        }
      }
      WARN("IR DOESN'T CONTAIN 'nsz' ");
      return 0;
    }
  }
}

bool check_trapping_math_enabled(const char** Combination_CO, int Combination_CO_size,
                                 int max_thread_pos, int fast_math_present) {
  std::string block_name = "trapping_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (data.find("\"no-trapping-math\"=\"true\"") != -1) {
    return 1;
  } else {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR DOESN'T CONTAIN '\"no-trapping-math\"=\"true\"'");
    return 0;
  }
}

bool check_trapping_math_disabled(const char** Combination_CO, int Combination_CO_size,
                                  int max_thread_pos, int fast_math_present) {
  std::string block_name = "trapping_math";
  std::string kernel_name = get_string_parameters("kernel_name", block_name);
  const char* kername = kernel_name.c_str();
  std::string retrieved_CO = get_string_parameters("reverse_compiler_option", block_name);
  if (retrieved_CO == "") {
    WARN("COMPILER OPTION NOT PROVIDED FOR BLOCK NAME " << block_name);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    return 0;
  }
  int CO_IRadded_size = 3, a = 0;
  const char** CO_IRadded = new const char*[3];
  CO_IRadded[0] = retrieved_CO.c_str();
  CO_IRadded[1] = "-mllvm";
  CO_IRadded[2] = "-print-after=constmerge";
  std::string data =
      checking_IR(kername, CO_IRadded, CO_IRadded_size, Combination_CO, Combination_CO_size);
  if (data == "") {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR NOT GENERATED");
    return 0;
  }
  if (data.find("\"no-trapping-math\"=\"true\"") != -1) {
    return 1;
  } else {
    WARN("Compiler option : " << retrieved_CO);
    if (Combination_CO_size != -1) {
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
    }
    WARN("IR DOESN'T CONTAIN '\"no-trapping-math\"=\"true\"'");
    return 0;
  }
}

std::string checking_IR(const char* kername, const char** extra_CO_IRadded,
                        int extra_CO_IRadded_size, const char** Combination_CO,
                        int Combination_CO_size) {
  float *A_d, *B_d, *C_d;
  float *A_h, *B_h, *C_h, *result;
  float Nbytes = sizeof(float);
  A_h = new float[1];
  B_h = new float[1];
  C_h = new float[1];
  result = new float[1];
  for (int i = 0; i < 1; i++) {
    A_h[i] = 0.1f;
    B_h[i] = 0.1f;
    C_h[i] = 0.1f;
    result[i] = 0.2f;
  }
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&B_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_d, C_h, Nbytes, hipMemcpyHostToDevice));
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, ffp_contract_string, kername, 0, NULL, NULL));
  int Combination_CO_IRadded_size;
  CaptureStream capture(stderr);
  if (Combination_CO_size != -1) {
    Combination_CO_IRadded_size = Combination_CO_size + 2;
    int b = 0;
    std::vector<std::string> add_ir_forcombi(Combination_CO_size + 2, "");
    const char** Combination_CO_IRadded = new const char*[Combination_CO_size + 2];
    for (int i = 0; i < Combination_CO_size + 2; ++i) {
      if (i == Combination_CO_size) {
        Combination_CO_IRadded[i] = "-mllvm";
        Combination_CO_IRadded[i + 1] = "-print-after=constmerge";
        break;
      }
      add_ir_forcombi[i] = Combination_CO[b];
      Combination_CO_IRadded[i] = add_ir_forcombi[i].c_str();
      b++;
    }
    capture.Begin();
    hiprtcResult compileResult{
        hiprtcCompileProgram(prog, Combination_CO_IRadded_size, Combination_CO_IRadded)};
    capture.End();
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("Compiler option : " << extra_CO_IRadded[0]);
      WARN("FAILED IN COMBINATION :");
      for (int i = 0; i < Combination_CO_size; i++) {
        WARN(Combination_CO[i]);
      }
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return "";
    }
  } else {
    capture.Begin();
    hiprtcResult compileResult{hiprtcCompileProgram(prog, extra_CO_IRadded_size, extra_CO_IRadded)};
    capture.End();
    if (!(compileResult == HIPRTC_SUCCESS)) {
      WARN("hiprtcCompileProgram() api failed!! with error code: ");
      WARN(compileResult);
      size_t logSize;
      HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
      if (logSize) {
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
      }
      return "";
    }
  }
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  std::vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  void* kernelParam[] = {A_d, B_d, C_d};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, kername));
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, kernel_parameter));
  HIP_CHECK(hipMemcpy(result, C_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < 1; i++) {
    if (result[i] != ((A_h[i] * B_h[i]) + C_h[i])) {
      return "";
    }
  }
  std::string data = capture.getData();
  std::stringstream dataStream;
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return data;
}
