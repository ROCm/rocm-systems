/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <vector>
#include <fstream>

static constexpr auto saxpy{
    R"(
extern "C"
__global__
void saxpy(float a, float* x, float* y, float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
           out[tid] = a * x[tid] + y[tid];
    }
}
)"};

static constexpr size_t num_iterations = 200000;  // 20K times

bool CommitBCToFile(char* executable, size_t exe_size, const std::string& bit_code_file) {
  std::fstream bc_file;
  bc_file.open(bit_code_file, std::ios::out | std::ios::binary);

  if (!bc_file) {
    WARN("Cannot create file");
    ;
    return false;
  }

  bc_file.write(executable, exe_size);
  bc_file.close();

  return true;
}

void GetCodeObjectUsingRTC(size_t codeSize, std::vector<char>& code) {
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog, saxpy, "saxpy.cu", 0, nullptr, nullptr);

  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
#else
  std::string sarg = std::string("--fmad=false");
#endif
  const char* options[] = {sarg.c_str()};
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    WARN("Log: " << log);
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  code.resize(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}

/**
 * Test Description
 * ------------------------
 *    - Compiles a kernel using hiprtc (mainly to use dynamically find gpu arch).
 *    - Commits the compiled code object to a file.
 *    - Run hipModuleLoad and hipModuleUnload for 20K times.
 *
 * Test source
 * ------------------------
 *    - unit/module/hipModuleLoadUnload.cc
 *
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */

HIP_TEST_CASE(Stress_hipModuleLoadUnload) {
  size_t code_size = 0;
  std::vector<char> code;
  GetCodeObjectUsingRTC(code_size, code);

  std::string co_file("temp_code_object.txt");
  if (!CommitBCToFile(code.data(), code.size(), co_file)) {
    WARN("Could not commit CO to file ");
    return;
  }

  for (size_t iter_idx = 0; iter_idx < num_iterations; ++iter_idx) {
    if ((iter_idx % 2000) == 0) {
      std::cout << "Iteration :" << iter_idx << std::endl;
      UNSCOPED_INFO("Unscoped Prints");
    }

    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, co_file.c_str()));
    HIP_CHECK(hipModuleUnload(module));
  }
}
