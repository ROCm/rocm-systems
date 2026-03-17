/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

// This test verifies successful compilation of source to bitcode (with -fgpu-rdc option) when the
// hiprtc kernel has an undefined function which will be linked in the later stages using hiprtc
// linker APIs

TEST_CASE(Unit_hiprtc_bitcode_undefined_function) {
  using namespace std;

  static constexpr auto kernel{
      R"(
      __device__ void foo();

      extern "C"
      __global__ void gpu_kernel() {
          foo();
      }
  )"};

  hiprtcProgram prog;
  hiprtcCreateProgram(&prog,            // prog
                      kernel,           // buffer
                      "gpu_kernel.cu",  // name
                      0, nullptr, nullptr);
  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device));

  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
  const char* options[] = {sarg.c_str(), "-fgpu-rdc"};

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, options)};

  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }

  REQUIRE(compileResult == HIPRTC_SUCCESS);

  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}
