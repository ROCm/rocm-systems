/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <hip/hiprtc.h>

#include <sstream>
#include <string>
#include <system_error>
#include <vector>

static constexpr auto profileKernel = R"(
extern "C" __global__ void profile_kernel(int* output) {
  if (threadIdx.x == 0)
    *output = 1;
}
)";

HIP_TEST_CASE(Unit_hiprtc_ProfileRuntimeLink) {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  hiprtcProgram program;
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, profileKernel, "profile_kernel.hip", 0, nullptr, nullptr));

  std::string archOption = std::string("--offload-arch=") + props.gcnArchName;
  const char* options[] = {archOption.c_str(), "-fprofile-generate"};
  hiprtcResult compileResult = hiprtcCompileProgram(program, 2, options);
  std::string log;
  if (compileResult != HIPRTC_SUCCESS) {
    size_t logSize = 0;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &logSize));
    if (logSize > 1) {
      log.resize(logSize, '\0');
      HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
      log.resize(logSize - 1);
    }
    HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  }
  INFO(log);
  if (compileResult == HIPRTC_ERROR_COMPILATION) {
    // COMGR may use an installed or embedded runtime; inspect its diagnostic
    // instead of guessing the compiler's resource directory.
    const std::string missingArchive =
        "libclang_rt.profile.a: " +
        std::make_error_code(std::errc::no_such_file_or_directory).message();
    std::istringstream logStream(log);
    std::string line;
    while (std::getline(logStream, line)) {
      if (line.find("ld.lld: error: cannot open ") != std::string::npos &&
          line.find(missingArchive) != std::string::npos) {
        HIP_SKIP_TEST("AMDGPU profiling runtime archive is unavailable.");
      }
    }
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);

  // Code object generation requires the final COMGR link action to resolve
  // the profiling symbols emitted by the compile action.
  size_t codeSize = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(program, &codeSize));
  REQUIRE(codeSize > 0);

  std::vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(program, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
}
