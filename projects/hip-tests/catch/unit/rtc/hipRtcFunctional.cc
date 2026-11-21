/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* This test verifies few functional scenarios of hiprtc
 * hipRTC program should be created even if the name passed is empty or null
 * hipRTC program compilation should succeed even if gpu arch is not specified in the options
 * hipRTC should be able to compile kernels using  __forceinline__ keyword
 * hipRTC program compilation should succeed when the name is not a valid file name, such as a path
 *   with a Windows drive letter or a name with characters Windows does not allow in file names
 */
#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

const char* funname = "testinline";
static constexpr auto code{
    R"(
__forceinline__ __device__ float f() { return 123.4f; }
extern "C"
__global__ void testinline()
{
 f();
}
)"};

HIP_TEST_CASE(Unit_hiprtc_functional) {
  using namespace std;
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 0, 0)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  vector<char> codec(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, codec.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

#if HT_NVIDIA
  int device = 0;
  HIPCHECK(hipInit(0));
  hipCtx_t ctx;
  HIPCHECK(hipCtxCreate(&ctx, 0, device));
#endif

  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoadData(&module, codec.data()));
  HIP_CHECK(hipModuleGetFunction(&function, module, funname));

  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, 0, nullptr, 0));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipModuleUnload(module));

#if HT_NVIDIA
  HIPCHECK(hipCtxDestroy(ctx));
#endif
}

// The program name is only a name. Libraries such as CuPy pass the full path of
// a cache file, which on Windows starts with a drive letter (ROCm/TheRock#9848),
// and a name may hold characters Windows does not allow in file names.
HIP_TEST_CASE(Unit_hiprtc_name_not_a_file_name) {
  using namespace std;
  // "C:\src\", "C:\src\..", "src/" and ".." leave no file name; hiprtc then generates one.
  const char* name = GENERATE("C:\\src\\testinline.cu", "C:/src/testinline.cu", "C:testinline.cu",
                              "/tmp/a:b/testinline.cu", "C:\\src\\", "C:\\src\\..",
                              "testinline<float>.cu", "a|b.cu", "what?.cu", "k*.cu", "\"k\".cu",
                              "tab\tname.cu", "C:\\src\\k<1>.cu", "../../../../k.cu", "src/", "..");
  INFO("program name: " << name);
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, code, name, 0, nullptr, nullptr));

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 0, 0)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));
  REQUIRE(codeSize > 0);
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
}
