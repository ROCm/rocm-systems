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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hiprtc_std_headers hiprtc_std_headers
 * @{
 * @ingroup hiprtcHeadersTest
 * `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
 *                                    int numOptions,
 *                                    const char** options);` -
 * This test verifies hiprtc compilation when the C++ std headers such as type_traits,
 * iterator etc. are included in the program. HIPRTC also defines few std templates
 * and this should not cause conflicts with std headers.
 */

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

static constexpr auto source{
    R"(
#if __has_include(<type_traits>)
#include <type_traits>

template <typename T> __global__ void stdheader(T a, bool* passed) {
   *passed = std::is_same<T, float>::value;

   *passed &= (std::is_integral<T*>::value == false);

   *passed &= (std::is_arithmetic<T*>::value == false);

   *passed &= (std::is_floating_point<T*>::value == false);
}
#else
template <typename T> __global__ void stdheader(T a, bool* passed) {
   *passed = __hip_internal::is_same<T, float>::value;

   *passed &= (__hip_internal::is_integral<T>::value == false);

   *passed &= (__hip_internal::is_arithmetic<T>::value);

   *passed &= (__hip_internal::is_floating_point<T>::value);

}
#endif
)"};

/**
 * Test Description
 * ------------------------
 *  - Executes `hiprtcCompileProgram` with additional C++ std
 *    headers used in kernel source
 * Test source
 * ------------------------
 *  - unit/rtc/stdheaders.cc
 * Test requirements
 * ------------------------
 *  - ROCM_VERSION >= 7.0
 */
TEST_CASE("Unit_hiprtc_stdheaders") {
  HipTest::HIP_SKIP_TEST("Test disabled due to incorrect ROCm version");
  return;

  using namespace std;
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, source, "stdheader.cu", 0, nullptr, nullptr));

  string str = "stdheader<float>";
  hiprtcAddNameExpression(prog, str.c_str());

  string sarg = string("-I./headers");
  const char* options[] = {sarg.c_str()};

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};
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

  vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

  // Do hip malloc first so that we dont need to do a cuInit manually before calling hipModule APIs

  bool* dResult;
  HIP_CHECK(hipMalloc(&dResult, sizeof(bool)));

  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  const char* name;
  hiprtcGetLoweredName(prog, str.c_str(), &name);
  HIP_CHECK(hipModuleGetFunction(&kernel, module, name));

  float a = 5.1f;
  unique_ptr<bool> hResult{new bool};
  *hResult = false;

  HIP_CHECK(hipMemcpy(dResult, hResult.get(), sizeof(bool), hipMemcpyHostToDevice));

  struct {
    float a_;
    bool* b_;
  } args{a, dResult};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config));

  HIP_CHECK(hipMemcpy(hResult.get(), dResult, sizeof(bool), hipMemcpyDeviceToHost));

  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  REQUIRE(*hResult == true);
}

/**
 * End doxygen group hiprtc_std_headers.
 * @}
 */
