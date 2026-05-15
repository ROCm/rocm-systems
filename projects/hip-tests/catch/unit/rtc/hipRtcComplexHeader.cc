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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hiprtc_Complex_HeaderTst hiprtc_Complex_HeaderTst
 * @{
 * @ingroup hiprtcHeadersTest
 * `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
 *                                  int numOptions,
 *                                  const char** options);` -
 * These test cases are target including various header file in kernel
 * string and compile using the api mentioned above.
 */

#include <hip_test_common.hh>
#include <hip/hiprtc.h>
#include <hip_test_defgroups.hh>

static constexpr auto hip_complex_basic_string{
    R"(
extern "C"
__global__ void hip_complex_basic_kernel(unsigned int *res) {
  hipFloatComplex a = make_hipFloatComplex(2.5, 3.5);
  res[0] = ((a.x == 2.5) && (a.y == 3.5))? 1 : 0;
  res[1] = (hipCimagf(a) == 3.5)? 1 : 0;
  res[2] = (hipCrealf(a) == 2.5)? 1 : 0;
  hipFloatComplex b = hipConjf(a);
  res[3] = ((b.x == 2.5) && (b.y == (-3.5)))? 1 : 0;
  float c = hipCsqabsf(a);
  res[4] = (c == 18.5)? 1 : 0;
  hipFloatComplex d = make_hipFloatComplex(4.5, 5.5);
  hipFloatComplex e = hipCaddf(a, d);
  res[5] = ((e.x == 7.0) && (e.y == 9.0))? 1 : 0;
  hipFloatComplex f = hipCsubf(d, a);
  res[6] = ((f.x == 2.0) && (f.y == 2.0))? 1 : 0;
  hipFloatComplex g = hipCmulf(a, d);
  res[7] = ((g.x == -8.0) && (g.y == 29.5))? 1 : 0;
  hipFloatComplex h = hipCdivf(a, d);
  res[8] = ((fabs(h.x - 0.60396) < 0.000001f) && (fabs(h.y - 0.039604) < 0.000001f))? 1 : 0;
  float i = hipCabsf(a);
  res[9] = (fabs(i - 4.30116) < 0.00001f)? 1 : 0;
  hipDoubleComplex j = make_hipDoubleComplex(10.5, 20.5);
  res[10] = ((j.x == 10.5) && (j.y == 20.5))? 1 : 0;
  res[11] = (hipCreal(j) == 10.5)? 1 : 0;
  res[12] = (hipCimag(j) == 20.5)? 1 : 0;
  hipDoubleComplex k = hipConj(j);
  res[13] = ((k.x == 10.5) && (k.y == (-20.5)))? 1 : 0;
  double l = hipCsqabs(j);
  res[14] = (((j.x * j.x) + (j.y * j.y)) == l)? 1 : 0;
  hipDoubleComplex m = make_hipDoubleComplex(4.5, 5.5);
  hipDoubleComplex n = hipCadd(j, m);
  res[15] = ((n.x == 15.0) && (n.y == 26.0))? 1 : 0;
  hipDoubleComplex o = hipCsub(j, m);
  res[16] = ((o.x == 6.0) && (o.y == 15.0))? 1 : 0;
  hipDoubleComplex p = hipCmul(j, m);
  res[17] = (p.x == -65.5) && (p.y == 150);
  hipComplex q = make_hipComplex(6.5, 7.5);
  res[18] = ((q.x == 6.5) && (q.y == 7.5))? 1 : 0;
  hipFloatComplex r = hipComplexDoubleToFloat(j);
  res[19] = ((r.x == 10.5) && (r.y == 20.5))? 1 : 0;
  hipDoubleComplex s = hipComplexFloatToDouble(a);
  res[20] = ((s.x == 2.5) && (s.y == 3.5))? 1 : 0;
  hipFloatComplex t = make_hipFloatComplex(1.5, 4.5);
  hipComplex u = hipCfmaf(a, d, t);
  res[21] = ((u.x == -6.5) && (u.y == 34.0))? 1 : 0;
  hipDoubleComplex v = make_hipDoubleComplex(2.5, 3.5);
  hipDoubleComplex w = hipCfma(j, m, v);
  res[22] = ((w.x == -63.0) && (w.y == 153.5))? 1 : 0;
  hipDoubleComplex x = hipCdiv(j, m);
  res[23] = ((fabs(x.x - 3.16832) < 0.00001f) && (fabs(x.y - 0.683168) < 0.00001f))? 1 : 0;
  double y = hipCabs(j);
  res[24] = (fabs(y - 23.0325) < 0.0001f)? 1 : 0;
}
)"};

static constexpr auto hip_complex_corner_float_string{
    R"(
extern "C"
__global__ void hip_complex_corner_float_kernel(int *res) {
   hipFloatComplex a = make_hipFloatComplex(0, 0);
   hipFloatComplex b = make_hipFloatComplex(4.5, 7.5);
   // 0 devide by complex number
   hipFloatComplex c = hipCdivf(a, b);
   // complex number devide by 0
   hipFloatComplex d = hipCdivf(b, a);
   res[0] = ((c.x == 0) && (c.y == 0))? 1 : 0;
   res[1] = (isnan(d.x) && isnan(d.y))? 1 : 0;
   hipFloatComplex e = make_hipFloatComplex(0, 1.5);
   hipFloatComplex f = hipCdivf(b, e);
   res[2] = ((f.x == 5.0) && (f.y == -3.0))? 1 : 0;
   hipFloatComplex g = make_hipFloatComplex(1.5, 0);
   hipFloatComplex h = hipCdivf(b, g);
   res[3] = ((h.x == 3.0) && (h.y == 5.0))? 1 : 0;
   hipFloatComplex i = hipCmulf(a, b);
   res[4] = ((i.x == 0.0) && (i.y == 0.0))? 1 : 0;
   hipFloatComplex j = make_hipFloatComplex(-4.5, -7.5);
   float k = hipCabsf(j);
   res[5] = (fabs(k - 8.7464) < 0.001)? 1 : 0;
}
)"};

static constexpr auto hip_complex_corner_double_string{
    R"(
extern "C"
__global__ void hip_complex_corner_double_kernel(int *res) {
   hipDoubleComplex a = make_hipDoubleComplex(0, 0);
   hipDoubleComplex b = make_hipDoubleComplex(4.5, 7.5);
   // 0 devide by complex number
   hipDoubleComplex c = hipCdiv(a, b);
   // complex number devide by 0
   hipDoubleComplex d = hipCdiv(b, a);
   res[0] = ((c.x == 0) && (c.y == 0))? 1 : 0;
   res[1] = (isnan(d.x) && isnan(d.y))? 1 : 0;
   hipDoubleComplex e = make_hipDoubleComplex(0, 1.5);
   hipDoubleComplex f = hipCdiv(b, e);
   res[2] = ((f.x == 5.0) && (f.y == -3.0))? 1 : 0;
   hipDoubleComplex g = make_hipDoubleComplex(1.5, 0);
   hipDoubleComplex h = hipCdiv(b, g);
   res[3] = ((h.x == 3.0) && (h.y == 5.0))? 1 : 0;
   hipDoubleComplex i = hipCmul(a, b);
   res[4] = ((i.x == 0.0) && (i.y == 0.0))? 1 : 0;
   hipDoubleComplex j = make_hipDoubleComplex(-4.5, -7.5);
   float k = hipCabs(j);
   res[5] = (fabs(k - 8.7464) < 0.001)? 1 : 0;
}
)"};

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hiprtcCompileProgram
 *  - To test working of "hip/hip_complex.h"  header inside kernel string
 * Test source
 * ------------------------
 *  - unit/rtc/hipRtcComplexHeader.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_Rtc_HipComplex_header") {
  int n = 0;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string architecture = prop.gcnArchName;
  std::string complete_CO = "--gpu-architecture=" + architecture;
  const char* compiler_option = complete_CO.c_str();
  hiprtcProgram prog = nullptr;
  const char** compiler_options = new const char*[2];
  compiler_options[0] = compiler_option;
  compiler_options[1] = "-includehip/hip_complex.h";
  std::string kernel_name = "";
  SECTION("Test Case covering all Complex API's") {
    n = 25;
    kernel_name = "hip_complex_basic_kernel";
    HIPRTC_CHECK(
        hiprtcCreateProgram(&prog, hip_complex_basic_string, kernel_name.c_str(), 0, NULL, NULL));
  }
  SECTION("Corner cases for Float type") {
    n = 6;
    kernel_name = "hip_complex_corner_float_kernel";
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, hip_complex_corner_float_string, kernel_name.c_str(), 0,
                                     NULL, NULL));
  }
  SECTION("Corner cases for Double type") {
    n = 6;
    kernel_name = "hip_complex_corner_double_kernel";
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, hip_complex_corner_double_string, kernel_name.c_str(),
                                     0, NULL, NULL));
  }
  unsigned int* result_h;
  unsigned int* result_d;
  unsigned int Nbytes = n * sizeof(unsigned int);
  result_h = new unsigned int[n];
  for (int i = 0; i < n; i++) {
    result_h[i] = 0;
  }
  HIP_CHECK(hipMalloc(&result_d, Nbytes));
  HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));
  const char* kername = kernel_name.c_str();
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
    if (result_h[i] != 1) {
      WARN("FAIL for " << i << " iteration");
      WARN(result_h[i]);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipModuleUnload(module));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  HIP_CHECK(hipFree(result_d));
  delete[] compiler_options;
  delete[] result_h;
}

/**
 * End doxygen group hiprtcHeadersTest.
 * @}
 */
