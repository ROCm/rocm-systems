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

/**
 * @addtogroup hiprtc_MathConstants_HeaderTst hiprtc_MathConstants_HeaderTst
 * @{
 * @ingroup hiprtcHeaders
 * `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,
 *                                  int numOptions,
 *                                  const char** options);` -
 * These test cases are target including various header file in kernel
 * string and compile using the api mentioned above.
 */

#include <hip_test_common.hh>
#include <hip/hiprtc.h>

static constexpr auto mathConstants_string{
    R"(
extern "C"
__global__ void mathConstants(float *res) {
  // single precision constants
  res[0] = (HIP_INF_F == __int_as_float(0x7f800000U));
  res[1] = (HIP_REMQUO_MASK_F == (~((~0U)<<HIP_REMQUO_BITS_F)));
  res[2] = (HIP_MIN_DENORM_F == __int_as_float(0x00000001U));
  res[3] = (HIP_MAX_NORMAL_F == __int_as_float(0x7f7fffffU));
  res[4] = (HIP_NEG_ZERO_F == __int_as_float(0x80000000U));
  res[5] = (HIP_ZERO_F == 0.0f);
  res[6] = (HIP_ZERO_F == 0.0f);
  res[7] = (HIP_ONE_F == 1.0f);
  res[8] = (HIP_SQRT_HALF_F == 0.707106781f);
  res[9] = (HIP_SQRT_HALF_HI_F == 0.707106781f);
  res[10] = (HIP_SQRT_HALF_LO_F == 1.210161749e-08F);
  res[11] = (HIP_SQRT_TWO_F == 1.414213562F);
  res[12] = (HIP_THIRD_F == 0.333333333F);
  res[13] = (HIP_PIO4_F == 0.785398163F);
  res[14] = (HIP_PIO2_F == 1.570796327F);
  res[15] = (HIP_3PIO4_F == 2.356194490F);
  res[16] = (HIP_2_OVER_PI_F == 0.636619772F);
  res[17] = (HIP_SQRT_2_OVER_PI_F == 0.797884561F);
  res[18] = (HIP_PI_F == 3.141592654F);
  res[19] = (HIP_L2E_F == 1.442695041F);
  res[20] = (HIP_L2T_F == 3.321928094F);
  res[21] = (HIP_LG2_F == 0.301029996F);
  res[22] = (HIP_LGE_F == 0.434294482F);
  res[23] = (HIP_LN2_F == 0.693147181F);
  res[24] = (HIP_LNT_F == 2.302585093F);
  res[25] = (HIP_LNPI_F == 1.144729886F);
  res[26] = (HIP_TWO_TO_M126_F == 1.175494351e-38F);
  res[27] = (HIP_NORM_HUGE_F == 3.402823466e38F);
  res[28] = (HIP_TWO_TO_23_F == 8388608.0F);
  res[29] = (HIP_TWO_TO_24_F == 16777216.0F);
  res[30] = (HIP_TWO_TO_31_F == 2147483648.0F);
  res[31] = (HIP_TWO_TO_32_F == 4294967296.0F);
  res[32] = (HIP_REMQUO_BITS_F == 3U);
  res[33] = (HIP_TRIG_PLOSS_F == 105615.0F);

  // double precision constants
  res[34] = (HIP_DBL2INT_CVT == 6755399441055744.0L);
  res[35] = (HIP_INF == __longlong_as_double(0x7ff0000000000000ULL));
  res[36] = (HIP_NEG_ZERO == __longlong_as_double(0x8000000000000000ULL));
  res[37] = (HIP_MIN_DENORM == __longlong_as_double(0x0000000000000001ULL));
  res[38] = (HIP_ZERO == 0.0f);
  res[39] = (HIP_ONE == 1.0f);
  res[40] = (HIP_SQRT_TWO == 1.4142135623730951e+0L);
  res[41] = (HIP_SQRT_HALF == 7.0710678118654757e-1L);
  res[42] = (HIP_SQRT_HALF_HI == 7.0710678118654757e-1L);
  res[43] = (HIP_SQRT_HALF_LO == (-4.8336466567264567e-17L));
  res[44] = (HIP_THIRD == 3.3333333333333333e-1L);
  res[45] = (HIP_TWOTHIRD == 6.6666666666666667e-1L);
  res[46] = (HIP_PIO4 == 7.8539816339744828e-1L);
  res[47] = (HIP_PIO4_HI == 7.8539816339744828e-1L);
  res[48] = (HIP_PIO4_LO == 3.0616169978683830e-17L);
  res[49] = (HIP_PIO2 == 1.5707963267948966e+0L);
  res[50] = (HIP_PIO2_HI == 1.5707963267948966e+0L);
  res[51] = (HIP_PIO2_LO == 6.1232339957367660e-17L);
  res[52] = (HIP_3PIO4 == 2.3561944901923448e+0L);
  res[53] = (HIP_2_OVER_PI == 6.3661977236758138e-1L);
  res[54] = (HIP_PI == 3.1415926535897931e+0L);
  res[55] = (HIP_PI_HI == 3.1415926535897931e+0L);
  res[56] = (HIP_PI_LO == 1.2246467991473532e-16L);
  res[57] = (HIP_SQRT_2PI == 2.5066282746310007e+0L);
  res[58] = (HIP_SQRT_2PI_HI == 2.5066282746310007e+0L);
  res[59] = (HIP_SQRT_2PI_LO == (-1.8328579980459167e-16L));
  res[60] = (HIP_SQRT_PIO2 == 1.2533141373155003e+0L);
  res[61] = (HIP_SQRT_PIO2_HI == 1.2533141373155003e+0L);
  res[62] = (HIP_SQRT_PIO2_LO == (-9.1642899902295834e-17L));
  res[63] = (HIP_SQRT_2OPI == 7.9788456080286536e-1L);
  res[64] = (HIP_L2E == 1.4426950408889634e+0L);
  res[65] = (HIP_L2E_HI == 1.4426950408889634e+0L);
  res[66] = (HIP_L2E_LO == 2.0355273740931033e-17L);
  res[67] = (HIP_L2T == 3.3219280948873622e+0L);
  res[68] = (HIP_LG2 == 3.0102999566398120e-1L);
  res[69] = (HIP_LG2_HI == 3.0102999566398120e-1L);
  res[70] = (HIP_LG2_LO == (-2.8037281277851704e-18L));
  res[71] = (HIP_LGE == 4.3429448190325182e-1L);
  res[72] = (HIP_LGE_HI == 4.3429448190325182e-1L);
  res[73] = (HIP_LGE_LO == 1.09831965021676510e-17L);
  res[74] = (HIP_LN2 == 6.9314718055994529e-1L);
  res[75] = (HIP_LN2_HI == 6.9314718055994529e-1L);
  res[76] = (HIP_LN2_LO == 2.3190468138462996e-17L);
  res[77] = (HIP_LNT == 2.3025850929940459e+0L);
  res[78] = (HIP_LNT_HI == 2.3025850929940459e+0L);
  res[79] = (HIP_LNT_LO == (-2.1707562233822494e-16L));
  res[80] = (HIP_LNPI == 1.1447298858494002e+0L);
  res[81] = (HIP_LN2_X_1024 == 7.0978271289338397e+2L);
  res[82] = (HIP_LN2_X_1025 == 7.1047586007394398e+2L);
  res[83] = (HIP_LN2_X_1075 == 7.4513321910194122e+2L);
  res[84] = (HIP_LG2_X_1024 == 3.0825471555991675e+2L);
  res[85] = (HIP_LG2_X_1075 == 3.2360724533877976e+2);
  res[86] = (HIP_TWO_TO_23 == 8388608.0F);
  res[87] = (HIP_TWO_TO_52 == 4503599627370496.0L);
  res[88] = (HIP_TWO_TO_53 == 9007199254740992.0L);
  res[89] = (HIP_TWO_TO_54 == 18014398509481984.0L);
  res[90] = (HIP_TWO_TO_M54 == 5.5511151231257827e-17L);
  res[91] = (HIP_TWO_TO_M1022 == 2.22507385850720140e-308L);
  res[92] = (HIP_TRIG_PLOSS == 2147483648.0L);
}
)"};
/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hiprtcCompileProgram
 *    1) To test working of "hip/hip_math_constants.h"  header inside kernel string
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtc_MathConstants_HeaderTst.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_Rtc_MathConstants_header") {
  std::string kernel_name = "mathConstants";
  const char* kername = kernel_name.c_str();
  float* result_h;
  float* result_d;
  int n = 93;
  float Nbytes = n * sizeof(float);
  result_h = new float[n];
  for (int i = 0; i < n; i++) {
    result_h[i] = 0;
  }
  HIP_CHECK(hipMalloc(&result_d, Nbytes));
  HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string architecture = prop.gcnArchName;
  std::string complete_CO = "--gpu-architecture=" + architecture;
  const char* compiler_option = complete_CO.c_str();
  for (int scenario = 0; scenario < 2; scenario++) {
    hiprtcProgram prog;
    std::array<const char*, 2> compiler_options = {compiler_option, ""};
    HIPRTC_CHECK(hiprtcCreateProgram(&prog, mathConstants_string, kername, 0, NULL, NULL));
    if (scenario == 0) {
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
    } else {
      hiprtcResult compileResult{hiprtcCompileProgram(prog, 2, compiler_options.data())};
      if (!(compileResult == HIPRTC_SUCCESS)) {
        WARN("hiprtcCompileProgram() api failed!!");
        size_t logSize;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
        std::string log(logSize, '\0');
        HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
        WARN(log);
        REQUIRE(false);
      }
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
      if (result_h[i] != 1.0f) {
        WARN("FAIL for " << i << " iteration");
        WARN(result_h[i]);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipModuleUnload(module));
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  }
  HIP_CHECK(hipFree(result_d));
  delete[] result_h;
  REQUIRE(true);
}

/**
 * End doxygen group hiprtcHeaders.
 * @}
 */
