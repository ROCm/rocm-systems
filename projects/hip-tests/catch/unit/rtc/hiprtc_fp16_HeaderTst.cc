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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hiprtc_fp16_HeaderTst hiprtc_fp16_HeaderTst
 * @{
 * @ingroup hiprtcHeaders
 * `hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions,
 *                                    const char** options);` -
 * These test cases are target including various header file in kernel
 * string and compile using the api mentioned above.
 */

#include <hip_test_common.hh>
#include <hip/hiprtc.h>

static constexpr auto fp16_string{
    R"(
extern "C"
__global__ void fp16(float *res) {

  __half x = 10, y = 2, z = 0, a = 3.8;
  a += 2;     res[0] = __heq(a, 5.8);
  a -= 2;     res[1] = __heq(a, 3.8);
  a *= 2;     res[2] = __heq(a, 7.6);
  a /= 2;     res[3] = __heq(a, 3.8);
  a++;        res[4] = __heq(a, 4.8);
  a--;        res[5] = __heq(a, 3.8);
  ++a;        res[6] = __heq(a, 4.8);
  --a;        res[7] = __heq(a, 3.8);
  z = x + __half(1.1);    res[8] = __heq(z, 11.1);
  z = x - __half(1.1);    res[9] = __heq(z, 8.9);
  z = x + y;           res[10] = __heq(z, 12);
  z = x - y;           res[11] = __heq(z, 8);
  z = x * y;           res[12] = __heq(z, 20);
  z = x / y;           res[13] = __heq(z, 5);
  x = 2.2; y = 2.22;   res[14] = __heq((x == y), 0);
  x = 2.2; y = 2.2;    res[15] = __heq((x != y), 0);
  x = 2.2; y = 2.22;   res[16] = __heq((x < y), 1);
  x = 2.2; y = 2.3;    res[17] = __heq((x > y), 0);
  x = 2.2; y = 2.3;    res[18] = __heq((x <= y), 1);
  x = 2.2; y = 2.2;    res[19] = __heq((x >= y), 1);

  __half2 d = __half2{5, 8}, e = __half2{2, 10}, f = 0, g = __half2{3, 4};
  g += 2;     res[20] = __heq((__half2{5, 6} == g), 1);
  g -= 2;     res[21] = __heq((__half2{3, 4} == g), 1);
  g *= 2;     res[22] = __heq((__half2{6, 8} == g), 1);
  g /= 2;     res[23] = __heq((__half2{3, 4} == g), 1);
  g++;        res[24] = __heq((__half2{4, 5} == g), 1);
  g--;        res[25] = __heq((__half2{3, 4} == g), 1);
  ++g;        res[26] = __heq((__half2{4, 5} == g), 1);
  --g;        res[27] = __heq((__half2{3, 4} == g), 1);
  f = d + __half2{1, 1};    res[28] = __heq((__half2{6, 9} == f), 1);
  f = d - __half2{1, 1};    res[29] = __heq((__half2{4, 7} == f), 1);
  f = d + e;     res[30] = __heq((__half2{7, 18} == f), 1);
  f = d - e;     res[31] = __heq((__half2{3, -2} == f), 1);
  f = d * e;     res[32] = __heq((__half2{10, 80} == f), 1);
  f = d / e;     res[33] = __heq((__half2{2.5, 0.8} == f), 1);
  d = __half2{5, 8};   e = __half2{5.1, 7.9};  res[34] = __heq((d == e), 0);
  d = __half2{5, 8};   e = __half2{5.1, 8};    res[35] = __heq((d != e), 1);
  d = __half2{4, 7};   e = __half2{5, 8};      res[36] = __heq((d < e), 1);
  d = __half2{3, 8};   e = __half2{2, 8};      res[37] = __heq((d > e), 0);
  d = __half2{2, 8};   e = __half2{2, 10};     res[38] = __heq((d <= e), 1);
  d = __half2{5, 8};   e = __half2{2, 10};     res[39] = __heq((d >= e), 1);

  res[40] = __heq((make_half2(2, 2) == __half2{2, 2}), 1);
  res[41] = __heq(__low2half(make_half2(2, 2)), 2);
  res[42] = __heq(__high2half(make_half2(3, 3)), 3);
  res[43] = __heq((__half2half2(3) == __half2{3, 3}), 1);
  res[44] = __heq((__halves2half2(3, 4) == __half2{3, 4}), 1);
  res[45] = __heq((__low2half2(__half2{2, 3}) == __half2{2, 2}), 1);
  res[46] = __heq((__high2half2(__half2{4, 5}) == __half2{5, 5}), 1);
  res[47] = __heq((__lows2half2(__half2{2, 4}, __half2{3, 5}) == __half2{2, 3}), 1);
  res[48] = __heq((__highs2half2(__half2{2, 4}, __half2{3, 5}) == __half2{4, 5}), 1);
  res[49] = __heq((__lowhigh2highlow(__half2{2, 3}) == __half2{3, 2}), 1);

  res[50] = __heq(__half_as_short(3), 16896);
  res[51] = __heq(__half_as_ushort(11), 18816);
  res[52] = __heq(__short_as_half(16896), 3);
  res[53] = __heq(__ushort_as_half(18816), 11);

  res[54] = __heq(__float2half(3.1234), 3.123047);
  res[55] = __heq(__float2half_rn(3.1234), 3.123047);
  res[56] = __heq(__float2half_rz(3.1234), 3.123047);
  res[57] = __heq(__float2half_rd(3.1234), 3.123047);
  res[58] = __heq(__float2half_ru(3.1234), 3.125);
  res[59] = __heq((__float2half2_rn(3) == __half2{3, 3}), 1);
  res[60] = __heq((__floats2half2_rn(3, 2) == __half2{3, 2}), 1);
  res[61] = __heq((__float22half2_rn(make_float2(3, 4)) == __half2{3, 4}), 1);
  res[62] = __heq(__half2float(3.3), 3.3);
  res[63] = __heq(__low2float(__half2{3, 4}), 3);
  res[64] = __heq(__high2float(__half2{3, 4}), 4);
  res[65] = __heq((__half22float2(__half2{3, 4}) == make_float2(3, 4)), 1);
  res[66] = __heq(__half2int_rn(1.1234), 1);
  res[67] = __heq(__half2int_rz(1.1234), 1);
  res[68] = __heq(__half2int_rd(1.1234), 1);
  res[69] = __heq(__half2int_ru(1.1234), 1);
  res[70] = __heq(__int2half_rn(2), 2);
  res[71] = __heq(__int2half_rz(2), 2);
  res[72] = __heq(__int2half_rd(2), 2);
  res[73] = __heq(__int2half_ru(2), 2);
  res[74] = __heq(__half2short_rn(1.1234), 1);
  res[75] = __heq(__half2short_rz(1.1234), 1);
  res[76] = __heq(__half2short_rd(1.1234), 1);
  res[77] = __heq(__half2short_ru(1.1234), 1);
  res[78] = __heq(__short2half_rn(2), 2);
  res[79] = __heq(__short2half_rz(2), 2);
  res[80] = __heq(__short2half_rd(2), 2);
  res[81] = __heq(__short2half_ru(2), 2);
  res[82] = __heq(__half2ll_rn(1.1234), 1);
  res[83] = __heq(__half2ll_rz(1.1234), 1);
  res[84] = __heq(__half2ll_rd(1.1234), 1);
  res[85] = __heq(__half2ll_ru(1.1234), 1);
  res[86] = __heq(__ll2half_rn(2), 2);
  res[87] = __heq(__ll2half_rz(2), 2);
  res[88] = __heq(__ll2half_rd(2), 2);
  res[89] = __heq(__ll2half_ru(2), 2);
  res[90] = __heq(__half2uint_rn(1.1234), 1);
  res[91] = __heq(__half2uint_rz(1.1234), 1);
  res[92] = __heq(__half2uint_rd(1.1234), 1);
  res[93] = __heq(__half2uint_ru(1.1234), 1);
  res[94] = __heq(__uint2half_rn(2), 2);
  res[95] = __heq(__uint2half_rz(2), 2);
  res[96] = __heq(__uint2half_rd(2), 2);
  res[97] = __heq(__uint2half_ru(2), 2);
  res[98] = __heq(__half2ushort_rn(1.1234), 1);
  res[99] = __heq(__half2ushort_rz(1.1234), 1);
  res[100] = __heq(__half2ushort_rd(1.1234), 1);
  res[101] = __heq(__half2ushort_ru(1.1234), 1);
  res[102] = __heq(__ushort2half_rn(2), 2);
  res[103] = __heq(__ushort2half_rz(2), 2);
  res[104] = __heq(__ushort2half_rd(2), 2);
  res[105] = __heq(__ushort2half_ru(2), 2);
  res[106] = __heq(__half2ull_rn(1.1234), 1);
  res[107] = __heq(__half2ull_rz(1.1234), 1);
  res[108] = __heq(__half2ull_rd(1.1234), 1);
  res[109] = __heq(__half2ull_ru(1.1234), 1);
  res[110] = __heq(__ull2half_rn(2), 2);
  res[111] = __heq(__ull2half_rz(2), 2);
  res[112] = __heq(__ull2half_rd(2), 2);
  res[113] = __heq(__ull2half_ru(2), 2);

  __half b = a;
  res[114] = __heq(__ldg(&b), a);
  res[115] = __heq(__ldcg(&b), a);
  res[116] = __heq(__ldca(&b), a);
  res[117] = __heq(__ldcs(&b), a);

  __half2 m, n; m = n = __half2{5, 7};
  res[118] = __heq((__ldg(&m) == n), 1);
  res[119] = __heq((__ldcg(&m) == n), 1);
  res[120] = __heq((__ldca(&m) == n), 1);
  res[121] = __heq((__ldcs(&m) == n), 1);

  a = 2.22; b = 2.22;    res[122] = (__heq(a, b) == 1);
  a = 2.2;  b = 2.22;    res[123] = __heq(__hne(a, b), 1);
  a = 2.2;  b = 2.201;   res[124] = __heq(__hle(a, b), 1);
  a = 2.21; b = 2.201;   res[125] = __heq(__hge(a, b), 1);
  a = 2.2;  b = 2.201;   res[126] = __heq(__hlt(a, b), 1);
  a = 2.2;  b = 2.21;    res[127] = __heq(__hgt(a, b), 0);
  a = 2.21; b = 2.201;   res[128] = __heq(__hequ(a, b), 0);
  a = 2.201; b = 2.201;  res[129] = __heq(__hneu(a, b), 0);
  a = 2.201; b = 2.21;   res[130] = __heq(__hleu(a, b), 1);
  a = 2.21;  b = 2.22;   res[131] = __heq(__hgeu(a, b), 0);
  a = 2.201; b = 2.201;  res[132] = __heq(__hltu(a, b), 0);
  a = 2.21;  b = 2.201;  res[133] = __heq(__hgtu(a, b), 1);

  res[134] = __heq((__heq2(__half2{1, 2}, __half2{1, 2}) == __half2{1, 1}), 1);
  res[135] = __heq((__hne2(__half2{1, 2}, __half2{2, 2}) == __half2{1, 0}), 1);
  res[136] = __heq((__hle2(__half2{1, 2}, __half2{1, 1}) == __half2{1, 0}), 1);
  res[137] = __heq((__hge2(__half2{1, 2}, __half2{1, 1}) == __half2{1, 1}), 1);
  res[138] = __heq((__hlt2(__half2{1, 2}, __half2{2, 2}) == __half2{1, 0}), 1);
  res[139] = __heq((__hgt2(__half2{1, 2}, __half2{1, 1}) == __half2{0, 1}), 1);
  res[140] = __heq((__hequ2(__half2{2, 3}, __half2{2, 2}) == __half2{1, 0}), 1);
  res[141] = __heq((__hneu2(__half2{1, 2}, __half2{1, 4}) == __half2{0, 1}), 1);
  res[142] = __heq((__hleu2(__half2{2, 3}, __half2{2, 2}) == __half2{1, 0}), 1);
  res[143] = __heq((__hgeu2(__half2{2, 3}, __half2{3, 2}) == __half2{0, 1}), 1);
  res[144] = __heq((__hltu2(__half2{2, 3}, __half2{3, 2}) == __half2{1, 0}), 1);
  res[145] = __heq((__hgtu2(__half2{2, 3}, __half2{3, 2}) == __half2{0, 1}), 1);
  res[146] = __heq(__hbeq2(__half2{3, 3}, __half2{3, -3}), 0);
  res[147] = __heq(__hbne2(__half2{4, 3}, __half2{3, 3}), 0);
  res[148] = __heq(__hble2(__half2{3, 3}, __half2{3, 4}), 1);
  res[149] = __heq(__hbge2(__half2{2, 4}, __half2{3, 3}), 0);
  res[150] = __heq(__hblt2(__half2{2, 4}, __half2{2, 4}), 0);
  res[151] = __heq(__hbgt2(__half2{2, 4}, __half2{2, 3}), 0);
  res[152] = __heq(__hbequ2(__half2{3, 2}, __half2{3, 3}), 0);
  res[153] = __heq(__hbneu2(__half2{4, 3}, __half2{4, 2}), 0);
  res[154] = __heq(__hbleu2(__half2{2, 3}, __half2{2, 3}), 1);
  res[155] = __heq(__hbgeu2(__half2{3, 3}, __half2{2, 3}), 1);
  res[156] = __heq(__hbltu2(__half2{2, 3}, __half2{2, 2}), 0);
  res[157] = __heq(__hbgtu2(__half2{3, 3}, __half2{2, 3}), 0);

  a = 2.2;  b = 2.22;    res[158] = __heq(__hmax(a, b), b);
  a = 2.2;  b = 2.202;   res[159] = __heq(__hmax_nan(a, b), b);
  a = 2.2;  b = 2.22;    res[160] = __heq(__hmin(a, b), a);
  a = 2.2;  b = 2.222;   res[161] = __heq(__hmin_nan(a, b), a);
  a = 2.2;               res[162] = __heq(__clamp_01(a), 1);
  a = -2.2;              res[163] = __heq(__clamp_01(a), 0);
  a = 0.2;               res[164] = __heq(__clamp_01(a), a);
  a = -2.2;              res[165] = __heq(__habs(a), 2.2);
  a = 2.2;  b = 2.22;    res[166] = __heq(__hadd(a, b), 4.42);
  a = 2.2;  b = -2.2;    res[167] = __heq(__hsub(a, b), 4.4);
  a = 2.2;  b = 2;       res[168] = __heq(__hmul(a, b), 4.4);
  a = 2.2;  b = 2.22;    res[169] = __heq(__hadd_sat(a, b), 1);
  a = 2.2;  b = 2.22;    res[170] = __heq(__hsub_sat(a, b), 0);
  a = 2.2;  b = 2;       res[171] = __heq(__hmul_sat(a, b), 1);
  a = 2.2;  b = 1;       res[172] = __heq(__hfma(a, b, a), 4.4);
  a = 2.2;  b = -1;      res[173] = __heq(__hfma_sat(a, b, a), 0);
  a = 4.2;  b = 2;       res[174] = __heq(__hdiv(a, b), 2.1);

  res[175] = __heq(__hbeq2(__habs2(__half2{-1, -4}), __half2{1, 4}), 1);
  res[176] = __heq(__hbeq2(__hadd2(__half2{1, 4}, __half2{2, -5}), __half2{3, -1}), 1);
  res[177] = __heq(__hbeq2(__hsub2(__half2{1, 4}, __half2{2, -2}), __half2{-1, 6}), 1);
  res[178] = __heq(__hbeq2(__hmul2(__half2{1, 3}, __half2{5, -2}), __half2{5, -6}), 1);
  res[179] = __heq(__hbeq2(__hadd2_sat(__half2{1, 3}, __half2{2, -5}), __half2{1, 0}), 1);
  res[180] = __heq(__hbeq2(__hsub2_sat(__half2{2, 3}, __half2{2, -2}), __half2{0, 1}), 1);
  res[181] = __heq(__hbeq2(__hmul2_sat(__half2{1, 3}, __half2{5, -2}), __half2{1, 0}), 1);
  res[182] = __heq(__hbeq2(__hfma2(__half2{1, 3}, __half2{5, -2}, __half2{-5, 8}), __half2{0, 2}), 1);
  res[183] = __heq(__hbeq2(__hfma2_sat(__half2{1, 3}, __half2{5, -2}, __half2{-5, 8}), __half2{0, 1}), 1);
  res[184] = __heq(__hbeq2(__h2div(__half2{1, 3}, __half2{5, -2}), __half2{0.2, -1.5}), 1);
  res[185] = __heq(amd_mixed_dot(__half2{1, 3}, __half2{3, 3}, 2, 1), 14);

  res[186] = __heq(htrunc(2.8), 2);
  res[187] = __heq(hceil(2.8), 3);
  res[188] = __heq(hfloor(2.8), 2);
  res[189] = __heq(hrint(2.8), 3);
  res[190] = __heq(hsin(0), 0);
  res[191] = __heq(hcos(0), 1);
  res[192] = __heq(hexp(2), 7.390625);
  res[193] = __heq(hexp2(2), 4);
  res[194] = __heq(hexp10(2), 100);
  res[195] = __heq(hlog(7.390625), 2);
  res[196] = __heq(hlog2(4), 2);
  res[197] = __heq(hlog10(100), 2);
  res[198] = __heq(hrcp(4), 0.25);
  res[199] = __heq(hrsqrt(0.25), 2);
  res[200] = __heq(hsqrt(1.21), 1.1);
  res[201] = __heq(__hisinf(1), 0);
  res[202] = __heq(__hisnan(1), 0);
  res[203] = __heq(__hneg(1.25), -1.25);

  res[204] = __heq(__hbeq2(h2trunc(__half2{3.4, 5.2}), __half2{3, 5}), 1);
  res[205] = __heq(__hbeq2(h2ceil(__half2{3.4, 5.2}), __half2{4, 6}), 1);
  res[206] = __heq(__hbeq2(h2floor(__half2{3.4, 5.2}), __half2{3, 5}), 1);
  res[207] = __heq(__hbeq2(h2rint(__half2{3.4, 5.2}), __half2{3, 5}), 1);
  res[208] = __heq(__hbeq2(h2sin(__half2{0, 0}), __half2{0, 0}), 1);
  res[209] = __heq(__hbeq2(h2cos(__half2{0, 0}), __half2{1, 1}), 1);
  res[210] = __heq(__hbeq2(h2exp(__half2{2, 0}), __half2{7.390625, 1}), 1);
  res[211] = __heq(__hbeq2(h2exp2(__half2{3, 2}), __half2{8, 4}), 1);
  res[212] = __heq(__hbeq2(h2exp10(__half2{2, 3}), __half2{100, 1000}), 1);
  res[213] = __heq(__hbeq2(h2log(__half2{2.718750, 1}), __half2{1, 0}), 1);
  res[214] = __heq(__hbeq2(h2log2(__half2{8, 16}), __half2{3, 4}), 1);
  res[215] = __heq(__hbeq2(h2log10(__half2{1000, 100}), __half2{3, 2}), 1);
  res[216] = __heq(__hbeq2(h2rcp(__half2{4, 5}), __half2{0.25, 0.2}), 1);
  res[217] = __heq(__hbeq2(h2rsqrt(__half2{100, 25}), __half2{0.1, 0.2}), 1);
  res[218] = __heq(__hbeq2(h2sqrt(__half2{100, 25}), __half2{10, 5}), 1);
  res[219] = __heq(__hbeq2(__hisinf2(__half2{100, 0}), __half2{0, 0}), 1);
  res[220] = __heq(__hbeq2(__hisnan2(__half2{100, 25}), __half2{0, 0}), 1);
  res[221] = __heq(__hbeq2(__hneg2(__half2{2.1, -25}), __half2{-2.1, 25}), 1);
}
)"};

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hiprtcCompileProgram
 *    1) To test list of apis in "hip/hip_fp16.h" header using kernel string
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtc_fp16_HeaderTst.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */

TEST_CASE("Unit_Rtc_fp16_header") {
  std::string kernel_name = "fp16";
  const char* kername = kernel_name.c_str();
  float* result_h;
  float* result_d;
  int n = 222;
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
  hiprtcProgram prog;

  HIPRTC_CHECK(hiprtcCreateProgram(&prog, fp16_string, kername, 0, NULL, NULL));
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
  delete[] result_h;
  REQUIRE(true);
}

/**
 * End doxygen group hiprtcHeaders.
 * @}
 */
