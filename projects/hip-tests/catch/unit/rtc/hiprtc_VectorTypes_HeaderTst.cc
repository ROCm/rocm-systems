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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hiprtc_VectorTypes_HeaderTst hiprtc_VectorTypes_HeaderTst
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

static constexpr auto vectorTypes_string{
    R"(
extern "C"

#define EPSILON 0.000001

#define isEqualFloat1(f1, f11) (fabsf(fabsf(f1.x) - fabsf(f11.x)) < EPSILON)

#define isEqualFloat2(f2, f21) ((fabsf(fabsf(f2.x) - fabsf(f21.x)) < EPSILON) && \
                                (fabsf(fabsf(f2.y) - fabsf(f21.y)) < EPSILON))

#define isEqualFloat3(f3, f31) ((fabsf(fabsf(f3.x) - fabsf(f31.x)) < EPSILON) && \
                                (fabsf(fabsf(f3.y) - fabsf(f31.y)) < EPSILON) && \
                                (fabsf(fabsf(f3.z) - fabsf(f31.z)) < EPSILON))

#define isEqualFloat4(f4, f41) ((fabsf(fabsf(f4.x) - fabsf(f41.x)) < EPSILON) && \
                                (fabsf(fabsf(f4.y) - fabsf(f41.y)) < EPSILON) && \
                                (fabsf(fabsf(f4.z) - fabsf(f41.z)) < EPSILON) && \
                                (fabsf(fabsf(f4.w) - fabsf(f41.w)) < EPSILON))

#define isEqualDouble1(d1, d11) isEqualFloat1(d1, d11)
#define isEqualDouble2(d2, d21) isEqualFloat2(d2, d21)
#define isEqualDouble3(d3, d31) isEqualFloat3(d3, d31)
#define isEqualDouble4(d4, d41) isEqualFloat4(d4, d41)

__global__ void vectorTypes(int *res) {
  char1 ch1(0), ch11(0);   res[0] = (ch1 == ch11);
  ch1 += char1(2);         res[1] = (ch1 == 2);
  ch1 -= char1(1);         res[2] = (ch1 == 1);
  ch1 *= char1(2);         res[3] = (ch1 == 2);
  ch1 /= char1(2);         res[4] = (ch1 == 1);
  ch1 %= char1(2);         res[5] = (ch1 == 1);
  ch1 ^= char1(2);         res[6] = (ch1 == 3);
  ch1 |= char1(5);         res[7] = (ch1 == 7);
  ch1 &= char1(2);         res[8] = (ch1 == 2);
  ch1 <<= char1(3);        res[9] = (ch1 == 16);
  ch1 >>= char1(2);        res[10] = (ch1 == 4);
  ch11 = char1(4);         res[11] = (ch1 == ch11);
  ch11 = char1(3);         res[12] = (ch1 != ch11);
  ch1++;                   res[13] = (ch1 == 5);
  ++ch1;                   res[14] = (ch1 == 6);
  ch1--;                   res[15] = (ch1 == 5);
  --ch1;                   res[16] = (ch1 == 4);
  ch1 = ch11 + char1(2);   res[17] = (ch1 == 5);
  ch1 = ch11 - char1(2);   res[18] = (ch1 == 1);
  ch1 = ch11 * char1(2);   res[19] = (ch1 == 6);
  ch1 = ch11 / char1(2);   res[20] = (ch1 == 1);
  ch1 = ch11 % char1(2);   res[21] = (ch1 == 1);
  ch1 = ch11 ^ char1(1);   res[22] = (ch1 == 2);
  ch1 = ch11 | char1(6);   res[23] = (ch1 == 7);
  ch1 = ch11 & char1(6);   res[24] = (ch1 == 2);
  ch1 = ch11 << char1(3);  res[25] = (ch1 == 24);
  ch1 = ch11 >> char1(3);  res[26] = (ch1 == 0);
  ch11 = char1(3); ch1 = ~ch11;   res[27] = (ch1 == -4);

  char2 ch2(1, 2), ch21(1, 2);    res[28] = (ch2 == ch21);
  ch2 += char2(2, 3);         res[29] = (ch2 == char2(3, 5));
  ch2 -= char2(1, 2);         res[30] = (ch2 == char2(2, 3));
  ch2 *= char2(2, 4);         res[31] = (ch2 == char2(4, 12));
  ch2 /= char2(2, 2);         res[32] = (ch2 == char2(2, 6));
  ch2 %= char2(3, 4);         res[33] = (ch2 == char2(2, 2));
  ch2 ^= char2(1, 3);         res[34] = (ch2 == char2(3, 1));
  ch2 |= char2(4, 5);         res[35] = (ch2 == char2(7, 5));
  ch2 &= char2(3, 4);         res[36] = (ch2 == char2(3, 4));
  ch2 <<= char2(2, 3);        res[37] = (ch2 == char2(12, 32));
  ch2 >>= char2(2, 4);        res[38] = (ch2 == char2(3, 2));
  ch21 = char2(3, 2);         res[39] = (ch2 == ch21);
  ch21 = char2(3, 3);         res[40] = (ch2 != ch21);
  ch2++;                      res[41] = (ch2 == char2(4, 3));
  ++ch2;                      res[42] = (ch2 == char2(5, 4));
  ch2--;                      res[43] = (ch2 == char2(4, 3));
  --ch2;                      res[44] = (ch2 == char2(3, 2));
  ch2 = ch21 + char2(2, 3);   res[45] = (ch2 == char2(5, 6));
  ch2 = ch21 - char2(2, 1);   res[46] = (ch2 == char2(1, 2));
  ch2 = ch21 * char2(3, 2);   res[47] = (ch2 == char2(9, 6));
  ch2 = ch21 / char2(2, 3);   res[48] = (ch2 == char2(1, 1));
  ch2 = ch21 % char2(2, 1);   res[49] = (ch2 == char2(1, 0));
  ch2 = ch21 ^ char2(1, 2);   res[50] = (ch2 == char2(2, 1));
  ch2 = ch21 | char2(4, 5);   res[51] = (ch2 == char2(7, 7));
  ch2 = ch21 & char2(5, 4);   res[52] = (ch2 == char2(1, 0));
  ch2 = ch21 << char2(2, 3);  res[53] = (ch2 == char2(12, 24));
  ch2 = ch21 >> char2(1, 2);  res[54] = (ch2 == char2(1, 0));
  ch21 = char2(2, 3); ch2 = ~ch21;   res[55] = (ch2 == char2(-3, -4));

  char3 ch3(1, 2, 3), ch31(1, 2, 3);    res[56] = (ch3 == ch31);
  ch3 += char3(2, 3, 4);         res[57] = (ch3 == char3(3, 5, 7));
  ch3 -= char3(1, 2, 3);         res[58] = (ch3 == char3(2, 3, 4));
  ch3 *= char3(2, 4, -5);        res[59] = (ch3 == char3(4, 12, -20));
  ch3 /= char3(2, 2, -4);        res[60] = (ch3 == char3(2, 6, 5));
  ch3 %= char3(3, 4, 6);         res[61] = (ch3 == char3(2, 2, 5));
  ch3 ^= char3(1, 3, 4);         res[62] = (ch3 == char3(3, 1, 1));
  ch3 |= char3(4, 5, 3);         res[63] = (ch3 == char3(7, 5, 3));
  ch3 &= char3(3, 4, 7);         res[64] = (ch3 == char3(3, 4, 3));
  ch3 <<= char3(2, 3, 4);        res[65] = (ch3 == char3(12, 32, 48));
  ch3 >>= char3(2, 4, 3);        res[66] = (ch3 == char3(3, 2, 6));
  ch31 = char3(3, 2, 6);         res[67] = (ch3 == ch31);
  ch31 = char3(3, 3, 9);         res[68] = (ch3 != ch31);
  ch3++;                         res[69] = (ch3 == char3(4, 3, 7));
  ++ch3;                         res[70] = (ch3 == char3(5, 4, 8));
  ch3--;                         res[71] = (ch3 == char3(4, 3, 7));
  --ch3;                         res[72] = (ch3 == char3(3, 2, 6));
  ch3 = ch31 + char3(2, 3, -8);  res[73] = (ch3 == char3(5, 6, 1));
  ch3 = ch31 - char3(2, 1, -1);  res[74] = (ch3 == char3(1, 2, 10));
  ch3 = ch31 * char3(3, 2, -2);  res[75] = (ch3 == char3(9, 6, -18));
  ch3 = ch31 / char3(2, 3, -3);  res[76] = (ch3 == char3(1, 1, -3));
  ch3 = ch31 % char3(2, 1, 3);   res[77] = (ch3 == char3(1, 0, 0));
  ch3 = ch31 ^ char3(1, 2, 4);   res[78] = (ch3 == char3(2, 1, 13));
  ch3 = ch31 | char3(4, 5, 4);   res[79] = (ch3 == char3(7, 7, 13));
  ch3 = ch31 & char3(5, 4, 8);   res[80] = (ch3 == char3(1, 0, 8));
  ch3 = ch31 << char3(2, 3, 2);  res[81] = (ch3 == char3(12, 24, 36));
  ch3 = ch31 >> char3(1, 2, 3);  res[82] = (ch3 == char3(1, 0, 1));
  ch31 = char3(2, 3, -2); ch3 = ~ch31;   res[83] = (ch3 == char3(-3, -4, 1));

  char4 ch4(1, 2, 3, 4), ch41(1, 2, 3, 4);    res[84] = (ch4 == ch41);
  ch4 += char4(2, 3, 4, 5);         res[85] = (ch4 == char4(3, 5, 7, 9));
  ch4 -= char4(1, 2, 3, 4);         res[86] = (ch4 == char4(2, 3, 4, 5));
  ch4 *= char4(2, 4, -5, 3);        res[87] = (ch4 == char4(4, 12, -20, 15));
  ch4 /= char4(2, 2, -4, 4);        res[88] = (ch4 == char4(2, 6, 5, 3));
  ch4 %= char4(3, 4, 6, 3);         res[89] = (ch4 == char4(2, 2, 5, 0));
  ch4 ^= char4(1, 3, 4, 2);         res[90] = (ch4 == char4(3, 1, 1, 2));
  ch4 |= char4(4, 5, 3, 4);         res[91] = (ch4 == char4(7, 5, 3, 6));
  ch4 &= char4(3, 4, 7, 3);         res[92] = (ch4 == char4(3, 4, 3, 2));
  ch4 <<= char4(2, 3, 4, 5);        res[93] = (ch4 == char4(12, 32, 48, 64));
  ch4 >>= char4(2, 4, 3, 4);        res[94] = (ch4 == char4(3, 2, 6, 4));
  ch41 = char4(3, 2, 6, 4);         res[95] = (ch4 == ch41);
  ch41 = char4(3, 3, 9, 4);         res[96] = (ch4 != ch41);
  ch4++;                            res[97] = (ch4 == char4(4, 3, 7, 5));
  ++ch4;                            res[98] = (ch4 == char4(5, 4, 8, 6));
  ch4--;                            res[99] = (ch4 == char4(4, 3, 7, 5));
  --ch4;                            res[100] = (ch4 == char4(3, 2, 6, 4));
  ch4 = ch41 + char4(2, 3, -8, -6); res[101] = (ch4 == char4(5, 6, 1, -2));
  ch4 = ch41 - char4(2, 1, -1, 6);  res[102] = (ch4 == char4(1, 2, 10, -2));
  ch4 = ch41 * char4(3, 2, -2, 3);  res[103] = (ch4 == char4(9, 6, -18, 12));
  ch4 = ch41 / char4(2, 3, -3, 5);  res[104] = (ch4 == char4(1, 1, -3, 0));
  ch4 = ch41 % char4(2, 1, 3, 5);   res[105] = (ch4 == char4(1, 0, 0, 4));
  ch4 = ch41 ^ char4(1, 2, 4, 3);   res[106] = (ch4 == char4(2, 1, 13, 7));
  ch4 = ch41 | char4(4, 5, 4, 8);   res[107] = (ch4 == char4(7, 7, 13, 12));
  ch4 = ch41 & char4(5, 4, 8, 6);   res[108] = (ch4 == char4(1, 0, 8, 4));
  ch4 = ch41 << char4(2, 3, 2, 3);  res[109] = (ch4 == char4(12, 24, 36, 32));
  ch4 = ch41 >> char4(1, 2, 3, 4);  res[110] = (ch4 == char4(1, 0, 1, 0));
  ch41 = char4(2, 3, -2, 4); ch4 = ~ch41;   res[111] = (ch4 == char4(-3, -4, 1, -5));

  uchar1 uch1(0), uch11(0);   res[112] = (uch1 == uch11);
  uch1 += uchar1(2);          res[113] = (uch1 == 2);
  uch1 -= uchar1(1);          res[114] = (uch1 == 1);
  uch1 *= uchar1(2);          res[115] = (uch1 == 2);
  uch1 /= uchar1(2);          res[116] = (uch1 == 1);
  uch1 %= uchar1(2);          res[117] = (uch1 == 1);
  uch1 ^= uchar1(2);          res[118] = (uch1 == 3);
  uch1 |= uchar1(5);          res[119] = (uch1 == 7);
  uch1 &= uchar1(2);          res[120] = (uch1 == 2);
  uch1 <<= uchar1(3);         res[121] = (uch1 == 16);
  uch1 >>= uchar1(2);         res[122] = (uch1 == 4);
  uch11 = uchar1(4);          res[123] = (uch1 == uch11);
  uch11 = uchar1(3);          res[124] = (uch1 != uch11);
  uch1++;                     res[125] = (uch1 == 5);
  ++uch1;                     res[126] = (uch1 == 6);
  uch1--;                     res[127] = (uch1 == 5);
  --uch1;                     res[128] = (uch1 == 4);
  uch1 = uch11 + uchar1(2);   res[129] = (uch1 == 5);
  uch1 = uch11 - uchar1(2);   res[130] = (uch1 == 1);
  uch1 = uch11 * uchar1(2);   res[131] = (uch1 == 6);
  uch1 = uch11 / uchar1(2);   res[132] = (uch1 == 1);
  uch1 = uch11 % uchar1(2);   res[133] = (uch1 == 1);
  uch1 = uch11 ^ uchar1(1);   res[134] = (uch1 == 2);
  uch1 = uch11 | uchar1(6);   res[135] = (uch1 == 7);
  uch1 = uch11 & uchar1(6);   res[136] = (uch1 == 2);
  uch1 = uch11 << uchar1(3);  res[137] = (uch1 == 24);
  uch1 = uch11 >> uchar1(3);  res[138] = (uch1 == 0);
  uch11 = uchar1(3); uch1 = ~uch11;   res[139] = (uch1 == 252);

  uchar2 uch2(1, 2), uch21(1, 2);    res[140] = (uch2 == uch21);
  uch2 += uchar2(2, 3);          res[141] = (uch2 == uchar2(3, 5));
  uch2 -= uchar2(1, 2);          res[142] = (uch2 == uchar2(2, 3));
  uch2 *= uchar2(2, 4);          res[143] = (uch2 == uchar2(4, 12));
  uch2 /= uchar2(2, 2);          res[144] = (uch2 == uchar2(2, 6));
  uch2 %= uchar2(3, 4);          res[145] = (uch2 == uchar2(2, 2));
  uch2 ^= uchar2(1, 3);          res[146] = (uch2 == uchar2(3, 1));
  uch2 |= uchar2(4, 5);          res[147] = (uch2 == uchar2(7, 5));
  uch2 &= uchar2(3, 4);          res[148] = (uch2 == uchar2(3, 4));
  uch2 <<= uchar2(2, 3);         res[149] = (uch2 == uchar2(12, 32));
  uch2 >>= uchar2(2, 4);         res[150] = (uch2 == uchar2(3, 2));
  uch21 = uchar2(3, 2);          res[151] = (uch2 == uch21);
  uch21 = uchar2(3, 3);          res[152] = (uch2 != uch21);
  uch2++;                        res[153] = (uch2 == uchar2(4, 3));
  ++uch2;                        res[154] = (uch2 == uchar2(5, 4));
  uch2--;                        res[155] = (uch2 == uchar2(4, 3));
  --uch2;                        res[156] = (uch2 == uchar2(3, 2));
  uch2 = uch21 + uchar2(2, 3);   res[157] = (uch2 == uchar2(5, 6));
  uch2 = uch21 - uchar2(2, 1);   res[158] = (uch2 == uchar2(1, 2));
  uch2 = uch21 * uchar2(3, 2);   res[159] = (uch2 == uchar2(9, 6));
  uch2 = uch21 / uchar2(2, 3);   res[160] = (uch2 == uchar2(1, 1));
  uch2 = uch21 % uchar2(2, 1);   res[161] = (uch2 == uchar2(1, 0));
  uch2 = uch21 ^ uchar2(1, 2);   res[162] = (uch2 == uchar2(2, 1));
  uch2 = uch21 | uchar2(4, 5);   res[163] = (uch2 == uchar2(7, 7));
  uch2 = uch21 & uchar2(5, 4);   res[164] = (uch2 == uchar2(1, 0));
  uch2 = uch21 << uchar2(2, 3);  res[165] = (uch2 == uchar2(12, 24));
  uch2 = uch21 >> uchar2(1, 2);  res[166] = (uch2 == uchar2(1, 0));
  uch21 = uchar2(2, 3); uch2 = ~uch21;   res[167] = (uch2 == uchar2(-3, -4));

  uchar3 uch3(1, 2, 3), uch31(1, 2, 3);    res[168] = (uch3 == uch31);
  uch3 += uchar3(2, 3, 4);          res[169] = (uch3 == uchar3(3, 5, 7));
  uch3 -= uchar3(1, 2, 3);          res[170] = (uch3 == uchar3(2, 3, 4));
  uch3 *= uchar3(2, 4, 5);          res[171] = (uch3 == uchar3(4, 12, 20));
  uch3 /= uchar3(2, 2, 4);          res[172] = (uch3 == uchar3(2, 6, 5));
  uch3 %= uchar3(3, 4, 6);          res[173] = (uch3 == uchar3(2, 2, 5));
  uch3 ^= uchar3(1, 3, 4);          res[174] = (uch3 == uchar3(3, 1, 1));
  uch3 |= uchar3(4, 5, 3);          res[175] = (uch3 == uchar3(7, 5, 3));
  uch3 &= uchar3(3, 4, 7);          res[176] = (uch3 == uchar3(3, 4, 3));
  uch3 <<= uchar3(2, 3, 4);         res[177] = (uch3 == uchar3(12, 32, 48));
  uch3 >>= uchar3(2, 4, 3);         res[178] = (uch3 == uchar3(3, 2, 6));
  uch31 = uchar3(3, 2, 6);          res[179] = (uch3 == uch31);
  uch31 = uchar3(3, 3, 9);          res[180] = (uch3 != uch31);
  uch3++;                           res[181] = (uch3 == uchar3(4, 3, 7));
  ++uch3;                           res[182] = (uch3 == uchar3(5, 4, 8));
  uch3--;                           res[183] = (uch3 == uchar3(4, 3, 7));
  --uch3;                           res[184] = (uch3 == uchar3(3, 2, 6));
  uch3 = uch31 + uchar3(2, 3, -8);  res[185] = (uch3 == uchar3(5, 6, 1));
  uch3 = uch31 - uchar3(2, 1, -1);  res[186] = (uch3 == uchar3(1, 2, 10));
  uch3 = uch31 * uchar3(3, 2, -2);  res[187] = (uch3 == uchar3(9, 6, -18));
  uch3 = uch31 / uchar3(2, 3, -3);  res[188] = (uch3 == uchar3(1, 1, 0));
  uch3 = uch31 % uchar3(2, 1, 3);   res[189] = (uch3 == uchar3(1, 0, 0));
  uch3 = uch31 ^ uchar3(1, 2, 4);   res[190] = (uch3 == uchar3(2, 1, 13));
  uch3 = uch31 | uchar3(4, 5, 4);   res[191] = (uch3 == uchar3(7, 7, 13));
  uch3 = uch31 & uchar3(5, 4, 8);   res[192] = (uch3 == uchar3(1, 0, 8));
  uch3 = uch31 << uchar3(2, 3, 2);  res[193] = (uch3 == uchar3(12, 24, 36));
  uch3 = uch31 >> uchar3(1, 2, 3);  res[194] = (uch3 == uchar3(1, 0, 1));
  uch31 = uchar3(2, 3, -2); uch3 = ~uch31;   res[195] = (uch3 == uchar3(-3, -4, 1));

  uchar4 uch4(1, 2, 3, 4), uch41(1, 2, 3, 4);    res[196] = (uch4 == uch41);
  uch4 += uchar4(2, 3, 4, 5);          res[197] = (uch4 == uchar4(3, 5, 7, 9));
  uch4 -= uchar4(1, 2, 3, 4);          res[198] = (uch4 == uchar4(2, 3, 4, 5));
  uch4 *= uchar4(2, 4, 5, 3);          res[199] = (uch4 == uchar4(4, 12, 20, 15));
  uch4 /= uchar4(2, 2, 4, 4);          res[200] = (uch4 == uchar4(2, 6, 5, 3));
  uch4 %= uchar4(3, 4, 6, 3);          res[201] = (uch4 == uchar4(2, 2, 5, 0));
  uch4 ^= uchar4(1, 3, 4, 2);          res[202] = (uch4 == uchar4(3, 1, 1, 2));
  uch4 |= uchar4(4, 5, 3, 4);          res[203] = (uch4 == uchar4(7, 5, 3, 6));
  uch4 &= uchar4(3, 4, 7, 3);          res[204] = (uch4 == uchar4(3, 4, 3, 2));
  uch4 <<= uchar4(2, 3, 4, 5);         res[205] = (uch4 == uchar4(12, 32, 48, 64));
  uch4 >>= uchar4(2, 4, 3, 4);         res[206] = (uch4 == uchar4(3, 2, 6, 4));
  uch41 = uchar4(3, 2, 6, 4);          res[207] = (uch4 == uch41);
  uch41 = uchar4(3, 3, 9, 4);          res[208] = (uch4 != uch41);
  uch4++;                              res[209] = (uch4 == uchar4(4, 3, 7, 5));
  ++uch4;                              res[210] = (uch4 == uchar4(5, 4, 8, 6));
  uch4--;                              res[211] = (uch4 == uchar4(4, 3, 7, 5));
  --uch4;                              res[212] = (uch4 == uchar4(3, 2, 6, 4));
  uch4 = uch41 + uchar4(2, 3, -8, -6); res[213] = (uch4 == uchar4(5, 6, 1, -2));
  uch4 = uch41 - uchar4(2, 1, -1, 6);  res[214] = (uch4 == uchar4(1, 2, 10, -2));
  uch4 = uch41 * uchar4(3, 2, -2, 3);  res[215] = (uch4 == uchar4(9, 6, -18, 12));
  uch4 = uch41 / uchar4(2, 3, 3, 5);   res[216] = (uch4 == uchar4(1, 1, 3, 0));
  uch4 = uch41 % uchar4(2, 1, 3, 5);   res[217] = (uch4 == uchar4(1, 0, 0, 4));
  uch4 = uch41 ^ uchar4(1, 2, 4, 3);   res[218] = (uch4 == uchar4(2, 1, 13, 7));
  uch4 = uch41 | uchar4(4, 5, 4, 8);   res[219] = (uch4 == uchar4(7, 7, 13, 12));
  uch4 = uch41 & uchar4(5, 4, 8, 6);   res[220] = (uch4 == uchar4(1, 0, 8, 4));
  uch4 = uch41 << uchar4(2, 3, 2, 3);  res[221] = (uch4 == uchar4(12, 24, 36, 32));
  uch4 = uch41 >> uchar4(1, 2, 3, 4);  res[222] = (uch4 == uchar4(1, 0, 1, 0));
  uch41 = uchar4(2, 3, -2, 4); uch4 = ~uch41;   res[223] = (uch4 == uchar4(-3, -4, 1, -5));

  ushort1 ush1(0), ush11(0);   res[224] = (ush1 == ush11);
  ush1 += ushort1(3);          res[225] = (ush1 == 3);
  ush1 -= ushort1(2);          res[226] = (ush1 == 1);
  ush1 *= ushort1(4);          res[227] = (ush1 == 4);
  ush1 /= ushort1(2);          res[228] = (ush1 == 2);
  ush1 %= ushort1(3);          res[229] = (ush1 == 2);
  ush1 ^= ushort1(1);          res[230] = (ush1 == 3);
  ush1 |= ushort1(5);          res[231] = (ush1 == 7);
  ush1 &= ushort1(2);          res[232] = (ush1 == 2);
  ush1 <<= ushort1(4);         res[233] = (ush1 == 32);
  ush1 >>= ushort1(2);         res[234] = (ush1 == 8);
  ush11 = ushort1(8);          res[235] = (ush1 == ush11);
  ush11 = ushort1(3);          res[236] = (ush1 != ush11);
  ush1++;                      res[237] = (ush1 == 9);
  ++ush1;                      res[238] = (ush1 == 10);
  ush1--;                      res[239] = (ush1 == 9);
  --ush1;                      res[240] = (ush1 == 8);
  ush1 = ush11 + ushort1(2);   res[241] = (ush1 == 5);
  ush1 = ush11 - ushort1(2);   res[242] = (ush1 == 1);
  ush1 = ush11 * ushort1(2);   res[243] = (ush1 == 6);
  ush1 = ush11 / ushort1(2);   res[244] = (ush1 == 1);
  ush1 = ush11 % ushort1(2);   res[245] = (ush1 == 1);
  ush1 = ush11 ^ ushort1(1);   res[246] = (ush1 == 2);
  ush1 = ush11 | ushort1(6);   res[247] = (ush1 == 7);
  ush1 = ush11 & ushort1(6);   res[248] = (ush1 == 2);
  ush1 = ush11 << ushort1(2);  res[249] = (ush1 == 12);
  ush1 = ush11 >> ushort1(1);  res[250] = (ush1 == 1);
  ush11 = ushort1(-4); ush1 = ~ush11;   res[251] = (ush1 == 3);

  ushort2 ush2(1, 2), ush21(1, 2);    res[252] = (ush2 == ush21);
  ush2 += ushort2(2, 3);          res[253] = (ush2 == ushort2(3, 5));
  ush2 -= ushort2(1, 2);          res[254] = (ush2 == ushort2(2, 3));
  ush2 *= ushort2(2, 4);          res[255] = (ush2 == ushort2(4, 12));
  ush2 /= ushort2(2, 2);          res[256] = (ush2 == ushort2(2, 6));
  ush2 %= ushort2(3, 4);          res[257] = (ush2 == ushort2(2, 2));
  ush2 ^= ushort2(1, 3);          res[258] = (ush2 == ushort2(3, 1));
  ush2 |= ushort2(4, 5);          res[259] = (ush2 == ushort2(7, 5));
  ush2 &= ushort2(3, 4);          res[260] = (ush2 == ushort2(3, 4));
  ush2 <<= ushort2(2, 3);         res[261] = (ush2 == ushort2(12, 32));
  ush2 >>= ushort2(3, 4);         res[262] = (ush2 == ushort2(1, 2));
  ush21 = ushort2(1, 2);          res[263] = (ush2 == ush21);
  ush21 = ushort2(3, 3);          res[264] = (ush2 != ush21);
  ush2++;                         res[265] = (ush2 == ushort2(2, 3));
  ++ush2;                         res[266] = (ush2 == ushort2(3, 4));
  ush2--;                         res[267] = (ush2 == ushort2(2, 3));
  --ush2;                         res[268] = (ush2 == ushort2(1, 2));
  ush2 = ush21 + ushort2(2, 3);   res[269] = (ush2 == ushort2(5, 6));
  ush2 = ush21 - ushort2(2, 1);   res[270] = (ush2 == ushort2(1, 2));
  ush2 = ush21 * ushort2(3, 2);   res[271] = (ush2 == ushort2(9, 6));
  ush2 = ush21 / ushort2(2, 3);   res[272] = (ush2 == ushort2(1, 1));
  ush2 = ush21 % ushort2(2, 1);   res[273] = (ush2 == ushort2(1, 0));
  ush2 = ush21 ^ ushort2(1, 2);   res[274] = (ush2 == ushort2(2, 1));
  ush2 = ush21 | ushort2(5, 6);   res[275] = (ush2 == ushort2(7, 7));
  ush2 = ush21 & ushort2(5, 4);   res[276] = (ush2 == ushort2(1, 0));
  ush2 = ush21 << ushort2(2, 3);  res[277] = (ush2 == ushort2(12, 24));
  ush2 = ush21 >> ushort2(1, 2);  res[278] = (ush2 == ushort2(1, 0));
  ush21 = ushort2(-2, -3); ush2 = ~ush21;   res[279] = (ush2 == ushort2(1, 2));

  ushort3 ush3(1, 2, 3), ush31(1, 2, 3);    res[280] = (ush3 == ush31);
  ush3 += ushort3(3, 5, 7);          res[281] = (ush3 == ushort3(4, 7, 10));
  ush3 -= ushort3(2, 4, 6);          res[282] = (ush3 == ushort3(2, 3, 4));
  ush3 *= ushort3(2, 3, 5);          res[283] = (ush3 == ushort3(4, 9, 20));
  ush3 /= ushort3(2, 3, 4);          res[284] = (ush3 == ushort3(2, 3, 5));
  ush3 %= ushort3(3, 4, 3);          res[285] = (ush3 == ushort3(2, 3, 2));
  ush3 ^= ushort3(1, 2, 4);          res[286] = (ush3 == ushort3(3, 1, 6));
  ush3 |= ushort3(4, 5, 3);          res[287] = (ush3 == ushort3(7, 5, 7));
  ush3 &= ushort3(3, 4, 3);          res[288] = (ush3 == ushort3(3, 4, 3));
  ush3 <<= ushort3(2, 3, 4);         res[289] = (ush3 == ushort3(12, 32, 48));
  ush3 >>= ushort3(2, 4, 3);         res[290] = (ush3 == ushort3(3, 2, 6));
  ush31 = ushort3(3, 2, 6);          res[291] = (ush3 == ush31);
  ush31 = ushort3(3, 3, 9);          res[292] = (ush3 != ush31);
  ush3++;                            res[293] = (ush3 == ushort3(4, 3, 7));
  ++ush3;                            res[294] = (ush3 == ushort3(5, 4, 8));
  ush3--;                            res[295] = (ush3 == ushort3(4, 3, 7));
  --ush3;                            res[296] = (ush3 == ushort3(3, 2, 6));
  ush3 = ush31 + ushort3(2, 3, -8);  res[297] = (ush3 == ushort3(5, 6, 1));
  ush3 = ush31 - ushort3(2, 1, -1);  res[298] = (ush3 == ushort3(1, 2, 10));
  ush3 = ush31 * ushort3(3, 2, -2);  res[299] = (ush3 == ushort3(9, 6, -18));
  ush3 = ush31 / ushort3(2, 3, -3);  res[300] = (ush3 == ushort3(1, 1, 0));
  ush3 = ush31 % ushort3(2, 1, 3);   res[301] = (ush3 == ushort3(1, 0, 0));
  ush3 = ush31 ^ ushort3(1, 2, 4);   res[302] = (ush3 == ushort3(2, 1, 13));
  ush3 = ush31 | ushort3(4, 5, 4);   res[303] = (ush3 == ushort3(7, 7, 13));
  ush3 = ush31 & ushort3(5, 4, 8);   res[304] = (ush3 == ushort3(1, 0, 8));
  ush3 = ush31 << ushort3(2, 3, 2);  res[305] = (ush3 == ushort3(12, 24, 36));
  ush3 = ush31 >> ushort3(1, 2, 3);  res[306] = (ush3 == ushort3(1, 0, 1));
  ush31 = ushort3(1, 4, -3); ush3 = ~ush31;   res[307] = (ush3 == ushort3(-2, -5, 2));

  ushort4 ush4(1, 2, 3, 4), ush41(1, 2, 3, 4);    res[308] = (ush4 == ush41);
  ush4 += ushort4(2, 5, 3, 4);          res[309] = (ush4 == ushort4(3, 7, 6, 8));
  ush4 -= ushort4(1, 4, 2, 3);          res[310] = (ush4 == ushort4(2, 3, 4, 5));
  ush4 *= ushort4(2, 3, 5, 4);          res[311] = (ush4 == ushort4(4, 9, 20, 20));
  ush4 /= ushort4(2, 3, 4, 6);          res[312] = (ush4 == ushort4(2, 3, 5, 3));
  ush4 %= ushort4(3, 5, 6, 3);          res[313] = (ush4 == ushort4(2, 3, 5, 0));
  ush4 ^= ushort4(1, 2, 4, 2);          res[314] = (ush4 == ushort4(3, 1, 1, 2));
  ush4 |= ushort4(4, 5, 3, 4);          res[315] = (ush4 == ushort4(7, 5, 3, 6));
  ush4 &= ushort4(3, 4, 7, 3);          res[316] = (ush4 == ushort4(3, 4, 3, 2));
  ush4 <<= ushort4(2, 3, 4, 5);         res[317] = (ush4 == ushort4(12, 32, 48, 64));
  ush4 >>= ushort4(2, 3, 3, 5);         res[318] = (ush4 == ushort4(3, 4, 6, 2));
  ush41 = ushort4(3, 4, 6, 2);          res[319] = (ush4 == ush41);
  ush41 = ushort4(3, 3, 9, 4);          res[320] = (ush4 != ush41);
  ush4++;                               res[321] = (ush4 == ushort4(4, 5, 7, 3));
  ++ush4;                               res[322] = (ush4 == ushort4(5, 6, 8, 4));
  ush4--;                               res[323] = (ush4 == ushort4(4, 5, 7, 3));
  --ush4;                               res[324] = (ush4 == ushort4(3, 4, 6, 2));
  ush4 = ush41 + ushort4(2, 3, -8, -6); res[325] = (ush4 == ushort4(5, 6, 1, -2));
  ush4 = ush41 - ushort4(2, 1, -1, 6);  res[326] = (ush4 == ushort4(1, 2, 10, -2));
  ush4 = ush41 * ushort4(3, 2, -2, 3);  res[327] = (ush4 == ushort4(9, 6, -18, 12));
  ush4 = ush41 / ushort4(2, 3, 3, 5);   res[328] = (ush4 == ushort4(1, 1, 3, 0));
  ush4 = ush41 % ushort4(2, 1, 3, 5);   res[329] = (ush4 == ushort4(1, 0, 0, 4));
  ush4 = ush41 ^ ushort4(1, 2, 4, 3);   res[330] = (ush4 == ushort4(2, 1, 13, 7));
  ush4 = ush41 | ushort4(4, 5, 4, 8);   res[331] = (ush4 == ushort4(7, 7, 13, 12));
  ush4 = ush41 & ushort4(5, 4, 8, 6);   res[332] = (ush4 == ushort4(1, 0, 8, 4));
  ush4 = ush41 << ushort4(2, 3, 2, 3);  res[333] = (ush4 == ushort4(12, 24, 36, 32));
  ush4 = ush41 >> ushort4(1, 2, 3, 4);  res[334] = (ush4 == ushort4(1, 0, 1, 0));
  ush41 = ushort4(2, 3, -2, 4); ush4 = ~ush41;   res[335] = (ush4 == ushort4(-3, -4, 1, -5));

  short1 sh1(1), sh11(1);   res[336] = (sh1 == sh11);
  sh1 += short1(1);         res[337] = (sh1 == 2);
  sh1 -= short1(1);         res[338] = (sh1 == 1);
  sh1 *= short1(3);         res[339] = (sh1 == 3);
  sh1 /= short1(3);         res[340] = (sh1 == 1);
  sh1 %= short1(4);         res[341] = (sh1 == 1);
  sh1 ^= short1(6);         res[342] = (sh1 == 7);
  sh1 |= short1(9);         res[343] = (sh1 == 15);
  sh1 &= short1(3);         res[344] = (sh1 == 3);
  sh1 <<= short1(4);        res[345] = (sh1 == 48);
  sh1 >>= short1(3);        res[346] = (sh1 == 6);
  sh11 = short1(6);         res[347] = (sh1 == sh11);
  sh11 = short1(3);         res[348] = (sh1 != sh11);
  sh1++;                    res[349] = (sh1 == 7);
  ++sh1;                    res[350] = (sh1 == 8);
  sh1--;                    res[351] = (sh1 == 7);
  --sh1;                    res[352] = (sh1 == 6);
  sh1 = sh11 + short1(4);   res[353] = (sh1 == 7);
  sh1 = sh11 - short1(2);   res[354] = (sh1 == 1);
  sh1 = sh11 * short1(3);   res[355] = (sh1 == 9);
  sh1 = sh11 / short1(2);   res[356] = (sh1 == 1);
  sh1 = sh11 % short1(2);   res[357] = (sh1 == 1);
  sh1 = sh11 ^ short1(1);   res[358] = (sh1 == 2);
  sh1 = sh11 | short1(6);   res[359] = (sh1 == 7);
  sh1 = sh11 & short1(6);   res[360] = (sh1 == 2);
  sh1 = sh11 << short1(3);  res[361] = (sh1 == 24);
  sh1 = sh11 >> short1(3);  res[362] = (sh1 == 0);
  sh11 = short1(3); sh1 = ~sh11;   res[363] = (sh1 == -4);

  short2 sh2(1, 2), sh21(1, 2);    res[364] = (sh2 == sh21);
  sh2 += short2(2, 3);         res[365] = (sh2 == short2(3, 5));
  sh2 -= short2(1, 2);         res[366] = (sh2 == short2(2, 3));
  sh2 *= short2(2, 4);         res[367] = (sh2 == short2(4, 12));
  sh2 /= short2(2, 2);         res[368] = (sh2 == short2(2, 6));
  sh2 %= short2(3, 4);         res[369] = (sh2 == short2(2, 2));
  sh2 ^= short2(1, 3);         res[370] = (sh2 == short2(3, 1));
  sh2 |= short2(4, 5);         res[371] = (sh2 == short2(7, 5));
  sh2 &= short2(3, 4);         res[372] = (sh2 == short2(3, 4));
  sh2 <<= short2(2, 3);        res[373] = (sh2 == short2(12, 32));
  sh2 >>= short2(2, 4);        res[374] = (sh2 == short2(3, 2));
  sh21 = short2(3, 2);         res[375] = (sh2 == sh21);
  sh21 = short2(3, 3);         res[376] = (sh2 != sh21);
  sh2++;                       res[377] = (sh2 == short2(4, 3));
  ++sh2;                       res[378] = (sh2 == short2(5, 4));
  sh2--;                       res[379] = (sh2 == short2(4, 3));
  --sh2;                       res[380] = (sh2 == short2(3, 2));
  sh2 = sh21 + short2(2, 3);   res[381] = (sh2 == short2(5, 6));
  sh2 = sh21 - short2(2, 1);   res[382] = (sh2 == short2(1, 2));
  sh2 = sh21 * short2(3, 2);   res[383] = (sh2 == short2(9, 6));
  sh2 = sh21 / short2(2, 3);   res[384] = (sh2 == short2(1, 1));
  sh2 = sh21 % short2(2, 1);   res[385] = (sh2 == short2(1, 0));
  sh2 = sh21 ^ short2(1, 2);   res[386] = (sh2 == short2(2, 1));
  sh2 = sh21 | short2(4, 5);   res[387] = (sh2 == short2(7, 7));
  sh2 = sh21 & short2(5, 4);   res[388] = (sh2 == short2(1, 0));
  sh2 = sh21 << short2(2, 3);  res[389] = (sh2 == short2(12, 24));
  sh2 = sh21 >> short2(1, 2);  res[390] = (sh2 == short2(1, 0));
  sh21 = short2(2, 3); sh2 = ~sh21;   res[391] = (sh2 == short2(-3, -4));

  short3 sh3(1, 2, 3), sh31(1, 2, 3);    res[392] = (sh3 == sh31);
  sh3 += short3(2, 3, 4);         res[393] = (sh3 == short3(3, 5, 7));
  sh3 -= short3(1, 2, 3);         res[394] = (sh3 == short3(2, 3, 4));
  sh3 *= short3(2, 4, -5);        res[395] = (sh3 == short3(4, 12, -20));
  sh3 /= short3(2, 2, -4);        res[396] = (sh3 == short3(2, 6, 5));
  sh3 %= short3(3, 4, 6);         res[397] = (sh3 == short3(2, 2, 5));
  sh3 ^= short3(1, 3, 4);         res[398] = (sh3 == short3(3, 1, 1));
  sh3 |= short3(4, 5, 3);         res[399] = (sh3 == short3(7, 5, 3));
  sh3 &= short3(3, 4, 7);         res[400] = (sh3 == short3(3, 4, 3));
  sh3 <<= short3(2, 3, 4);        res[401] = (sh3 == short3(12, 32, 48));
  sh3 >>= short3(2, 4, 3);        res[402] = (sh3 == short3(3, 2, 6));
  sh31 = short3(3, 2, 6);         res[403] = (sh3 == sh31);
  sh31 = short3(3, 3, 9);         res[404] = (sh3 != sh31);
  sh3++;                          res[405] = (sh3 == short3(4, 3, 7));
  ++sh3;                          res[406] = (sh3 == short3(5, 4, 8));
  sh3--;                          res[407] = (sh3 == short3(4, 3, 7));
  --sh3;                          res[408] = (sh3 == short3(3, 2, 6));
  sh3 = sh31 + short3(2, 3, -8);  res[409] = (sh3 == short3(5, 6, 1));
  sh3 = sh31 - short3(2, 1, -1);  res[410] = (sh3 == short3(1, 2, 10));
  sh3 = sh31 * short3(3, 2, -2);  res[411] = (sh3 == short3(9, 6, -18));
  sh3 = sh31 / short3(2, 3, -3);  res[412] = (sh3 == short3(1, 1, -3));
  sh3 = sh31 % short3(2, 1, 3);   res[413] = (sh3 == short3(1, 0, 0));
  sh3 = sh31 ^ short3(1, 2, 4);   res[414] = (sh3 == short3(2, 1, 13));
  sh3 = sh31 | short3(4, 5, 4);   res[415] = (sh3 == short3(7, 7, 13));
  sh3 = sh31 & short3(5, 4, 8);   res[416] = (sh3 == short3(1, 0, 8));
  sh3 = sh31 << short3(2, 3, 2);  res[417] = (sh3 == short3(12, 24, 36));
  sh3 = sh31 >> short3(1, 2, 3);  res[418] = (sh3 == short3(1, 0, 1));
  sh31 = short3(2, 3, -2); sh3 = ~sh31;   res[419] = (sh3 == short3(-3, -4, 1));

  short4 sh4(1, 2, 3, 4), sh41(1, 2, 3, 4);    res[420] = (sh4 == sh41);
  sh4 += short4(2, 3, 4, 5);         res[421] = (sh4 == short4(3, 5, 7, 9));
  sh4 -= short4(1, 2, 3, 4);         res[422] = (sh4 == short4(2, 3, 4, 5));
  sh4 *= short4(2, 4, -5, 3);        res[423] = (sh4 == short4(4, 12, -20, 15));
  sh4 /= short4(2, 2, -4, 4);        res[424] = (sh4 == short4(2, 6, 5, 3));
  sh4 %= short4(3, 4, 6, 3);         res[425] = (sh4 == short4(2, 2, 5, 0));
  sh4 ^= short4(1, 3, 4, 2);         res[426] = (sh4 == short4(3, 1, 1, 2));
  sh4 |= short4(4, 5, 3, 4);         res[427] = (sh4 == short4(7, 5, 3, 6));
  sh4 &= short4(3, 4, 7, 3);         res[428] = (sh4 == short4(3, 4, 3, 2));
  sh4 <<= short4(2, 3, 4, 5);        res[429] = (sh4 == short4(12, 32, 48, 64));
  sh4 >>= short4(2, 4, 3, 4);        res[430] = (sh4 == short4(3, 2, 6, 4));
  sh41 = short4(3, 2, 6, 4);         res[431] = (sh4 == sh41);
  sh41 = short4(3, 3, 9, 4);         res[432] = (sh4 != sh41);
  sh4++;                             res[433] = (sh4 == short4(4, 3, 7, 5));
  ++sh4;                             res[434] = (sh4 == short4(5, 4, 8, 6));
  sh4--;                             res[435] = (sh4 == short4(4, 3, 7, 5));
  --sh4;                             res[436] = (sh4 == short4(3, 2, 6, 4));
  sh4 = sh41 + short4(2, 3, -8, -6); res[437] = (sh4 == short4(5, 6, 1, -2));
  sh4 = sh41 - short4(2, 1, -1, 6);  res[438] = (sh4 == short4(1, 2, 10, -2));
  sh4 = sh41 * short4(3, 2, -2, 3);  res[439] = (sh4 == short4(9, 6, -18, 12));
  sh4 = sh41 / short4(2, 3, -3, 5);  res[440] = (sh4 == short4(1, 1, -3, 0));
  sh4 = sh41 % short4(2, 1, 3, 5);   res[441] = (sh4 == short4(1, 0, 0, 4));
  sh4 = sh41 ^ short4(1, 2, 4, 3);   res[442] = (sh4 == short4(2, 1, 13, 7));
  sh4 = sh41 | short4(4, 5, 4, 8);   res[443] = (sh4 == short4(7, 7, 13, 12));
  sh4 = sh41 & short4(5, 4, 8, 6);   res[444] = (sh4 == short4(1, 0, 8, 4));
  sh4 = sh41 << short4(2, 3, 2, 3);  res[445] = (sh4 == short4(12, 24, 36, 32));
  sh4 = sh41 >> short4(1, 2, 3, 4);  res[446] = (sh4 == short4(1, 0, 1, 0));
  sh41 = short4(2, 3, -2, 4); sh4 = ~sh41;   res[447] = (sh4 == short4(-3, -4, 1, -5));

  uint1 ui1(0), ui11(0);    res[448] = (ui1 == ui11);
  ui1 += uint1(3);          res[449] = (ui1 == 3);
  ui1 -= uint1(2);          res[450] = (ui1 == 1);
  ui1 *= uint1(4);          res[451] = (ui1 == 4);
  ui1 /= uint1(2);          res[452] = (ui1 == 2);
  ui1 %= uint1(3);          res[453] = (ui1 == 2);
  ui1 ^= uint1(1);          res[454] = (ui1 == 3);
  ui1 |= uint1(5);          res[455] = (ui1 == 7);
  ui1 &= uint1(2);          res[456] = (ui1 == 2);
  ui1 <<= uint1(4);         res[457] = (ui1 == 32);
  ui1 >>= uint1(2);         res[458] = (ui1 == 8);
  ui11 = uint1(8);          res[459] = (ui1 == ui11);
  ui11 = uint1(3);          res[460] = (ui1 != ui11);
  ui1++;                    res[461] = (ui1 == 9);
  ++ui1;                    res[462] = (ui1 == 10);
  ui1--;                    res[463] = (ui1 == 9);
  --ui1;                    res[464] = (ui1 == 8);
  ui1 = ui11 + uint1(2);    res[465] = (ui1 == 5);
  ui1 = ui11 - uint1(2);    res[466] = (ui1 == 1);
  ui1 = ui11 * uint1(2);    res[467] = (ui1 == 6);
  ui1 = ui11 / uint1(2);    res[468] = (ui1 == 1);
  ui1 = ui11 % uint1(2);    res[469] = (ui1 == 1);
  ui1 = ui11 ^ uint1(1);    res[470] = (ui1 == 2);
  ui1 = ui11 | uint1(6);    res[471] = (ui1 == 7);
  ui1 = ui11 & uint1(6);    res[472] = (ui1 == 2);
  ui1 = ui11 << uint1(2);   res[473] = (ui1 == 12);
  ui1 = ui11 >> uint1(1);   res[474] = (ui1 == 1);
  ui11 = uint1(-5); ui1 = ~ui11;   res[475] = (ui1 == 4);

  uint2 ui2(1, 2), ui21(1, 2);     res[476] = (ui2 == ui21);
  ui2 += uint2(2, 3);          res[477] = (ui2 == uint2(3, 5));
  ui2 -= uint2(1, 2);          res[478] = (ui2 == uint2(2, 3));
  ui2 *= uint2(2, 4);          res[479] = (ui2 == uint2(4, 12));
  ui2 /= uint2(2, 2);          res[480] = (ui2 == uint2(2, 6));
  ui2 %= uint2(3, 4);          res[481] = (ui2 == uint2(2, 2));
  ui2 ^= uint2(1, 3);          res[482] = (ui2 == uint2(3, 1));
  ui2 |= uint2(4, 5);          res[483] = (ui2 == uint2(7, 5));
  ui2 &= uint2(3, 4);          res[484] = (ui2 == uint2(3, 4));
  ui2 <<= uint2(2, 3);         res[485] = (ui2 == uint2(12, 32));
  ui2 >>= uint2(2, 4);         res[486] = (ui2 == uint2(3, 2));
  ui21 = uint2(3, 2);          res[487] = (ui2 == ui21);
  ui21 = uint2(3, 3);          res[488] = (ui2 != ui21);
  ui2++;                       res[489] = (ui2 == uint2(4, 3));
  ++ui2;                       res[490] = (ui2 == uint2(5, 4));
  ui2--;                       res[491] = (ui2 == uint2(4, 3));
  --ui2;                       res[492] = (ui2 == uint2(3, 2));
  ui2 = ui21 + uint2(2, 3);    res[493] = (ui2 == uint2(5, 6));
  ui2 = ui21 - uint2(2, 1);    res[494] = (ui2 == uint2(1, 2));
  ui2 = ui21 * uint2(3, 2);    res[495] = (ui2 == uint2(9, 6));
  ui2 = ui21 / uint2(2, 3);    res[496] = (ui2 == uint2(1, 1));
  ui2 = ui21 % uint2(2, 1);    res[497] = (ui2 == uint2(1, 0));
  ui2 = ui21 ^ uint2(1, 2);    res[498] = (ui2 == uint2(2, 1));
  ui2 = ui21 | uint2(4, 5);    res[499] = (ui2 == uint2(7, 7));
  ui2 = ui21 & uint2(5, 4);    res[500] = (ui2 == uint2(1, 0));
  ui2 = ui21 << uint2(2, 3);   res[501] = (ui2 == uint2(12, 24));
  ui2 = ui21 >> uint2(1, 2);   res[502] = (ui2 == uint2(1, 0));
  ui21 = uint2(1, -5); ui2 = ~ui21;   res[503] = (ui2 == uint2(-2, 4));

  uint3 ui3(1, 2, 3), ui31(1, 2, 3);    res[504] = (ui3 == ui31);
  ui3 += uint3(2, 3, 4);          res[505] = (ui3 == uint3(3, 5, 7));
  ui3 -= uint3(1, 2, 3);          res[506] = (ui3 == uint3(2, 3, 4));
  ui3 *= uint3(2, 4, 5);          res[507] = (ui3 == uint3(4, 12, 20));
  ui3 /= uint3(2, 2, 4);          res[508] = (ui3 == uint3(2, 6, 5));
  ui3 %= uint3(3, 4, 6);          res[509] = (ui3 == uint3(2, 2, 5));
  ui3 ^= uint3(1, 3, 4);          res[510] = (ui3 == uint3(3, 1, 1));
  ui3 |= uint3(4, 5, 3);          res[511] = (ui3 == uint3(7, 5, 3));
  ui3 &= uint3(3, 4, 7);          res[512] = (ui3 == uint3(3, 4, 3));
  ui3 <<= uint3(2, 3, 4);         res[513] = (ui3 == uint3(12, 32, 48));
  ui3 >>= uint3(2, 4, 3);         res[514] = (ui3 == uint3(3, 2, 6));
  ui31 = uint3(3, 2, 6);          res[515] = (ui3 == ui31);
  ui31 = uint3(3, 3, 9);          res[516] = (ui3 != ui31);
  ui3++;                          res[517] = (ui3 == uint3(4, 3, 7));
  ++ui3;                          res[518] = (ui3 == uint3(5, 4, 8));
  ui3--;                          res[519] = (ui3 == uint3(4, 3, 7));
  --ui3;                          res[520] = (ui3 == uint3(3, 2, 6));
  ui3 = ui31 + uint3(2, 3, -8);   res[521] = (ui3 == uint3(5, 6, 1));
  ui3 = ui31 - uint3(2, 1, -1);   res[522] = (ui3 == uint3(1, 2, 10));
  ui3 = ui31 * uint3(3, 2, -2);   res[523] = (ui3 == uint3(9, 6, -18));
  ui3 = ui31 / uint3(2, 3, -3);   res[524] = (ui3 == uint3(1, 1, 0));
  ui3 = ui31 % uint3(2, 1, 3);    res[525] = (ui3 == uint3(1, 0, 0));
  ui3 = ui31 ^ uint3(1, 2, 4);    res[526] = (ui3 == uint3(2, 1, 13));
  ui3 = ui31 | uint3(4, 5, 4);    res[527] = (ui3 == uint3(7, 7, 13));
  ui3 = ui31 & uint3(5, 4, 8);    res[528] = (ui3 == uint3(1, 0, 8));
  ui3 = ui31 << uint3(2, 3, 2);   res[529] = (ui3 == uint3(12, 24, 36));
  ui3 = ui31 >> uint3(1, 2, 3);   res[530] = (ui3 == uint3(1, 0, 1));
  ui31 = uint3(2, 3, -2); ui3 = ~ui31;   res[531] = (ui3 == uint3(-3, -4, 1));

  uint4 ui4(1, 2, 3, 4), ui41(1, 2, 3, 4);    res[532] = (ui4 == ui41);
  ui4 += uint4(2, 3, 4, 5);          res[533] = (ui4 == uint4(3, 5, 7, 9));
  ui4 -= uint4(1, 2, 3, 4);          res[534] = (ui4 == uint4(2, 3, 4, 5));
  ui4 *= uint4(2, 4, 5, 3);          res[535] = (ui4 == uint4(4, 12, 20, 15));
  ui4 /= uint4(2, 2, 4, 4);          res[536] = (ui4 == uint4(2, 6, 5, 3));
  ui4 %= uint4(3, 4, 6, 3);          res[537] = (ui4 == uint4(2, 2, 5, 0));
  ui4 ^= uint4(1, 3, 4, 2);          res[538] = (ui4 == uint4(3, 1, 1, 2));
  ui4 |= uint4(4, 5, 3, 4);          res[539] = (ui4 == uint4(7, 5, 3, 6));
  ui4 &= uint4(3, 4, 7, 3);          res[540] = (ui4 == uint4(3, 4, 3, 2));
  ui4 <<= uint4(2, 3, 4, 5);         res[541] = (ui4 == uint4(12, 32, 48, 64));
  ui4 >>= uint4(2, 4, 3, 4);         res[542] = (ui4 == uint4(3, 2, 6, 4));
  ui41 = uint4(3, 2, 6, 4);          res[543] = (ui4 == ui41);
  ui41 = uint4(3, 3, 9, 4);          res[544] = (ui4 != ui41);
  ui4++;                             res[545] = (ui4 == uint4(4, 3, 7, 5));
  ++ui4;                             res[546] = (ui4 == uint4(5, 4, 8, 6));
  ui4--;                             res[547] = (ui4 == uint4(4, 3, 7, 5));
  --ui4;                             res[548] = (ui4 == uint4(3, 2, 6, 4));
  ui4 = ui41 + uint4(2, 3, -8, -6);  res[549] = (ui4 == uint4(5, 6, 1, -2));
  ui4 = ui41 - uint4(2, 1, -1, 6);   res[550] = (ui4 == uint4(1, 2, 10, -2));
  ui4 = ui41 * uint4(3, 2, -2, 3);   res[551] = (ui4 == uint4(9, 6, -18, 12));
  ui4 = ui41 / uint4(2, 3, 3, 5);    res[552] = (ui4 == uint4(1, 1, 3, 0));
  ui4 = ui41 % uint4(2, 1, 3, 5);    res[553] = (ui4 == uint4(1, 0, 0, 4));
  ui4 = ui41 ^ uint4(1, 2, 4, 3);    res[554] = (ui4 == uint4(2, 1, 13, 7));
  ui4 = ui41 | uint4(4, 5, 4, 8);    res[555] = (ui4 == uint4(7, 7, 13, 12));
  ui4 = ui41 & uint4(5, 4, 8, 6);    res[556] = (ui4 == uint4(1, 0, 8, 4));
  ui4 = ui41 << uint4(2, 3, 2, 3);   res[557] = (ui4 == uint4(12, 24, 36, 32));
  ui4 = ui41 >> uint4(1, 2, 3, 4);   res[558] = (ui4 == uint4(1, 0, 1, 0));
  ui41 = uint4(2, 3, -2, 4); ui4 = ~ui41;   res[559] = (ui4 == uint4(-3, -4, 1, -5));

  int1 i1(1), i11(1);    res[560] = (i1 == i11);
  i1 += int1(1);         res[561] = (i1 == 2);
  i1 -= int1(1);         res[562] = (i1 == 1);
  i1 *= int1(3);         res[563] = (i1 == 3);
  i1 /= int1(3);         res[564] = (i1 == 1);
  i1 %= int1(4);         res[565] = (i1 == 1);
  i1 ^= int1(6);         res[566] = (i1 == 7);
  i1 |= int1(9);         res[567] = (i1 == 15);
  i1 &= int1(3);         res[568] = (i1 == 3);
  i1 <<= int1(4);        res[569] = (i1 == 48);
  i1 >>= int1(3);        res[570] = (i1 == 6);
  i11 = int1(6);         res[571] = (i1 == i11);
  i11 = int1(3);         res[572] = (i1 != i11);
  i1++;                  res[573] = (i1 == 7);
  ++i1;                  res[574] = (i1 == 8);
  i1--;                  res[575] = (i1 == 7);
  --i1;                  res[576] = (i1 == 6);
  i1 = i11 + int1(4);    res[577] = (i1 == 7);
  i1 = i11 - int1(2);    res[578] = (i1 == 1);
  i1 = i11 * int1(3);    res[579] = (i1 == 9);
  i1 = i11 / int1(2);    res[580] = (i1 == 1);
  i1 = i11 % int1(2);    res[581] = (i1 == 1);
  i1 = i11 ^ int1(1);    res[582] = (i1 == 2);
  i1 = i11 | int1(6);    res[583] = (i1 == 7);
  i1 = i11 & int1(6);    res[584] = (i1 == 2);
  i1 = i11 << int1(3);   res[585] = (i1 == 24);
  i1 = i11 >> int1(3);   res[586] = (i1 == 0);
  i11 = int1(3); i1 = ~i11;   res[587] = (i1 == -4);

  int2 i2(1, 2), i21(1, 2);   res[588] = (i2 == i21);
  i2 += int2(2, 3);           res[589] = (i2 == int2(3, 5));
  i2 -= int2(1, 2);           res[590] = (i2 == int2(2, 3));
  i2 *= int2(2, 4);           res[591] = (i2 == int2(4, 12));
  i2 /= int2(2, 2);           res[592] = (i2 == int2(2, 6));
  i2 %= int2(3, 4);           res[593] = (i2 == int2(2, 2));
  i2 ^= int2(1, 3);           res[594] = (i2 == int2(3, 1));
  i2 |= int2(4, 5);           res[595] = (i2 == int2(7, 5));
  i2 &= int2(3, 4);           res[596] = (i2 == int2(3, 4));
  i2 <<= int2(2, 3);          res[597] = (i2 == int2(12, 32));
  i2 >>= int2(2, 4);          res[598] = (i2 == int2(3, 2));
  i21 = int2(3, 2);           res[599] = (i2 == i21);
  i21 = int2(3, 3);           res[600] = (i2 != i21);
  i2++;                       res[601] = (i2 == int2(4, 3));
  ++i2;                       res[602] = (i2 == int2(5, 4));
  i2--;                       res[603] = (i2 == int2(4, 3));
  --i2;                       res[604] = (i2 == int2(3, 2));
  i2 = i21 + int2(2, 3);      res[605] = (i2 == int2(5, 6));
  i2 = i21 - int2(2, 1);      res[606] = (i2 == int2(1, 2));
  i2 = i21 * int2(3, 2);      res[607] = (i2 == int2(9, 6));
  i2 = i21 / int2(2, 3);      res[608] = (i2 == int2(1, 1));
  i2 = i21 % int2(2, 1);      res[609] = (i2 == int2(1, 0));
  i2 = i21 ^ int2(1, 2);      res[610] = (i2 == int2(2, 1));
  i2 = i21 | int2(4, 5);      res[611] = (i2 == int2(7, 7));
  i2 = i21 & int2(5, 4);      res[612] = (i2 == int2(1, 0));
  i2 = i21 << int2(2, 3);     res[613] = (i2 == int2(12, 24));
  i2 = i21 >> int2(1, 2);     res[614] = (i2 == int2(1, 0));
  i21 = int2(2, 3); i2 = ~i21;   res[615] = (i2 == int2(-3, -4));

  int3 i3(1, 2, 3), i31(1, 2, 3);    res[616] = (i3 == i31);
  i3 += int3(2, 3, 4);         res[617] = (i3 == int3(3, 5, 7));
  i3 -= int3(1, 2, 3);         res[618] = (i3 == int3(2, 3, 4));
  i3 *= int3(2, 4, -5);        res[619] = (i3 == int3(4, 12, -20));
  i3 /= int3(2, 2, -4);        res[620] = (i3 == int3(2, 6, 5));
  i3 %= int3(3, 4, 6);         res[621] = (i3 == int3(2, 2, 5));
  i3 ^= int3(1, 3, 4);         res[622] = (i3 == int3(3, 1, 1));
  i3 |= int3(4, 5, 3);         res[623] = (i3 == int3(7, 5, 3));
  i3 &= int3(3, 4, 7);         res[624] = (i3 == int3(3, 4, 3));
  i3 <<= int3(2, 3, 4);        res[625] = (i3 == int3(12, 32, 48));
  i3 >>= int3(2, 4, 3);        res[626] = (i3 == int3(3, 2, 6));
  i31 = int3(3, 2, 6);         res[627] = (i3 == i31);
  i31 = int3(3, 3, 9);         res[628] = (i3 != i31);
  i3++;                        res[629] = (i3 == int3(4, 3, 7));
  ++i3;                        res[630] = (i3 == int3(5, 4, 8));
  i3--;                        res[631] = (i3 == int3(4, 3, 7));
  --i3;                        res[632] = (i3 == int3(3, 2, 6));
  i3 = i31 + int3(2, 3, -8);   res[633] = (i3 == int3(5, 6, 1));
  i3 = i31 - int3(2, 1, -1);   res[634] = (i3 == int3(1, 2, 10));
  i3 = i31 * int3(3, 2, -2);   res[635] = (i3 == int3(9, 6, -18));
  i3 = i31 / int3(2, 3, -3);   res[636] = (i3 == int3(1, 1, -3));
  i3 = i31 % int3(2, 1, 3);    res[637] = (i3 == int3(1, 0, 0));
  i3 = i31 ^ int3(1, 2, 4);    res[638] = (i3 == int3(2, 1, 13));
  i3 = i31 | int3(4, 5, 4);    res[639] = (i3 == int3(7, 7, 13));
  i3 = i31 & int3(5, 4, 8);    res[640] = (i3 == int3(1, 0, 8));
  i3 = i31 << int3(2, 3, 2);   res[641] = (i3 == int3(12, 24, 36));
  i3 = i31 >> int3(1, 2, 3);   res[642] = (i3 == int3(1, 0, 1));
  i31 = int3(5, -3, -4); i3 = ~i31;   res[643] = (i3 == int3(-6, 2, 3));

  int4 i4(1, 2, 3, 4), i41(1, 2, 3, 4);    res[644] = (i4 == i41);
  i4 += int4(2, 3, 4, 5);         res[645] = (i4 == int4(3, 5, 7, 9));
  i4 -= int4(1, 2, 3, 4);         res[646] = (i4 == int4(2, 3, 4, 5));
  i4 *= int4(2, 4, -5, 3);        res[647] = (i4 == int4(4, 12, -20, 15));
  i4 /= int4(2, 2, -4, 4);        res[648] = (i4 == int4(2, 6, 5, 3));
  i4 %= int4(3, 4, 6, 3);         res[649] = (i4 == int4(2, 2, 5, 0));
  i4 ^= int4(1, 3, 4, 2);         res[650] = (i4 == int4(3, 1, 1, 2));
  i4 |= int4(4, 5, 3, 4);         res[651] = (i4 == int4(7, 5, 3, 6));
  i4 &= int4(3, 4, 7, 3);         res[652] = (i4 == int4(3, 4, 3, 2));
  i4 <<= int4(2, 3, 4, 5);        res[653] = (i4 == int4(12, 32, 48, 64));
  i4 >>= int4(2, 4, 3, 4);        res[654] = (i4 == int4(3, 2, 6, 4));
  i41 = int4(3, 2, 6, 4);         res[655] = (i4 == i41);
  i41 = int4(3, 3, 9, 4);         res[656] = (i4 != i41);
  i4++;                           res[657] = (i4 == int4(4, 3, 7, 5));
  ++i4;                           res[658] = (i4 == int4(5, 4, 8, 6));
  i4--;                           res[659] = (i4 == int4(4, 3, 7, 5));
  --i4;                           res[660] = (i4 == int4(3, 2, 6, 4));
  i4 = i41 + int4(2, 3, -8, -6);  res[661] = (i4 == int4(5, 6, 1, -2));
  i4 = i41 - int4(2, 1, -1, 6);   res[662] = (i4 == int4(1, 2, 10, -2));
  i4 = i41 * int4(3, 2, -2, 3);   res[663] = (i4 == int4(9, 6, -18, 12));
  i4 = i41 / int4(2, 3, -3, 5);   res[664] = (i4 == int4(1, 1, -3, 0));
  i4 = i41 % int4(2, 1, 3, 5);    res[665] = (i4 == int4(1, 0, 0, 4));
  i4 = i41 ^ int4(1, 2, 4, 3);    res[666] = (i4 == int4(2, 1, 13, 7));
  i4 = i41 | int4(4, 5, 4, 8);    res[667] = (i4 == int4(7, 7, 13, 12));
  i4 = i41 & int4(5, 4, 8, 6);    res[668] = (i4 == int4(1, 0, 8, 4));
  i4 = i41 << int4(2, 3, 2, 3);   res[669] = (i4 == int4(12, 24, 36, 32));
  i4 = i41 >> int4(1, 2, 3, 4);   res[670] = (i4 == int4(1, 0, 1, 0));
  i41 = int4(2, 5, -2, 4); i4 = ~i41;   res[671] = (i4 == int4(-3, -6, 1, -5));

  ulong1 ul1(0), ul11(0);    res[672] = (ul1 == ul11);
  ul1 += ulong1(3);          res[673] = (ul1 == 3);
  ul1 -= ulong1(2);          res[674] = (ul1 == 1);
  ul1 *= ulong1(4);          res[675] = (ul1 == 4);
  ul1 /= ulong1(2);          res[676] = (ul1 == 2);
  ul1 %= ulong1(3);          res[677] = (ul1 == 2);
  ul1 ^= ulong1(1);          res[678] = (ul1 == 3);
  ul1 |= ulong1(5);          res[679] = (ul1 == 7);
  ul1 &= ulong1(2);          res[680] = (ul1 == 2);
  ul1 <<= ulong1(4);         res[681] = (ul1 == 32);
  ul1 >>= ulong1(2);         res[682] = (ul1 == 8);
  ul11 = ulong1(8);          res[683] = (ul1 == ul11);
  ul11 = ulong1(3);          res[684] = (ul1 != ul11);
  ul1++;                     res[685] = (ul1 == 9);
  ++ul1;                     res[686] = (ul1 == 10);
  ul1--;                     res[687] = (ul1 == 9);
  --ul1;                     res[688] = (ul1 == 8);
  ul1 = ul11 + ulong1(2);    res[689] = (ul1 == 5);
  ul1 = ul11 - ulong1(2);    res[690] = (ul1 == 1);
  ul1 = ul11 * ulong1(2);    res[691] = (ul1 == 6);
  ul1 = ul11 / ulong1(2);    res[692] = (ul1 == 1);
  ul1 = ul11 % ulong1(2);    res[693] = (ul1 == 1);
  ul1 = ul11 ^ ulong1(1);    res[694] = (ul1 == 2);
  ul1 = ul11 | ulong1(6);    res[695] = (ul1 == 7);
  ul1 = ul11 & ulong1(6);    res[696] = (ul1 == 2);
  ul1 = ul11 << ulong1(2);   res[697] = (ul1 == 12);
  ul1 = ul11 >> ulong1(1);   res[698] = (ul1 == 1);
  ul11 = ulong1(2); ul1 = ~ul11;   res[699] = (ul1 == -3);

  ulong2 ul2(1, 2), ul21(1, 2);    res[700] = (ul2 == ul21);
  ul2 += ulong2(2, 3);          res[701] = (ul2 == ulong2(3, 5));
  ul2 -= ulong2(1, 2);          res[702] = (ul2 == ulong2(2, 3));
  ul2 *= ulong2(2, 4);          res[703] = (ul2 == ulong2(4, 12));
  ul2 /= ulong2(2, 2);          res[704] = (ul2 == ulong2(2, 6));
  ul2 %= ulong2(3, 4);          res[705] = (ul2 == ulong2(2, 2));
  ul2 ^= ulong2(1, 3);          res[706] = (ul2 == ulong2(3, 1));
  ul2 |= ulong2(4, 5);          res[707] = (ul2 == ulong2(7, 5));
  ul2 &= ulong2(3, 4);          res[708] = (ul2 == ulong2(3, 4));
  ul2 <<= ulong2(2, 3);         res[709] = (ul2 == ulong2(12, 32));
  ul2 >>= ulong2(3, 4);         res[710] = (ul2 == ulong2(1, 2));
  ul21 = ulong2(1, 2);          res[711] = (ul2 == ul21);
  ul21 = ulong2(3, 3);          res[712] = (ul2 != ul21);
  ul2++;                        res[713] = (ul2 == ulong2(2, 3));
  ++ul2;                        res[714] = (ul2 == ulong2(3, 4));
  ul2--;                        res[715] = (ul2 == ulong2(2, 3));
  --ul2;                        res[716] = (ul2 == ulong2(1, 2));
  ul2 = ul21 + ulong2(2, 3);    res[717] = (ul2 == ulong2(5, 6));
  ul2 = ul21 - ulong2(2, 1);    res[718] = (ul2 == ulong2(1, 2));
  ul2 = ul21 * ulong2(3, 2);    res[719] = (ul2 == ulong2(9, 6));
  ul2 = ul21 / ulong2(2, 3);    res[720] = (ul2 == ulong2(1, 1));
  ul2 = ul21 % ulong2(2, 1);    res[721] = (ul2 == ulong2(1, 0));
  ul2 = ul21 ^ ulong2(1, 2);    res[722] = (ul2 == ulong2(2, 1));
  ul2 = ul21 | ulong2(5, 6);    res[723] = (ul2 == ulong2(7, 7));
  ul2 = ul21 & ulong2(5, 4);    res[724] = (ul2 == ulong2(1, 0));
  ul2 = ul21 << ulong2(2, 3);   res[725] = (ul2 == ulong2(12, 24));
  ul2 = ul21 >> ulong2(1, 2);   res[726] = (ul2 == ulong2(1, 0));
  ul21 = ulong2(-4, 5); ul2 = ~ul21;   res[727] = (ul2 == ulong2(3, -6));

  ulong3 ul3(1, 2, 3), ul31(1, 2, 3);    res[728] = (ul3 == ul31);
  ul3 += ulong3(3, 5, 7);          res[729] = (ul3 == ulong3(4, 7, 10));
  ul3 -= ulong3(2, 4, 6);          res[730] = (ul3 == ulong3(2, 3, 4));
  ul3 *= ulong3(2, 3, 5);          res[731] = (ul3 == ulong3(4, 9, 20));
  ul3 /= ulong3(2, 3, 4);          res[732] = (ul3 == ulong3(2, 3, 5));
  ul3 %= ulong3(3, 4, 3);          res[733] = (ul3 == ulong3(2, 3, 2));
  ul3 ^= ulong3(1, 2, 4);          res[734] = (ul3 == ulong3(3, 1, 6));
  ul3 |= ulong3(4, 5, 3);          res[735] = (ul3 == ulong3(7, 5, 7));
  ul3 &= ulong3(3, 4, 3);          res[736] = (ul3 == ulong3(3, 4, 3));
  ul3 <<= ulong3(2, 3, 4);         res[737] = (ul3 == ulong3(12, 32, 48));
  ul3 >>= ulong3(2, 4, 3);         res[738] = (ul3 == ulong3(3, 2, 6));
  ul31 = ulong3(3, 2, 6);          res[739] = (ul3 == ul31);
  ul31 = ulong3(3, 3, 9);          res[740] = (ul3 != ul31);
  ul3++;                           res[741] = (ul3 == ulong3(4, 3, 7));
  ++ul3;                           res[742] = (ul3 == ulong3(5, 4, 8));
  ul3--;                           res[743] = (ul3 == ulong3(4, 3, 7));
  --ul3;                           res[744] = (ul3 == ulong3(3, 2, 6));
  ul3 = ul31 + ulong3(2, 3, -8);   res[745] = (ul3 == ulong3(5, 6, 1));
  ul3 = ul31 - ulong3(2, 1, -1);   res[746] = (ul3 == ulong3(1, 2, 10));
  ul3 = ul31 * ulong3(3, 2, -2);   res[747] = (ul3 == ulong3(9, 6, -18));
  ul3 = ul31 / ulong3(2, 3, -3);   res[748] = (ul3 == ulong3(1, 1, 0));
  ul3 = ul31 % ulong3(2, 1, 3);    res[749] = (ul3 == ulong3(1, 0, 0));
  ul3 = ul31 ^ ulong3(1, 2, 4);    res[750] = (ul3 == ulong3(2, 1, 13));
  ul3 = ul31 | ulong3(4, 5, 4);    res[751] = (ul3 == ulong3(7, 7, 13));
  ul3 = ul31 & ulong3(5, 4, 8);    res[752] = (ul3 == ulong3(1, 0, 8));
  ul3 = ul31 << ulong3(2, 3, 2);   res[753] = (ul3 == ulong3(12, 24, 36));
  ul3 = ul31 >> ulong3(1, 2, 3);   res[754] = (ul3 == ulong3(1, 0, 1));
  ul31 = ulong3(6, 4, -3); ul3 = ~ul31;   res[755] = (ul3 == ulong3(-7, -5, 2));

  ulong4 ul4(1, 2, 3, 4), ul41(1, 2, 3, 4);    res[756] = (ul4 == ul41);
  ul4 += ulong4(2, 5, 3, 4);          res[757] = (ul4 == ulong4(3, 7, 6, 8));
  ul4 -= ulong4(1, 4, 2, 3);          res[758] = (ul4 == ulong4(2, 3, 4, 5));
  ul4 *= ulong4(2, 3, 5, 4);          res[759] = (ul4 == ulong4(4, 9, 20, 20));
  ul4 /= ulong4(2, 3, 4, 6);          res[760] = (ul4 == ulong4(2, 3, 5, 3));
  ul4 %= ulong4(3, 5, 6, 3);          res[761] = (ul4 == ulong4(2, 3, 5, 0));
  ul4 ^= ulong4(1, 2, 4, 2);          res[762] = (ul4 == ulong4(3, 1, 1, 2));
  ul4 |= ulong4(4, 5, 3, 4);          res[763] = (ul4 == ulong4(7, 5, 3, 6));
  ul4 &= ulong4(3, 4, 7, 3);          res[764] = (ul4 == ulong4(3, 4, 3, 2));
  ul4 <<= ulong4(2, 3, 4, 5);         res[765] = (ul4 == ulong4(12, 32, 48, 64));
  ul4 >>= ulong4(2, 3, 3, 5);         res[766] = (ul4 == ulong4(3, 4, 6, 2));
  ul41 = ulong4(3, 4, 6, 2);          res[767] = (ul4 == ul41);
  ul41 = ulong4(3, 3, 9, 4);          res[768] = (ul4 != ul41);
  ul4++;                              res[769] = (ul4 == ulong4(4, 5, 7, 3));
  ++ul4;                              res[770] = (ul4 == ulong4(5, 6, 8, 4));
  ul4--;                              res[771] = (ul4 == ulong4(4, 5, 7, 3));
  --ul4;                              res[772] = (ul4 == ulong4(3, 4, 6, 2));
  ul4 = ul41 + ulong4(2, 3, -8, -6);  res[773] = (ul4 == ulong4(5, 6, 1, -2));
  ul4 = ul41 - ulong4(2, 1, -1, 6);   res[774] = (ul4 == ulong4(1, 2, 10, -2));
  ul4 = ul41 * ulong4(3, 2, -2, 3);   res[775] = (ul4 == ulong4(9, 6, -18, 12));
  ul4 = ul41 / ulong4(2, 3, 3, 5);    res[776] = (ul4 == ulong4(1, 1, 3, 0));
  ul4 = ul41 % ulong4(2, 1, 3, 5);    res[777] = (ul4 == ulong4(1, 0, 0, 4));
  ul4 = ul41 ^ ulong4(1, 2, 4, 3);    res[778] = (ul4 == ulong4(2, 1, 13, 7));
  ul4 = ul41 | ulong4(4, 5, 4, 8);    res[779] = (ul4 == ulong4(7, 7, 13, 12));
  ul4 = ul41 & ulong4(5, 4, 8, 6);    res[780] = (ul4 == ulong4(1, 0, 8, 4));
  ul4 = ul41 << ulong4(2, 3, 2, 3);   res[781] = (ul4 == ulong4(12, 24, 36, 32));
  ul4 = ul41 >> ulong4(1, 2, 3, 4);   res[782] = (ul4 == ulong4(1, 0, 1, 0));
  ul41 = ulong4(5, 3, -2, 4); ul4 = ~ul41;   res[783] = (ul4 == ulong4(-6, -4, 1, -5));

  long1 l1(1), l11(1);    res[784] = (l1 == l11);
  l1 += long1(1);         res[785] = (l1 == 2);
  l1 -= long1(1);         res[786] = (l1 == 1);
  l1 *= long1(3);         res[787] = (l1 == 3);
  l1 /= long1(3);         res[788] = (l1 == 1);
  l1 %= long1(4);         res[789] = (l1 == 1);
  l1 ^= long1(6);         res[790] = (l1 == 7);
  l1 |= long1(9);         res[791] = (l1 == 15);
  l1 &= long1(3);         res[792] = (l1 == 3);
  l1 <<= long1(4);        res[793] = (l1 == 48);
  l1 >>= long1(3);        res[794] = (l1 == 6);
  l11 = long1(6);         res[795] = (l1 == l11);
  l11 = long1(3);         res[796] = (l1 != l11);
  l1++;                   res[797] = (l1 == 7);
  ++l1;                   res[798] = (l1 == 8);
  l1--;                   res[799] = (l1 == 7);
  --l1;                   res[800] = (l1 == 6);
  l1 = l11 + long1(4);    res[801] = (l1 == 7);
  l1 = l11 - long1(2);    res[802] = (l1 == 1);
  l1 = l11 * long1(3);    res[803] = (l1 == 9);
  l1 = l11 / long1(2);    res[804] = (l1 == 1);
  l1 = l11 % long1(2);    res[805] = (l1 == 1);
  l1 = l11 ^ long1(1);    res[806] = (l1 == 2);
  l1 = l11 | long1(6);    res[807] = (l1 == 7);
  l1 = l11 & long1(6);    res[808] = (l1 == 2);
  l1 = l11 << long1(3);   res[809] = (l1 == 24);
  l1 = l11 >> long1(3);   res[810] = (l1 == 0);
  l11 = long1(2); l1 = ~l11;   res[811] = (l1 == -3);

  long2 l2(1, 2), l21(1, 2);   res[812] = (l2 == l21);
  l2 += long2(2, 3);           res[813] = (l2 == long2(3, 5));
  l2 -= long2(1, 2);           res[814] = (l2 == long2(2, 3));
  l2 *= long2(2, 4);           res[815] = (l2 == long2(4, 12));
  l2 /= long2(2, 2);           res[816] = (l2 == long2(2, 6));
  l2 %= long2(3, 4);           res[817] = (l2 == long2(2, 2));
  l2 ^= long2(1, 3);           res[818] = (l2 == long2(3, 1));
  l2 |= long2(4, 5);           res[819] = (l2 == long2(7, 5));
  l2 &= long2(3, 4);           res[820] = (l2 == long2(3, 4));
  l2 <<= long2(2, 3);          res[821] = (l2 == long2(12, 32));
  l2 >>= long2(2, 4);          res[822] = (l2 == long2(3, 2));
  l21 = long2(3, 2);           res[823] = (l2 == l21);
  l21 = long2(3, 3);           res[824] = (l2 != l21);
  l2++;                        res[825] = (l2 == long2(4, 3));
  ++l2;                        res[826] = (l2 == long2(5, 4));
  l2--;                        res[827] = (l2 == long2(4, 3));
  --l2;                        res[828] = (l2 == long2(3, 2));
  l2 = l21 + long2(2, 3);      res[829] = (l2 == long2(5, 6));
  l2 = l21 - long2(2, 1);      res[830] = (l2 == long2(1, 2));
  l2 = l21 * long2(3, 2);      res[831] = (l2 == long2(9, 6));
  l2 = l21 / long2(2, 3);      res[832] = (l2 == long2(1, 1));
  l2 = l21 % long2(2, 1);      res[833] = (l2 == long2(1, 0));
  l2 = l21 ^ long2(1, 2);      res[834] = (l2 == long2(2, 1));
  l2 = l21 | long2(4, 5);      res[835] = (l2 == long2(7, 7));
  l2 = l21 & long2(5, 4);      res[836] = (l2 == long2(1, 0));
  l2 = l21 << long2(2, 3);     res[837] = (l2 == long2(12, 24));
  l2 = l21 >> long2(1, 2);     res[838] = (l2 == long2(1, 0));
  l21 = long2(3, -2); l2 = ~l21;   res[839] = (l2 == long2(-4, 1));

  long3 l3(1, 2, 3), l31(1, 2, 3);    res[840] = (l3 == l31);
  l3 += long3(2, 3, 4);         res[841] = (l3 == long3(3, 5, 7));
  l3 -= long3(1, 2, 3);         res[842] = (l3 == long3(2, 3, 4));
  l3 *= long3(2, 4, -5);        res[843] = (l3 == long3(4, 12, -20));
  l3 /= long3(2, 2, -4);        res[844] = (l3 == long3(2, 6, 5));
  l3 %= long3(3, 4, 6);         res[845] = (l3 == long3(2, 2, 5));
  l3 ^= long3(1, 3, 4);         res[846] = (l3 == long3(3, 1, 1));
  l3 |= long3(4, 5, 3);         res[847] = (l3 == long3(7, 5, 3));
  l3 &= long3(3, 4, 7);         res[848] = (l3 == long3(3, 4, 3));
  l3 <<= long3(2, 3, 4);        res[849] = (l3 == long3(12, 32, 48));
  l3 >>= long3(2, 4, 3);        res[850] = (l3 == long3(3, 2, 6));
  l31 = long3(3, 2, 6);         res[851] = (l3 == l31);
  l31 = long3(3, 3, 9);         res[852] = (l3 != l31);
  l3++;                         res[853] = (l3 == long3(4, 3, 7));
  ++l3;                         res[854] = (l3 == long3(5, 4, 8));
  l3--;                         res[855] = (l3 == long3(4, 3, 7));
  --l3;                         res[856] = (l3 == long3(3, 2, 6));
  l3 = l31 + long3(2, 3, -8);   res[857] = (l3 == long3(5, 6, 1));
  l3 = l31 - long3(2, 1, -1);   res[858] = (l3 == long3(1, 2, 10));
  l3 = l31 * long3(3, 2, -2);   res[859] = (l3 == long3(9, 6, -18));
  l3 = l31 / long3(2, 3, -3);   res[860] = (l3 == long3(1, 1, -3));
  l3 = l31 % long3(2, 1, 3);    res[861] = (l3 == long3(1, 0, 0));
  l3 = l31 ^ long3(1, 2, 4);    res[862] = (l3 == long3(2, 1, 13));
  l3 = l31 | long3(4, 5, 4);    res[863] = (l3 == long3(7, 7, 13));
  l3 = l31 & long3(5, 4, 8);    res[864] = (l3 == long3(1, 0, 8));
  l3 = l31 << long3(2, 3, 2);   res[865] = (l3 == long3(12, 24, 36));
  l3 = l31 >> long3(1, 2, 3);   res[866] = (l3 == long3(1, 0, 1));
  l31 = long3(1, -4, -7); l3 = ~l31;   res[867] = (l3 == long3(-2, 3, 6));

  long4 l4(1, 2, 3, 4), l41(1, 2, 3, 4);    res[868] = (l4 == l41);
  l4 += long4(2, 3, 4, 5);         res[869] = (l4 == long4(3, 5, 7, 9));
  l4 -= long4(1, 2, 3, 4);         res[870] = (l4 == long4(2, 3, 4, 5));
  l4 *= long4(2, 4, -5, 3);        res[871] = (l4 == long4(4, 12, -20, 15));
  l4 /= long4(2, 2, -4, 4);        res[872] = (l4 == long4(2, 6, 5, 3));
  l4 %= long4(3, 4, 6, 3);         res[873] = (l4 == long4(2, 2, 5, 0));
  l4 ^= long4(1, 3, 4, 2);         res[874] = (l4 == long4(3, 1, 1, 2));
  l4 |= long4(4, 5, 3, 4);         res[875] = (l4 == long4(7, 5, 3, 6));
  l4 &= long4(3, 4, 7, 3);         res[876] = (l4 == long4(3, 4, 3, 2));
  l4 <<= long4(2, 3, 4, 5);        res[877] = (l4 == long4(12, 32, 48, 64));
  l4 >>= long4(2, 4, 3, 4);        res[878] = (l4 == long4(3, 2, 6, 4));
  l41 = long4(3, 2, 6, 4);         res[879] = (l4 == l41);
  l41 = long4(3, 3, 9, 4);         res[880] = (l4 != l41);
  l4++;                            res[881] = (l4 == long4(4, 3, 7, 5));
  ++l4;                            res[882] = (l4 == long4(5, 4, 8, 6));
  l4--;                            res[883] = (l4 == long4(4, 3, 7, 5));
  --l4;                            res[884] = (l4 == long4(3, 2, 6, 4));
  l4 = l41 + long4(2, 3, -8, -6);  res[885] = (l4 == long4(5, 6, 1, -2));
  l4 = l41 - long4(2, 1, -1, 6);   res[886] = (l4 == long4(1, 2, 10, -2));
  l4 = l41 * long4(3, 2, -2, 3);   res[887] = (l4 == long4(9, 6, -18, 12));
  l4 = l41 / long4(2, 3, -3, 5);   res[888] = (l4 == long4(1, 1, -3, 0));
  l4 = l41 % long4(2, 1, 3, 5);    res[889] = (l4 == long4(1, 0, 0, 4));
  l4 = l41 ^ long4(1, 2, 4, 3);    res[890] = (l4 == long4(2, 1, 13, 7));
  l4 = l41 | long4(4, 5, 4, 8);    res[891] = (l4 == long4(7, 7, 13, 12));
  l4 = l41 & long4(5, 4, 8, 6);    res[892] = (l4 == long4(1, 0, 8, 4));
  l4 = l41 << long4(2, 3, 2, 3);   res[893] = (l4 == long4(12, 24, 36, 32));
  l4 = l41 >> long4(1, 2, 3, 4);   res[894] = (l4 == long4(1, 0, 1, 0));
  l41 = long4(4, 8, -8, 2); l4 = ~l41;   res[895] = (l4 == long4(-5, -9, 7, -3));

  ulonglong1 ull1(0), ull11(0);    res[896] = (ull1 == ull11);
  ull1 += ulonglong1(3);           res[897] = (ull1 == 3);
  ull1 -= ulonglong1(2);           res[898] = (ull1 == 1);
  ull1 *= ulonglong1(4);           res[899] = (ull1 == 4);
  ull1 /= ulonglong1(2);           res[900] = (ull1 == 2);
  ull1 %= ulonglong1(3);           res[901] = (ull1 == 2);
  ull1 ^= ulonglong1(1);           res[902] = (ull1 == 3);
  ull1 |= ulonglong1(5);           res[903] = (ull1 == 7);
  ull1 &= ulonglong1(2);           res[904] = (ull1 == 2);
  ull1 <<= ulonglong1(4);          res[905] = (ull1 == 32);
  ull1 >>= ulonglong1(2);          res[906] = (ull1 == 8);
  ull11 = ulonglong1(8);           res[907] = (ull1 == ull11);
  ull11 = ulonglong1(3);           res[908] = (ull1 != ull11);
  ull1++;                          res[909] = (ull1 == 9);
  ++ull1;                          res[910] = (ull1 == 10);
  ull1--;                          res[911] = (ull1 == 9);
  --ull1;                          res[912] = (ull1 == 8);
  ull1 = ull11 + ulonglong1(2);    res[913] = (ull1 == 5);
  ull1 = ull11 - ulonglong1(2);    res[914] = (ull1 == 1);
  ull1 = ull11 * ulonglong1(2);    res[915] = (ull1 == 6);
  ull1 = ull11 / ulonglong1(2);    res[916] = (ull1 == 1);
  ull1 = ull11 % ulonglong1(2);    res[917] = (ull1 == 1);
  ull1 = ull11 ^ ulonglong1(1);    res[918] = (ull1 == 2);
  ull1 = ull11 | ulonglong1(6);    res[919] = (ull1 == 7);
  ull1 = ull11 & ulonglong1(6);    res[920] = (ull1 == 2);
  ull1 = ull11 << ulonglong1(2);   res[921] = (ull1 == 12);
  ull1 = ull11 >> ulonglong1(1);   res[922] = (ull1 == 1);
  ull11 = ulonglong1(-9); ull1 = ~ull11;   res[923] = (ull1 == 8);

  ulonglong2 ull2(1, 2), ull21(1, 2);     res[924] = (ull2 == ull21);
  ull2 += ulonglong2(2, 3);           res[925] = (ull2 == ulonglong2(3, 5));
  ull2 -= ulonglong2(1, 2);           res[926] = (ull2 == ulonglong2(2, 3));
  ull2 *= ulonglong2(2, 4);           res[927] = (ull2 == ulonglong2(4, 12));
  ull2 /= ulonglong2(2, 2);           res[928] = (ull2 == ulonglong2(2, 6));
  ull2 %= ulonglong2(3, 4);           res[929] = (ull2 == ulonglong2(2, 2));
  ull2 ^= ulonglong2(1, 3);           res[930] = (ull2 == ulonglong2(3, 1));
  ull2 |= ulonglong2(4, 5);           res[931] = (ull2 == ulonglong2(7, 5));
  ull2 &= ulonglong2(3, 4);           res[932] = (ull2 == ulonglong2(3, 4));
  ull2 <<= ulonglong2(2, 3);          res[933] = (ull2 == ulonglong2(12, 32));
  ull2 >>= ulonglong2(2, 4);          res[934] = (ull2 == ulonglong2(3, 2));
  ull21 = ulonglong2(3, 2);           res[935] = (ull2 == ull21);
  ull21 = ulonglong2(3, 3);           res[936] = (ull2 != ull21);
  ull2++;                             res[937] = (ull2 == ulonglong2(4, 3));
  ++ull2;                             res[938] = (ull2 == ulonglong2(5, 4));
  ull2--;                             res[939] = (ull2 == ulonglong2(4, 3));
  --ull2;                             res[940] = (ull2 == ulonglong2(3, 2));
  ull2 = ull21 + ulonglong2(2, 3);    res[941] = (ull2 == ulonglong2(5, 6));
  ull2 = ull21 - ulonglong2(2, 1);    res[942] = (ull2 == ulonglong2(1, 2));
  ull2 = ull21 * ulonglong2(3, 2);    res[943] = (ull2 == ulonglong2(9, 6));
  ull2 = ull21 / ulonglong2(2, 3);    res[944] = (ull2 == ulonglong2(1, 1));
  ull2 = ull21 % ulonglong2(2, 1);    res[945] = (ull2 == ulonglong2(1, 0));
  ull2 = ull21 ^ ulonglong2(1, 2);    res[946] = (ull2 == ulonglong2(2, 1));
  ull2 = ull21 | ulonglong2(4, 5);    res[947] = (ull2 == ulonglong2(7, 7));
  ull2 = ull21 & ulonglong2(5, 4);    res[948] = (ull2 == ulonglong2(1, 0));
  ull2 = ull21 << ulonglong2(2, 3);   res[949] = (ull2 == ulonglong2(12, 24));
  ull2 = ull21 >> ulonglong2(1, 2);   res[950] = (ull2 == ulonglong2(1, 0));
  ull21 = ulonglong2(3, -8); ull2 = ~ull21;   res[951] = (ull2 == ulonglong2(-4, 7));

  ulonglong3 ull3(1, 2, 3), ull31(1, 2, 3);    res[952] = (ull3 == ull31);
  ull3 += ulonglong3(2, 3, 4);           res[953] = (ull3 == ulonglong3(3, 5, 7));
  ull3 -= ulonglong3(1, 2, 3);           res[954] = (ull3 == ulonglong3(2, 3, 4));
  ull3 *= ulonglong3(2, 4, 5);           res[955] = (ull3 == ulonglong3(4, 12, 20));
  ull3 /= ulonglong3(2, 2, 4);           res[956] = (ull3 == ulonglong3(2, 6, 5));
  ull3 %= ulonglong3(3, 4, 6);           res[957] = (ull3 == ulonglong3(2, 2, 5));
  ull3 ^= ulonglong3(1, 3, 4);           res[958] = (ull3 == ulonglong3(3, 1, 1));
  ull3 |= ulonglong3(4, 5, 3);           res[959] = (ull3 == ulonglong3(7, 5, 3));
  ull3 &= ulonglong3(3, 4, 7);           res[960] = (ull3 == ulonglong3(3, 4, 3));
  ull3 <<= ulonglong3(2, 3, 4);          res[961] = (ull3 == ulonglong3(12, 32, 48));
  ull3 >>= ulonglong3(2, 4, 3);          res[962] = (ull3 == ulonglong3(3, 2, 6));
  ull31 = ulonglong3(3, 2, 6);           res[963] = (ull3 == ull31);
  ull31 = ulonglong3(3, 3, 9);           res[964] = (ull3 != ull31);
  ull3++;                                res[965] = (ull3 == ulonglong3(4, 3, 7));
  ++ull3;                                res[966] = (ull3 == ulonglong3(5, 4, 8));
  ull3--;                                res[967] = (ull3 == ulonglong3(4, 3, 7));
  --ull3;                                res[968] = (ull3 == ulonglong3(3, 2, 6));
  ull3 = ull31 + ulonglong3(2, 3, -8);   res[969] = (ull3 == ulonglong3(5, 6, 1));
  ull3 = ull31 - ulonglong3(2, 1, -1);   res[970] = (ull3 == ulonglong3(1, 2, 10));
  ull3 = ull31 * ulonglong3(3, 2, -2);   res[971] = (ull3 == ulonglong3(9, 6, -18));
  ull3 = ull31 / ulonglong3(2, 3, -3);   res[972] = (ull3 == ulonglong3(1, 1, 0));
  ull3 = ull31 % ulonglong3(2, 1, 3);    res[973] = (ull3 == ulonglong3(1, 0, 0));
  ull3 = ull31 ^ ulonglong3(1, 2, 4);    res[974] = (ull3 == ulonglong3(2, 1, 13));
  ull3 = ull31 | ulonglong3(4, 5, 4);    res[975] = (ull3 == ulonglong3(7, 7, 13));
  ull3 = ull31 & ulonglong3(5, 4, 8);    res[976] = (ull3 == ulonglong3(1, 0, 8));
  ull3 = ull31 << ulonglong3(2, 3, 2);   res[977] = (ull3 == ulonglong3(12, 24, 36));
  ull3 = ull31 >> ulonglong3(1, 2, 3);   res[978] = (ull3 == ulonglong3(1, 0, 1));
  ull31 = ulonglong3(5, 8, -6); ull3 = ~ull31;   res[979] = (ull3 == ulonglong3(-6, -9, 5));

  ulonglong4 ull4(1, 2, 3, 4), ull41(1, 2, 3, 4);    res[980] = (ull4 == ull41);
  ull4 += ulonglong4(2, 3, 4, 5);           res[981] = (ull4 == ulonglong4(3, 5, 7, 9));
  ull4 -= ulonglong4(1, 2, 3, 4);           res[982] = (ull4 == ulonglong4(2, 3, 4, 5));
  ull4 *= ulonglong4(2, 4, 5, 3);           res[983] = (ull4 == ulonglong4(4, 12, 20, 15));
  ull4 /= ulonglong4(2, 2, 4, 4);           res[984] = (ull4 == ulonglong4(2, 6, 5, 3));
  ull4 %= ulonglong4(3, 4, 6, 3);           res[985] = (ull4 == ulonglong4(2, 2, 5, 0));
  ull4 ^= ulonglong4(1, 3, 4, 2);           res[986] = (ull4 == ulonglong4(3, 1, 1, 2));
  ull4 |= ulonglong4(4, 5, 3, 4);           res[987] = (ull4 == ulonglong4(7, 5, 3, 6));
  ull4 &= ulonglong4(3, 4, 7, 3);           res[988] = (ull4 == ulonglong4(3, 4, 3, 2));
  ull4 <<= ulonglong4(2, 3, 4, 5);          res[989] = (ull4 == ulonglong4(12, 32, 48, 64));
  ull4 >>= ulonglong4(2, 4, 3, 4);          res[990] = (ull4 == ulonglong4(3, 2, 6, 4));
  ull41 = ulonglong4(3, 2, 6, 4);           res[991] = (ull4 == ull41);
  ull41 = ulonglong4(3, 3, 9, 4);           res[992] = (ull4 != ull41);
  ull4++;                                   res[993] = (ull4 == ulonglong4(4, 3, 7, 5));
  ++ull4;                                   res[994] = (ull4 == ulonglong4(5, 4, 8, 6));
  ull4--;                                   res[995] = (ull4 == ulonglong4(4, 3, 7, 5));
  --ull4;                                   res[996] = (ull4 == ulonglong4(3, 2, 6, 4));
  ull4 = ull41 + ulonglong4(2, 3, -8, -6);  res[997] = (ull4 == ulonglong4(5, 6, 1, -2));
  ull4 = ull41 - ulonglong4(2, 1, -1, 6);   res[998] = (ull4 == ulonglong4(1, 2, 10, -2));
  ull4 = ull41 * ulonglong4(3, 2, -2, 3);   res[999] = (ull4 == ulonglong4(9, 6, -18, 12));
  ull4 = ull41 / ulonglong4(2, 3, 3, 5);    res[1000] = (ull4 == ulonglong4(1, 1, 3, 0));
  ull4 = ull41 % ulonglong4(2, 1, 3, 5);    res[1001] = (ull4 == ulonglong4(1, 0, 0, 4));
  ull4 = ull41 ^ ulonglong4(1, 2, 4, 3);    res[1002] = (ull4 == ulonglong4(2, 1, 13, 7));
  ull4 = ull41 | ulonglong4(4, 5, 4, 8);    res[1003] = (ull4 == ulonglong4(7, 7, 13, 12));
  ull4 = ull41 & ulonglong4(5, 4, 8, 6);    res[1004] = (ull4 == ulonglong4(1, 0, 8, 4));
  ull4 = ull41 << ulonglong4(2, 3, 2, 3);   res[1005] = (ull4 == ulonglong4(12, 24, 36, 32));
  ull4 = ull41 >> ulonglong4(1, 2, 3, 4);   res[1006] = (ull4 == ulonglong4(1, 0, 1, 0));
  ull41 = ulonglong4(3, 5, -5, 7); ull4 = ~ull41;   res[1007] = (ull4 == ulonglong4(-4, -6, 4, -8));

  longlong1 ll1(1), ll11(1);   res[1008] = (ll1 == ll11);
  ll1 += longlong1(1);         res[1009] = (ll1 == 2);
  ll1 -= longlong1(1);         res[1010] = (ll1 == 1);
  ll1 *= longlong1(3);         res[1011] = (ll1 == 3);
  ll1 /= longlong1(3);         res[1012] = (ll1 == 1);
  ll1 %= longlong1(4);         res[1013] = (ll1 == 1);
  ll1 ^= longlong1(6);         res[1014] = (ll1 == 7);
  ll1 |= longlong1(9);         res[1015] = (ll1 == 15);
  ll1 &= longlong1(3);         res[1016] = (ll1 == 3);
  ll1 <<= longlong1(4);        res[1017] = (ll1 == 48);
  ll1 >>= longlong1(3);        res[1018] = (ll1 == 6);
  ll11 = longlong1(6);         res[1019] = (ll1 == ll11);
  ll11 = longlong1(3);         res[1020] = (ll1 != ll11);
  ll1++;                       res[1021] = (ll1 == 7);
  ++ll1;                       res[1022] = (ll1 == 8);
  ll1--;                       res[1023] = (ll1 == 7);
  --ll1;                       res[1024] = (ll1 == 6);
  ll1 = ll11 + longlong1(4);   res[1025] = (ll1 == 7);
  ll1 = ll11 - longlong1(2);   res[1026] = (ll1 == 1);
  ll1 = ll11 * longlong1(3);   res[1027] = (ll1 == 9);
  ll1 = ll11 / longlong1(2);   res[1028] = (ll1 == 1);
  ll1 = ll11 % longlong1(2);   res[1029] = (ll1 == 1);
  ll1 = ll11 ^ longlong1(1);   res[1030] = (ll1 == 2);
  ll1 = ll11 | longlong1(6);   res[1031] = (ll1 == 7);
  ll1 = ll11 & longlong1(6);   res[1032] = (ll1 == 2);
  ll1 = ll11 << longlong1(3);  res[1033] = (ll1 == 24);
  ll1 = ll11 >> longlong1(3);  res[1034] = (ll1 == 0);
  ll11 = longlong1(7); ll1 = ~ll11;   res[1035] = (ll1 == -8);

  longlong2 ll2(1, 2), ll21(1, 2);    res[1036] = (ll2 == ll21);
  ll2 += longlong2(2, 3);         res[1037] = (ll2 == longlong2(3, 5));
  ll2 -= longlong2(1, 2);         res[1038] = (ll2 == longlong2(2, 3));
  ll2 *= longlong2(2, 4);         res[1039] = (ll2 == longlong2(4, 12));
  ll2 /= longlong2(2, 2);         res[1040] = (ll2 == longlong2(2, 6));
  ll2 %= longlong2(3, 4);         res[1041] = (ll2 == longlong2(2, 2));
  ll2 ^= longlong2(1, 3);         res[1042] = (ll2 == longlong2(3, 1));
  ll2 |= longlong2(4, 5);         res[1043] = (ll2 == longlong2(7, 5));
  ll2 &= longlong2(3, 4);         res[1044] = (ll2 == longlong2(3, 4));
  ll2 <<= longlong2(2, 3);        res[1045] = (ll2 == longlong2(12, 32));
  ll2 >>= longlong2(2, 4);        res[1046] = (ll2 == longlong2(3, 2));
  ll21 = longlong2(3, 2);         res[1047] = (ll2 == ll21);
  ll21 = longlong2(3, 3);         res[1048] = (ll2 != ll21);
  ll2++;                          res[1049] = (ll2 == longlong2(4, 3));
  ++ll2;                          res[1050] = (ll2 == longlong2(5, 4));
  ll2--;                          res[1051] = (ll2 == longlong2(4, 3));
  --ll2;                          res[1052] = (ll2 == longlong2(3, 2));
  ll2 = ll21 + longlong2(2, 3);   res[1053] = (ll2 == longlong2(5, 6));
  ll2 = ll21 - longlong2(2, 1);   res[1054] = (ll2 == longlong2(1, 2));
  ll2 = ll21 * longlong2(3, 2);   res[1055] = (ll2 == longlong2(9, 6));
  ll2 = ll21 / longlong2(2, 3);   res[1056] = (ll2 == longlong2(1, 1));
  ll2 = ll21 % longlong2(2, 1);   res[1057] = (ll2 == longlong2(1, 0));
  ll2 = ll21 ^ longlong2(1, 2);   res[1058] = (ll2 == longlong2(2, 1));
  ll2 = ll21 | longlong2(4, 5);   res[1059] = (ll2 == longlong2(7, 7));
  ll2 = ll21 & longlong2(5, 4);   res[1060] = (ll2 == longlong2(1, 0));
  ll2 = ll21 << longlong2(2, 3);  res[1061] = (ll2 == longlong2(12, 24));
  ll2 = ll21 >> longlong2(1, 2);  res[1062] = (ll2 == longlong2(1, 0));
  ll21 = longlong2(5, 1); ll2 = ~ll21;   res[1063] = (ll2 == longlong2(-6, -2));

  longlong3 ll3(1, 2, 3), ll31(1, 2, 3);    res[1064] = (ll3 == ll31);
  ll3 += longlong3(2, 3, 4);          res[1065] = (ll3 == longlong3(3, 5, 7));
  ll3 -= longlong3(1, 2, 3);          res[1066] = (ll3 == longlong3(2, 3, 4));
  ll3 *= longlong3(2, 4, -5);         res[1067] = (ll3 == longlong3(4, 12, -20));
  ll3 /= longlong3(2, 2, -4);         res[1068] = (ll3 == longlong3(2, 6, 5));
  ll3 %= longlong3(3, 4, 6);          res[1069] = (ll3 == longlong3(2, 2, 5));
  ll3 ^= longlong3(1, 3, 4);          res[1070] = (ll3 == longlong3(3, 1, 1));
  ll3 |= longlong3(4, 5, 3);          res[1071] = (ll3 == longlong3(7, 5, 3));
  ll3 &= longlong3(3, 4, 7);          res[1072] = (ll3 == longlong3(3, 4, 3));
  ll3 <<= longlong3(2, 3, 4);         res[1073] = (ll3 == longlong3(12, 32, 48));
  ll3 >>= longlong3(2, 4, 3);         res[1074] = (ll3 == longlong3(3, 2, 6));
  ll31 = longlong3(3, 2, 6);          res[1075] = (ll3 == ll31);
  ll31 = longlong3(3, 3, 9);          res[1076] = (ll3 != ll31);
  ll3++;                              res[1077] = (ll3 == longlong3(4, 3, 7));
  ++ll3;                              res[1078] = (ll3 == longlong3(5, 4, 8));
  ll3--;                              res[1079] = (ll3 == longlong3(4, 3, 7));
  --ll3;                              res[1080] = (ll3 == longlong3(3, 2, 6));
  ll3 = ll31 + longlong3(2, 3, -8);   res[1081] = (ll3 == longlong3(5, 6, 1));
  ll3 = ll31 - longlong3(2, 1, -1);   res[1082] = (ll3 == longlong3(1, 2, 10));
  ll3 = ll31 * longlong3(3, 2, -2);   res[1083] = (ll3 == longlong3(9, 6, -18));
  ll3 = ll31 / longlong3(2, 3, -3);   res[1084] = (ll3 == longlong3(1, 1, -3));
  ll3 = ll31 % longlong3(2, 1, 3);    res[1085] = (ll3 == longlong3(1, 0, 0));
  ll3 = ll31 ^ longlong3(1, 2, 4);    res[1086] = (ll3 == longlong3(2, 1, 13));
  ll3 = ll31 | longlong3(4, 5, 4);    res[1087] = (ll3 == longlong3(7, 7, 13));
  ll3 = ll31 & longlong3(5, 4, 8);    res[1088] = (ll3 == longlong3(1, 0, 8));
  ll3 = ll31 << longlong3(2, 3, 2);   res[1089] = (ll3 == longlong3(12, 24, 36));
  ll3 = ll31 >> longlong3(1, 2, 3);   res[1090] = (ll3 == longlong3(1, 0, 1));
  ll31 = longlong3(1, -8, -5); ll3 = ~ll31;   res[1091] = (ll3 == longlong3(-2, 7, 4));

  longlong4 ll4(1, 2, 3, 4), ll41(1, 2, 3, 4);    res[1092] = (ll4 == ll41);
  ll4 += longlong4(2, 3, 4, 5);          res[1093] = (ll4 == longlong4(3, 5, 7, 9));
  ll4 -= longlong4(1, 2, 3, 4);          res[1094] = (ll4 == longlong4(2, 3, 4, 5));
  ll4 *= longlong4(2, 4, -5, 3);         res[1095] = (ll4 == longlong4(4, 12, -20, 15));
  ll4 /= longlong4(2, 2, -4, 4);         res[1096] = (ll4 == longlong4(2, 6, 5, 3));
  ll4 %= longlong4(3, 4, 6, 3);          res[1097] = (ll4 == longlong4(2, 2, 5, 0));
  ll4 ^= longlong4(1, 3, 4, 2);          res[1098] = (ll4 == longlong4(3, 1, 1, 2));
  ll4 |= longlong4(4, 5, 3, 4);          res[1099] = (ll4 == longlong4(7, 5, 3, 6));
  ll4 &= longlong4(3, 4, 7, 3);          res[1100] = (ll4 == longlong4(3, 4, 3, 2));
  ll4 <<= longlong4(2, 3, 4, 5);         res[1101] = (ll4 == longlong4(12, 32, 48, 64));
  ll4 >>= longlong4(2, 4, 3, 4);         res[1102] = (ll4 == longlong4(3, 2, 6, 4));
  ll41 = longlong4(3, 2, 6, 4);          res[1103] = (ll4 == ll41);
  ll41 = longlong4(3, 3, 9, 4);          res[1104] = (ll4 != ll41);
  ll4++;                                 res[1105] = (ll4 == longlong4(4, 3, 7, 5));
  ++ll4;                                 res[1106] = (ll4 == longlong4(5, 4, 8, 6));
  ll4--;                                 res[1107] = (ll4 == longlong4(4, 3, 7, 5));
  --ll4;                                 res[1108] = (ll4 == longlong4(3, 2, 6, 4));
  ll4 = ll41 + longlong4(2, 3, -8, -6);  res[1109] = (ll4 == longlong4(5, 6, 1, -2));
  ll4 = ll41 - longlong4(2, 1, -1, 6);   res[1110] = (ll4 == longlong4(1, 2, 10, -2));
  ll4 = ll41 * longlong4(3, 2, -2, 3);   res[1111] = (ll4 == longlong4(9, 6, -18, 12));
  ll4 = ll41 / longlong4(2, 3, -3, 5);   res[1112] = (ll4 == longlong4(1, 1, -3, 0));
  ll4 = ll41 % longlong4(2, 1, 3, 5);    res[1113] = (ll4 == longlong4(1, 0, 0, 4));
  ll4 = ll41 ^ longlong4(1, 2, 4, 3);    res[1114] = (ll4 == longlong4(2, 1, 13, 7));
  ll4 = ll41 | longlong4(4, 5, 4, 8);    res[1115] = (ll4 == longlong4(7, 7, 13, 12));
  ll4 = ll41 & longlong4(5, 4, 8, 6);    res[1116] = (ll4 == longlong4(1, 0, 8, 4));
  ll4 = ll41 << longlong4(2, 3, 2, 3);   res[1117] = (ll4 == longlong4(12, 24, 36, 32));
  ll4 = ll41 >> longlong4(1, 2, 3, 4);   res[1118] = (ll4 == longlong4(1, 0, 1, 0));
  ll41 = longlong4(4, 7, -3, 6); ll4 = ~ll41;  res[1119] = (ll4 == longlong4(-5, -8, 2, -7));

  float1 f1(1.1), f11(1.1);  res[1120] = isEqualFloat1(f1, f11);
  f1 += float1(1.1);         res[1121] = isEqualFloat1(f1, float1(2.2));
  f1 -= float1(1.1);         res[1122] = isEqualFloat1(f1, float1(1.1));
  f1 *= float1(3.1);         res[1123] = isEqualFloat1(f1, float1(3.41));
  f1 /= float1(3.1);         res[1124] = isEqualFloat1(f1, float1(1.1));
  f11 = float1(1.1);         res[1125] = isEqualFloat1(f1, f11);
  f11 = float1(3.1);         res[1126] = !isEqualFloat1(f1, f11);
  f1++;                      res[1127] = isEqualFloat1(f1, float1(2.1));
  ++f1;                      res[1128] = isEqualFloat1(f1, float1(3.1));
  f1--;                      res[1129] = isEqualFloat1(f1, float1(2.1));
  --f1;                      res[1130] = isEqualFloat1(f1, float1(1.1));
  f1 = f11 + float1(4.1);    res[1131] = isEqualFloat1(f1, float1(7.2));
  f1 = f11 - float1(2.1);    res[1132] = isEqualFloat1(f1, float1(1.0));
  f1 = f11 * float1(1.1);    res[1133] = isEqualFloat1(f1, float1(3.41));
  f1 = f11 / float1(1.0);    res[1134] = isEqualFloat1(f1, float1(3.1));

  float2 f2(1.1, 2.1), f21(1.1, 2.1);  res[1135] = isEqualFloat2(f2, f21);
  f2 += float2(1.1, 1.2);         res[1136] = isEqualFloat2(f2, float2(2.2, 3.3));
  f2 -= float2(1.1, 2.2);         res[1137] = isEqualFloat2(f2, float2(1.1, 1.1));
  f2 *= float2(3.1, 1.3);         res[1138] = isEqualFloat2(f2, float2(3.41, 1.43));
  f2 /= float2(3.1, 1.1);         res[1139] = isEqualFloat2(f2, float2(1.1, 1.3));
  f21 = float2(1.1, 1.3);         res[1140] = isEqualFloat2(f2, f21);
  f21 = float2(3.1, 1.3);         res[1141] = !isEqualFloat2(f2, f21);
  f2++;                           res[1142] = isEqualFloat2(f2, float2(2.1, 2.3));
  ++f2;                           res[1143] = isEqualFloat2(f2, float2(3.1, 3.3));
  f2--;                           res[1144] = isEqualFloat2(f2, float2(2.1, 2.3));
  --f2;                           res[1145] = isEqualFloat2(f2, float2(1.1, 1.3));
  f2 = f21 + float2(4.1, 2.1);    res[1146] = isEqualFloat2(f2, float2(7.2, 3.4));
  f2 = f21 - float2(2.1, 1.1);    res[1147] = isEqualFloat2(f2, float2(1.0, 0.2));
  f2 = f21 * float2(1.1, 2.1);    res[1148] = isEqualFloat2(f2, float2(3.41, 2.73));
  f2 = f21 / float2(3.1, 1.3);    res[1149] = isEqualFloat2(f2, float2(1.0, 1.0));

  float3 f3(1.1, 2.1, 3.1), f31(1.1, 2.1, 3.1);  res[1150] = isEqualFloat3(f3, f31);
  f3 += float3(1.1, 1.2, 1.3);       res[1151] = isEqualFloat3(f3, float3(2.2, 3.3, 4.4));
  f3 -= float3(1.1, 2.2, 3.2);       res[1152] = isEqualFloat3(f3, float3(1.1, 1.1, 1.2));
  f3 *= float3(3.1, 1.3, 2.1);       res[1153] = isEqualFloat3(f3, float3(3.41, 1.43, 2.52));
  f3 /= float3(3.1, 1.1, 1.2);       res[1154] = isEqualFloat3(f3, float3(1.1, 1.3, 2.1));
  f31 = float3(1.1, 1.3, 2.1);       res[1155] = isEqualFloat3(f3, f31);
  f31 = float3(3.1, 1.3, 2.4);       res[1156] = !isEqualFloat3(f3, f31);
  f3++;                              res[1157] = isEqualFloat3(f3, float3(2.1, 2.3, 3.1));
  ++f3;                              res[1158] = isEqualFloat3(f3, float3(3.1, 3.3, 4.1));
  f3--;                              res[1159] = isEqualFloat3(f3, float3(2.1, 2.3, 3.1));
  --f3;                              res[1160] = isEqualFloat3(f3, float3(1.1, 1.3, 2.1));
  f3 = f31 + float3(4.1, 2.1, 3.2);  res[1161] = isEqualFloat3(f3, float3(7.2, 3.4, 5.6));
  f3 = f31 - float3(2.1, 1.1, 1.2);  res[1162] = isEqualFloat3(f3, float3(1.0, 0.2, 1.2));
  f3 = f31 * float3(1.1, 2.1, 3.1);  res[1163] = isEqualFloat3(f3, float3(3.41, 2.73, 7.44));
  f3 = f31 / float3(3.1, 1.3, 2.0);  res[1164] = isEqualFloat3(f3, float3(1.0, 1.0, 1.2));

  float4 f4(1.1, 2.1, 3.1, 4.1), f41(1.1, 2.1, 3.1, 4.1);  res[1165] = isEqualFloat4(f4, f41);
  f4 += float4(1.1, 1.2, 1.3, 1.4);       res[1166] = isEqualFloat4(f4, float4(2.2, 3.3, 4.4, 5.5));
  f4 -= float4(1.1, 2.2, 3.2, 4.2);       res[1167] = isEqualFloat4(f4, float4(1.1, 1.1, 1.2, 1.3));
  f4 *= float4(3.1, 1.3, 2.1, 4.2);       res[1168] = isEqualFloat4(f4, float4(3.41, 1.43, 2.52, 5.46));
  f4 /= float4(3.1, 1.1, 1.2, 2.1);       res[1169] = isEqualFloat4(f4, float4(1.1, 1.3, 2.1, 2.6));
  f41 = float4(1.1, 1.3, 2.1, 2.6);       res[1170] = isEqualFloat4(f4, f41);
  f41 = float4(3.1, 1.3, 2.4, 3.6);       res[1171] = !isEqualFloat4(f4, f41);
  f4++;                                   res[1172] = isEqualFloat4(f4, float4(2.1, 2.3, 3.1, 3.6));
  ++f4;                                   res[1173] = isEqualFloat4(f4, float4(3.1, 3.3, 4.1, 4.6));
  f4--;                                   res[1174] = isEqualFloat4(f4, float4(2.1, 2.3, 3.1, 3.6));
  --f4;                                   res[1175] = isEqualFloat4(f4, float4(1.1, 1.3, 2.1, 2.6));
  f4 = f41 + float4(4.1, 2.1, 3.2, 1.5);  res[1176] = isEqualFloat4(f4, float4(7.2, 3.4, 5.6, 5.1));
  f4 = f41 - float4(2.1, 1.1, 1.2, 2.3);  res[1177] = isEqualFloat4(f4, float4(1.0, 0.2, 1.2, 1.3));
  f4 = f41 * float4(1.1, 2.1, 3.1, 2.2);  res[1178] = isEqualFloat4(f4, float4(3.41, 2.73, 7.44, 7.92));
  f4 = f41 / float4(3.1, 1.3, 2.0, 1.2);  res[1179] = isEqualFloat4(f4, float4(1.0, 1.0, 1.2, 3.0));

  double1 d1(2.1), d11(2.1);  res[1180] = isEqualDouble1(d1, d11);
  d1 += double1(1.3);         res[1181] = isEqualDouble1(d1, double1(3.4));
  d1 -= double1(1.5);         res[1182] = isEqualDouble1(d1, double1(1.9));
  d1 *= double1(2.1);         res[1183] = isEqualDouble1(d1, double1(3.99));
  d1 /= double1(1.9);         res[1184] = isEqualDouble1(d1, double1(2.1));
  d11 = double1(2.1);         res[1185] = isEqualDouble1(d1, d11);
  d11 = double1(4.1);         res[1186] = !isEqualDouble1(d1, d11);
  d1++;                       res[1187] = isEqualDouble1(d1, double1(3.1));
  ++d1;                       res[1188] = isEqualDouble1(d1, double1(4.1));
  d1--;                       res[1189] = isEqualDouble1(d1, double1(3.1));
  --d1;                       res[1190] = isEqualDouble1(d1, double1(2.1));
  d1 = d11 + double1(5.1);    res[1191] = isEqualDouble1(d1, double1(9.2));
  d1 = d11 - double1(2.1);    res[1192] = isEqualDouble1(d1, double1(2.0));
  d1 = d11 * double1(1.1);    res[1193] = isEqualDouble1(d1, double1(4.51));
  d1 = d11 / double1(2.0);    res[1194] = isEqualDouble1(d1, double1(2.05));

  double2 d2(1.1, 3.1), d21(1.1, 3.1);  res[1195] = isEqualDouble2(d2, d21);
  d2 += double2(1.1, 2.2);        res[1196] = isEqualDouble2(d2, double2(2.2, 5.3));
  d2 -= double2(1.1, 2.1);        res[1197] = isEqualDouble2(d2, double2(1.1, 3.2));
  d2 *= double2(3.1, 1.4);        res[1198] = isEqualDouble2(d2, double2(3.41, 4.48));
  d2 /= double2(3.1, 2.0);        res[1199] = isEqualDouble2(d2, double2(1.1, 2.24));
  d21 = double2(1.1, 2.24);       res[1200] = isEqualDouble2(d2, d21);
  d21 = double2(3.1, 4.4);        res[1201] = !isEqualDouble2(d2, d21);
  d2++;                           res[1202] = isEqualDouble2(d2, double2(2.1, 3.24));
  ++d2;                           res[1203] = isEqualDouble2(d2, double2(3.1, 4.24));
  d2--;                           res[1204] = isEqualDouble2(d2, double2(2.1, 3.24));
  --d2;                           res[1205] = isEqualDouble2(d2, double2(1.1, 2.24));
  d2 = d21 + double2(4.1, 2.1);   res[1206] = isEqualDouble2(d2, double2(7.2, 6.5));
  d2 = d21 - double2(2.1, 1.1);   res[1207] = isEqualDouble2(d2, double2(1.0, 3.3));
  d2 = d21 * double2(1.1, 2.3);   res[1208] = isEqualDouble2(d2, double2(3.41, 10.12));
  d2 = d21 / double2(3.1, 1.1);   res[1209] = isEqualDouble2(d2, double2(1.0, 4.0));

  double3 d3(1.1, 3.1, 2.1), d31(1.1, 3.1, 2.1);  res[1210] = isEqualDouble3(d3, d31);
  d3 += double3(1.1, 2.2, 3.2);       res[1211] = isEqualDouble3(d3, double3(2.2, 5.3, 5.3));
  d3 -= double3(1.1, 2.1, 3.2);       res[1212] = isEqualDouble3(d3, double3(1.1, 3.2, 2.1));
  d3 *= double3(3.1, 1.4, 2.2);       res[1213] = isEqualDouble3(d3, double3(3.41, 4.48, 4.62));
  d3 /= double3(3.1, 2.0, 2.1);       res[1214] = isEqualDouble3(d3, double3(1.1, 2.24, 2.2));
  d31 = double3(1.1, 2.24, 2.2);      res[1215] = isEqualDouble3(d3, d31);
  d31 = double3(3.1, 4.4, 6.4);       res[1216] = !isEqualDouble3(d3, d31);
  d3++;                               res[1217] = isEqualDouble3(d3, double3(2.1, 3.24, 3.2));
  ++d3;                               res[1218] = isEqualDouble3(d3, double3(3.1, 4.24, 4.2));
  d3--;                               res[1219] = isEqualDouble3(d3, double3(2.1, 3.24, 3.2));
  --d3;                               res[1220] = isEqualDouble3(d3, double3(1.1, 2.24, 2.2));
  d3 = d31 + double3(4.1, 2.1, 3.3);  res[1221] = isEqualDouble3(d3, double3(7.2, 6.5, 9.7));
  d3 = d31 - double3(2.1, 1.1, 4.5);  res[1222] = isEqualDouble3(d3, double3(1.0, 3.3, 1.9));
  d3 = d31 * double3(1.1, 2.3, 3.1);  res[1223] = isEqualDouble3(d3, double3(3.41, 10.12, 19.84));
  d3 = d31 / double3(3.1, 1.1, 3.2);  res[1224] = isEqualDouble3(d3, double3(1.0, 4.0, 2.0));

  double4 d4(1.1, 2.1, 3.1, 4.1), d41(1.1, 2.1, 3.1, 4.1);  res[1225] = isEqualDouble4(d4, d41);
  d4 += double4(1.1, 1.2, 1.3, 1.4);       res[1226] = isEqualDouble4(d4, double4(2.2, 3.3, 4.4, 5.5));
  d4 -= double4(1.1, 2.2, 3.2, 4.2);       res[1227] = isEqualDouble4(d4, double4(1.1, 1.1, 1.2, 1.3));
  d4 *= double4(3.1, 1.3, 2.1, 4.2);       res[1228] = isEqualDouble4(d4, double4(3.41, 1.43, 2.52, 5.46));
  d4 /= double4(3.1, 1.1, 1.2, 2.1);       res[1229] = isEqualDouble4(d4, double4(1.1, 1.3, 2.1, 2.6));
  d41 = double4(1.1, 1.3, 2.1, 2.6);       res[1230] = isEqualDouble4(d4, d41);
  d41 = double4(3.1, 1.3, 2.4, 4.8);       res[1231] = !isEqualDouble4(d4, d41);
  d4++;                                    res[1232] = isEqualDouble4(d4, double4(2.1, 2.3, 3.1, 3.6));
  ++d4;                                    res[1233] = isEqualDouble4(d4, double4(3.1, 3.3, 4.1, 4.6));
  d4--;                                    res[1234] = isEqualDouble4(d4, double4(2.1, 2.3, 3.1, 3.6));
  --d4;                                    res[1235] = isEqualDouble4(d4, double4(1.1, 1.3, 2.1, 2.6));
  d4 = d41 + double4(4.1, 2.1, 3.2, 1.5);  res[1236] = isEqualDouble4(d4, double4(7.2, 3.4, 5.6, 6.3));
  d4 = d41 - double4(2.1, 1.1, 1.2, 2.3);  res[1237] = isEqualDouble4(d4, double4(1.0, 0.2, 1.2, 2.5));
  d4 = d41 * double4(1.1, 2.1, 3.1, 2.3);  res[1238] = isEqualDouble4(d4, double4(3.41, 2.73, 7.44, 11.04));
  d4 = d41 / double4(3.1, 1.3, 2.0, 2.4);  res[1239] = isEqualDouble4(d4, double4(1.0, 1.0, 1.2, 2.0));
}
)"};

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hiprtcCompileProgram
 *    1) To test working of "hip/hip_vector_types" header inside kernel string
 * Test source
 * ------------------------
 *  - unit/rtc/hiprtc_VectorTypes_HeaderTst.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_Rtc_VectorTypes_header") {
  std::string kernel_name = "vectorTypes";
  const char* kername = kernel_name.c_str();
  int* result_h;
  int* result_d;
  int n = 1240;
  int Nbytes = n * sizeof(int);
  result_h = new int[n];
  for (int i = 0; i < n; i++) {
    result_h[i] = 0;
  }
  HIP_CHECK(hipMalloc(&result_d, Nbytes));
  HIP_CHECK(hipMemcpy(result_d, result_h, Nbytes, hipMemcpyHostToDevice));

#ifdef __HIP_PLATFORM_AMD__
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string architecture = prop.gcnArchName;
  std::string complete_CO = "--gpu-architecture=" + architecture;
#else
  std::string complete_CO = "--fmad=false";
#endif

  const char* compiler_option = complete_CO.c_str();
  hiprtcProgram prog;

  HIPRTC_CHECK(hiprtcCreateProgram(&prog, vectorTypes_string, kername, 0, NULL, NULL));
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
