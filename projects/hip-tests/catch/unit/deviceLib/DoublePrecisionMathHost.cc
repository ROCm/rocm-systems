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

#include <hip_test_common.hh>
#include <cmath>

#pragma GCC diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wunused-variable"

__host__ static void double_precision_math_functions() {
  int iX;
  double fX, fY;

  acos(1.0);
  acosh(1.0);
  asin(0.0);
  asinh(0.0);
  atan(0.0);
  atan2(0.0, 1.0);
  atanh(0.0);
  cbrt(0.0);
  ceil(0.0);
  copysign(1.0, -2.0);
  cos(0.0);
  cosh(0.0);
  erf(0.0);
  erfc(0.0);
  exp(0.0);
#ifdef __unix__
  exp10(0.0);
#endif
  exp2(0.0);
  expm1(0.0);
  fabs(1.0);
  fdim(1.0, 0.0);
  floor(0.0);
  fma(1.0, 2.0, 3.0);
  fmax(0.0, 0.0);
  fmin(0.0, 0.0);
  fmod(0.0, 1.0);
  frexp(0.0, &iX);
  hypot(1.0, 0.0);
  ilogb(1.0);
  std::isfinite(0.0);
  std::isinf(0.0);
  std::isnan(0.0);
#ifdef __unix__
  j0(0.0);
  j1(0.0);
  jn(-1.0, 1.0);
#elif _WIN64
  _j0(0.0);
  _j1(0.0);
  _jn(-1.0, 1.0);
#endif
  ldexp(0.0, 0);
  llrint(0.0);
  llround(0.0);
  log(1.0);
  log10(1.0);
  log1p(-1.0);
  log2(1.0);
  logb(1.0);
  lrint(0.0);
  lround(0.0);
  modf(0.0, &fX);
  nan("1");
  nearbyint(0.0);
  fX = 1.0;
  pow(1.0, 0.0);
  remainder(2.0, 1.0);
  remquo(1.0, 2.0, &iX);
  rint(1.0);
  round(0.0);
  scalbln(0.0, 1);
  scalbn(0.0, 1);
  std::signbit(1.0);
  sin(0.0);
#ifdef _unix__
  sincos(0.0, &fX, &fY);
#endif
  sinh(0.0);
  sqrt(0.0);
  tan(0.0);
  tanh(0.0);
  tgamma(2.0);
  trunc(0.0);
#ifdef __unix__
  y0(1.0);
  y1(1.0);
  yn(1, 1.0);
#elif _WIN64
  _y0(1.0);
  _y1(1.0);
  _yn(1, 1.0);
#endif
}

TEST_CASE("Unit_DoublePrecisionMathHost") { double_precision_math_functions(); }
