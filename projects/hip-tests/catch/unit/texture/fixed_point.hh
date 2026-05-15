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

#pragma once

#include <cmath>

template <size_t fractional_bits> class FixedPoint {
 public:
  FixedPoint() = default;
  FixedPoint(float x) { fixed_point_ = static_cast<int16_t>(roundf(x * (1 << fractional_bits))); }

  operator float() const {
    return (static_cast<float>(fixed_point_) / static_cast<float>(1 << fractional_bits));
  }

  FixedPoint operator+(FixedPoint other) const {
    FixedPoint<fractional_bits> res;
    res.fixed_point_ = fixed_point_ + other.fixed_point_;
    return res;
  }

  FixedPoint operator-(FixedPoint other) const {
    FixedPoint<fractional_bits> res;
    res.fixed_point_ = fixed_point_ - other.fixed_point_;
    return res;
  }

  FixedPoint operator*(FixedPoint other) const {
    constexpr auto K = 1 << (fractional_bits - 1);

    FixedPoint<fractional_bits> res;
    int32_t temp;

    temp = static_cast<int32_t>(fixed_point_) * static_cast<int32_t>(other.fixed_point_);
    temp += K;

    res.fixed_point_ = Sat16(temp >> fractional_bits);

    return res;
  }

 private:
  int16_t fixed_point_;

  int16_t Sat16(int32_t x) const {
    if (x > 0x7FFF)
      return 0x7FFF;
    else if (x < -0x8000)
      return -0x8000;
    else
      return (int16_t)x;
  }
};