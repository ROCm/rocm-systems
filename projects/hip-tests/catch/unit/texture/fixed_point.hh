/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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