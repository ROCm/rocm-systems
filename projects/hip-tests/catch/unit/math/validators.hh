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

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <cstring>
#include <iomanip>

#include "Float16.hh"

// Define a new MatcherBase class with a public 'describe' member function because
// Catch::MatcherBase::describe is protected and thus can't be used via a pointer to
// Catch::MatcherBase.
template <typename T> class MatcherBase : public Catch::Matchers::MatcherBase<T> {
 public:
  virtual std::string describe() const = 0;
  virtual ~MatcherBase() = default;
};

template <typename T, typename Matcher> class ValidatorBase : public MatcherBase<T> {
 public:
  template <typename... Ts>
  ValidatorBase(T target, Ts&&... args) : matcher_{std::forward<Ts>(args)...}, target_{target} {}

  bool match(const T& val) const override {
    if (std::isnan(target_)) {
      return std::isnan(val);
    }

    return matcher_.match(val);
  }

  std::string describe() const override {
    if (std::isnan(target_)) {
      return "is not NaN";
    }

    return matcher_.describe();
  }

 private:
  Matcher matcher_;
  T target_;
  bool nan = false;
};

struct Float16WithinUlpsMatcher : MatcherBase<Float16> {
  Float16WithinUlpsMatcher(Float16 target, uint64_t ulps) : m_target(target), m_ulps(ulps) {}

  bool match(Float16 const& matchee) const override {
    // Comparison with NaN should always be false.
    // This way we can rule it out before getting into the ugly details
    if (__hisnan(matchee) || __hisnan(m_target)) {
      return false;
    }

    auto value_bits = convertFloat16toInt16(matchee);
    auto target_bits = convertFloat16toInt16(m_target);

    // If signs differ, handle the special +0 vs -0 case explicitly.
    if ((value_bits < 0) != (target_bits < 0)) {
      return matchee == m_target;
    }

    auto ulp_diff = std::abs(value_bits - target_bits);
    return static_cast<uint64_t>(ulp_diff) <= m_ulps;
  }

  std::string describe() const override {
    std::stringstream ret;

    ret << "is within " << m_ulps << " ULPs of ";

    write(ret, m_target);
    ret << 'f';
    ret << " ([";

    write(ret, step(m_target, -FLOAT16_MAX, m_ulps));
    ret << ", ";
    write(ret, step(m_target, FLOAT16_MAX, m_ulps));

    ret << "])";

    return ret.str();
  }

 private:
  Float16 getNextAfter(Float16 from, Float16 direction) const {
    constexpr int16_t signbit_float16 = 0x8000;

    // Encode inputs as 16-bit integers
    const int16_t from_bits = convertFloat16toInt16(from);
    const int16_t direction_bits = convertFloat16toInt16(direction);

    // Special cases
    if (from_bits == direction_bits) return direction_bits;
    if (std::abs(from_bits) == static_cast<int16_t>(0) &&
        std::abs(direction_bits) == static_cast<int16_t>(0))
      return direction;

    // Makes integer comparisons reflect numeric ordering across sign.
    const int16_t from_ordered = (from_bits < 0) ? signbit_float16 - from_bits : from_bits;
    const int16_t direction_ordered =
        (direction_bits < 0) ? signbit_float16 - direction_bits : direction_bits;

    // Decide whether to move up or down by one ULP
    const int16_t step = (from_ordered < direction_ordered) ? 1 : -1;

    // Take one step
    const int16_t after_step_ordered = from_ordered + step;

    // Map back from ordered space to raw Float16 bits.
    int16_t next_bits =
        (after_step_ordered < 0) ? signbit_float16 - after_step_ordered : after_step_ordered;

    // Handle boundary behavior for the most-negative edge case.
    if (from_ordered == -1 && (from_ordered < direction_ordered)) {
      next_bits = signbit_float16;
    }

    return convertInt16toFloat16(next_bits);
  }

  Float16 step(Float16 start, Float16 direction, uint64_t steps) const {
    Float16 result = start;
    for (uint64_t i = 0; i < steps; ++i) {
      result = getNextAfter(result, direction);
    }
    return result;
  }

  void write(std::ostream& out, Float16 num) const {
    const uint32_t float16_max_digits = 5;
    out << std::scientific << std::setprecision(float16_max_digits) << num;
  }

  static Float16 convertInt16toFloat16(int16_t d) {
    Float16 i;
    std::memcpy(&i, &d, sizeof(int16_t));
    return i;
  }

  static int16_t convertFloat16toInt16(Float16 d) {
    uint16_t i;
    std::memcpy(&i, &d, sizeof(Float16));
    return i;
  }

  Float16 m_target;
  uint64_t m_ulps;
};


template <typename T> auto ULPValidatorBuilderFactory(int64_t ulps) {
  return [=](T target, auto&&...) {
    return std::make_unique<ValidatorBase<T, Catch::Matchers::WithinUlpsMatcher>>(
        target, Catch::Matchers::WithinULP(target, ulps));
  };
};

template <> inline auto ULPValidatorBuilderFactory<Float16>(int64_t ulps) {
  return [=](Float16 target, auto&&...) {
    return std::make_unique<ValidatorBase<Float16, Float16WithinUlpsMatcher>>(
        target, Float16WithinUlpsMatcher(target, ulps));
  };
};

template <typename T> auto AbsValidatorBuilderFactory(double margin) {
  return [=](T target, auto&&...) {
    return std::make_unique<ValidatorBase<T, Catch::Matchers::WithinAbsMatcher>>(
        target, Catch::Matchers::WithinAbs(target, margin));
  };
}

template <typename T> auto RelValidatorBuilderFactory(T margin) {
  return [=](T target, auto&&...) {
    return std::make_unique<ValidatorBase<T, Catch::Matchers::WithinRelMatcher>>(
        target, Catch::Matchers::WithinRel(target, margin));
  };
}

template <typename T> class EqValidator : public MatcherBase<T> {
 public:
  EqValidator(T target) : target_{target} {}

  bool match(const T& val) const override {
    if (std::isnan(target_)) {
      return std::isnan(val);
    }

    return target_ == val;
  }

  std::string describe() const override {
    std::stringstream ss;
    ss << "is equal to " << target_;
    return ss.str();
  }

 private:
  T target_;
};

template <typename T> auto EqValidatorBuilderFactory() {
  return [](T val, auto&&...) { return std::make_unique<EqValidator<T>>(val); };
}

template <typename T, typename U, typename VBF, typename VBS>
class PairValidator : public MatcherBase<std::pair<T, U>> {
 public:
  PairValidator(const std::pair<T, U>& target, const VBF& vbf, const VBS& vbs)
      : first_matcher_{vbf(target.first)}, second_matcher_{vbs(target.second)} {}

  bool match(const std::pair<T, U>& val) const override {
    return first_matcher_->match(val.first) && second_matcher_->match(val.second);
  }

  std::string describe() const override {
    return "<" + first_matcher_->describe() + ", " + second_matcher_->describe() + ">";
  }

 private:
  decltype(std::declval<VBF>()(std::declval<T>())) first_matcher_;
  decltype(std::declval<VBS>()(std::declval<U>())) second_matcher_;
};

template <typename T, typename ValidatorBuilder>
auto PairValidatorBuilderFactory(const ValidatorBuilder& vb) {
  return [=](const std::pair<T, T>& t, auto&&...) {
    return std::make_unique<PairValidator<T, T, ValidatorBuilder, ValidatorBuilder>>(t, vb, vb);
  };
}

template <typename T, typename U, typename VBF, typename VBS>
auto PairValidatorBuilderFactory(const VBF& vbf, const VBS& vbs) {
  return [=](const std::pair<T, U>& t, auto&&...) {
    return std::make_unique<PairValidator<T, U, VBF, VBS>>(t, vbf, vbs);
  };
}

template <typename T> class NopValidator : public MatcherBase<T> {
 public:
  bool match(const T&) const override { return true; }

  std::string describe() const override { return ""; }
};

template <typename T> auto NopValidatorBuilderFactory() {
  return [](auto&&...) { return std::make_unique<NopValidator<T>>(); };
}
