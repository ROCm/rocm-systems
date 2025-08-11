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

#include <catch.hpp>

// Define a new MatcherBase class with a public 'describe' member function because
// Catch::MatcherBase::describe is protected and thus can't be used via a pointer to
// Catch::MatcherBase.
template <typename T> class MatcherBase : public Catch::MatcherBase<T> {
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

template <typename T> auto ULPValidatorBuilderFactory(int64_t ulps) {
  return [=](T target, auto&&...) {
    return std::make_unique<ValidatorBase<T, Catch::Matchers::Floating::WithinUlpsMatcher>>(
        target, Catch::WithinULP(target, ulps));
  };
};

template <typename T> auto AbsValidatorBuilderFactory(double margin) {
  return [=](T target, auto&&...) {
    return std::make_unique<ValidatorBase<T, Catch::Matchers::Floating::WithinAbsMatcher>>(
        target, Catch::WithinAbs(target, margin));
  };
}

template <typename T> auto RelValidatorBuilderFactory(T margin) {
  return [=](T target, auto&&...) {
    return std::make_unique<ValidatorBase<T, Catch::Matchers::Floating::WithinRelMatcher>>(
        target, Catch::WithinRel(target, margin));
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
    ss << " is not equal to " << target_;
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
