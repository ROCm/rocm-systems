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

#include "math_common.hh"
#include "validators.hh"

namespace cg = cooperative_groups;

#define CAST_HALF2_KERNEL_DEF(func_name, T)                                                        \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, Float16* const xs) {        \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(__half2{xs[i], -xs[i]});                                                   \
    }                                                                                              \
  }

#define CAST_BINARY_HALF2_KERNEL_DEF(func_name, T)                                                 \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, Float16* const x1s,         \
                                     Float16* const x2s) {                                         \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(__half2{x1s[i], -x1s[i]}, __half2{x2s[i], -x2s[i]});                       \
    }                                                                                              \
  }

template <typename VB> class Float2Validator : public MatcherBase<float2> {
 public:
  Float2Validator(const float2& target, const VB& vb)
      : first_matcher_{vb(target.x)}, second_matcher_{vb(target.y)} {}

  bool match(const float2& val) const override {
    return first_matcher_->match(val.x) && second_matcher_->match(val.y);
  }

  std::string describe() const override {
    return "<" + first_matcher_->describe() + ", " + second_matcher_->describe() + ">";
  }

 private:
  decltype(std::declval<VB>()(float())) first_matcher_;
  decltype(std::declval<VB>()(float())) second_matcher_;
};

template <typename ValidatorBuilder>
auto Float2ValidatorBuilderFactory(const ValidatorBuilder& vb) {
  return [=](const float2& t, auto&&...) {
    return std::make_unique<Float2Validator<ValidatorBuilder>>(t, vb);
  };
}

template <typename VB> class Half2Validator : public MatcherBase<__half2> {
 public:
  Half2Validator(const __half2& target, const VB& vb)
      : first_matcher_{vb(target.data.x)}, second_matcher_{vb(target.data.y)} {}

  bool match(const __half2& val) const override {
    return first_matcher_->match(val.data.x) && second_matcher_->match(val.data.y);
  }

  std::string describe() const override {
    return "<" + first_matcher_->describe() + ", " + second_matcher_->describe() + ">";
  }

 private:
  decltype(std::declval<VB>()(Float16())) first_matcher_;
  decltype(std::declval<VB>()(Float16())) second_matcher_;
};

template <typename ValidatorBuilder> auto Half2ValidatorBuilderFactory(const ValidatorBuilder& vb) {
  return [=](const __half2& t, auto&&...) {
    return std::make_unique<Half2Validator<ValidatorBuilder>>(t, vb);
  };
}
