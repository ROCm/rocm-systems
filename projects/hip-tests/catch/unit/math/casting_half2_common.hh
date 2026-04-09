/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
    for (size_t i = tid; i < num_xs; i += stride) {                                                \
      ys[i] = func_name(__half2{xs[i], -xs[i]});                                                   \
    }                                                                                              \
  }

#define CAST_BINARY_HALF2_KERNEL_DEF(func_name, T)                                                 \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, Float16* const x1s,         \
                                     Float16* const x2s) {                                         \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (size_t i = tid; i < num_xs; i += stride) {                                                \
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
