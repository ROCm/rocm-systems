// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Natural alphanumeric ordering of benchmark categories and instruction names.

#pragma once

#include "benchmark/sampling.h"

#include <string_view>

namespace rocjitsu::arith::test {

// Compare digit runs as integers without parsing into a bounded integer type.
// Equal numeric values use the shorter spelling first (2 before 02).
inline bool alphanumeric_less(std::string_view left, std::string_view right) {
  auto digit = [](char c) { return c >= '0' && c <= '9'; };
  while (!left.empty() && !right.empty()) {
    if (digit(left.front()) && digit(right.front())) {
      int left_digit_count = 0;
      int right_digit_count = 0;
      while (left_digit_count < int(left.size()) && digit(left[left_digit_count])) {
        ++left_digit_count;
      }
      while (right_digit_count < int(right.size()) && digit(right[right_digit_count])) {
        ++right_digit_count;
      }
      auto left_significant_digits = left.substr(0, left_digit_count);
      auto right_significant_digits = right.substr(0, right_digit_count);
      while (!left_significant_digits.empty() && left_significant_digits.front() == '0') {
        left_significant_digits.remove_prefix(1);
      }
      while (!right_significant_digits.empty() && right_significant_digits.front() == '0') {
        right_significant_digits.remove_prefix(1);
      }
      if (left_significant_digits.size() != right_significant_digits.size()) {
        return left_significant_digits.size() < right_significant_digits.size();
      }
      if (left_significant_digits != right_significant_digits) {
        return left_significant_digits < right_significant_digits;
      }
      if (left_digit_count != right_digit_count) {
        return left_digit_count < right_digit_count;
      }
      left.remove_prefix(left_digit_count);
      right.remove_prefix(right_digit_count);
    } else {
      if (left.front() != right.front()) {
        return left.front() < right.front();
      }
      left.remove_prefix(1);
      right.remove_prefix(1);
    }
  }
  return left.empty() && !right.empty();
}

// Order workloads by category, then instruction. Equivalent rows retain their
// semantics/CPU order when used with stable_sort.
inline bool benchmark_less(const Benchmark &left, const Benchmark &right) {
  if (left.category != right.category) {
    return alphanumeric_less(left.category, right.category);
  }
  return alphanumeric_less(left.instruction, right.instruction);
}

} // namespace rocjitsu::arith::test
