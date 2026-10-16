// SPDX-License-Identifier: MIT

// Natural alphanumeric ordering of benchmark categories and instruction names.

#pragma once

#include "benchmark/sampling.h"

#include <string_view>

namespace rocjitsu::arith::bench {

// Compare digit runs as integers without parsing into a bounded integer type.
// Equal numeric values use the shorter spelling first (2 before 02).
inline bool alphanumeric_less(std::string_view left, std::string_view right) {
  auto digit = [](char c) { return c >= '0' && c <= '9'; };
  while (!left.empty() && !right.empty()) {
    if (digit(left.front()) && digit(right.front())) {
      int a = 0, b = 0;
      while (a < int(left.size()) && digit(left[a])) {
        ++a;
      }
      while (b < int(right.size()) && digit(right[b])) {
        ++b;
      }
      auto x = left.substr(0, a), y = right.substr(0, b);
      while (!x.empty() && x.front() == '0') {
        x.remove_prefix(1);
      }
      while (!y.empty() && y.front() == '0') {
        y.remove_prefix(1);
      }
      if (x.size() != y.size()) {
        return x.size() < y.size();
      }
      if (x != y) {
        return x < y;
      }
      if (a != b) {
        return a < b;
      }
      left.remove_prefix(a);
      right.remove_prefix(b);
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

} // namespace rocjitsu::arith::bench
