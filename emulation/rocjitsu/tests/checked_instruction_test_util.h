// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/result.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace rocjitsu::test_support {

/// @brief Append a test instruction sequence, checking each builder before unwrapping it.
/// @details Guard the call with ASSERT_TRUE so a failed builder stops the consuming test.
inline ::testing::AssertionResult
append_checked_words(std::vector<uint32_t> &words,
                     std::initializer_list<util::FailureOr<uint32_t>> encodings) {
  for (const util::FailureOr<uint32_t> &encoding : encodings) {
    if (encoding.failed())
      return ::testing::AssertionFailure() << "instruction builder failed at word " << words.size();
    words.push_back(encoding.value());
  }
  return ::testing::AssertionSuccess();
}

} // namespace rocjitsu::test_support
