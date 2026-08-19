// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TESTS_TEST_ENCODINGS_UTIL_H_
#define ROCJITSU_TESTS_TEST_ENCODINGS_UTIL_H_

#include <cstddef>
#include <string_view>

namespace rocjitsu::test_encodings {

inline bool mnemonic_has_prefix(std::string_view mnemonic, std::string_view prefix) {
  return mnemonic.size() >= prefix.size() && mnemonic.substr(0, prefix.size()) == prefix;
}

template <size_t N>
inline bool mnemonic_has_any_prefix(std::string_view mnemonic,
                                    const std::string_view (&prefixes)[N]) {
  for (auto prefix : prefixes)
    if (mnemonic_has_prefix(mnemonic, prefix))
      return true;
  return false;
}

} // namespace rocjitsu::test_encodings

#endif // ROCJITSU_TESTS_TEST_ENCODINGS_UTIL_H_
