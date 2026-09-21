// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string_view>

namespace rocjitsu {

template <std::size_t EncodingCount>
inline bool mnemonic_has_any_prefix(std::string_view mnemonic,
                                    const std::string_view (&prefixes)[EncodingCount]) {
  for (std::string_view prefix : prefixes)
    if (mnemonic.starts_with(prefix))
      return true;
  return false;
}

} // namespace rocjitsu
