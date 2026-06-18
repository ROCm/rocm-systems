/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

template <typename Enum> using promoted_enum_t = decltype(+std::declval<Enum>());

template <typename Enum>
constexpr Enum
invalidEnum(std::underlying_type_t<Enum> value)
{
    return static_cast<Enum>(value); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}

// Return the max enum value representable by the result of integral promotion and the underlying type
template <typename Enum>
constexpr auto
maxEnum()
{
    return std::min<std::uintmax_t>(std::numeric_limits<std::underlying_type_t<Enum>>::max(),
                                    std::numeric_limits<promoted_enum_t<Enum>>::max());
}
