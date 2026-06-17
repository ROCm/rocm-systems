/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Create an invalid enum value without causing compiler errors. This works for
// enums that use int as the underlying type
template <typename Enum>
constexpr Enum
invalidEnum(int value)
{
    return static_cast<Enum>(value); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}
