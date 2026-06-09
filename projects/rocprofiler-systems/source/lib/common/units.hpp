// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocprofsys::inline common::units
{
inline constexpr std::int64_t nsec = 1;
inline constexpr std::int64_t usec = 1000 * nsec;
inline constexpr std::int64_t msec = 1000 * usec;
inline constexpr std::int64_t sec  = 1000 * msec;

inline constexpr std::int64_t byte     = 1;
inline constexpr std::int64_t kilobyte = 1000 * byte;
inline constexpr std::int64_t megabyte = 1000 * kilobyte;
inline constexpr std::int64_t gigabyte = 1000 * megabyte;

inline constexpr std::int64_t KB = kilobyte;
inline constexpr std::int64_t MB = megabyte;
inline constexpr std::int64_t GB = gigabyte;

inline constexpr std::int64_t nanowatt = 1;
inline constexpr std::int64_t watt     = 1000 * 1000 * 1000 * nanowatt;

}  // namespace rocprofsys::inline common::units
