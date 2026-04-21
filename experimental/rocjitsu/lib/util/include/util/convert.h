// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_CONVERT_H_
#define UTIL_CONVERT_H_

namespace util {

/// @brief Convert between API opaque handles and internal object types.
/// @tparam To Type being converted to.
/// @tparam From Type being converted from
/// @param from Pointer to object being converted.
/// @returns Pointer to objected that was converted to.
template <typename To, typename From> To convert(From from) { return reinterpret_cast<To>(from); }

} // namespace util

#endif // UTIL_CONVERT_H_
