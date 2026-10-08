// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_DETAIL_CONVERT_PORTABLE_H_
#define UTIL_DETAIL_CONVERT_PORTABLE_H_

#include <cstddef>
#include <cstdint>

namespace util::detail {

/// Leave the index unchanged so the public conversion's scalar tail handles all inputs.
inline void f16_to_f32_block_portable(const uint16_t *, float *, size_t, size_t &) {}
inline void bf16_to_f32_block_portable(const uint16_t *, float *, size_t, size_t &) {}
inline void i8_to_i32_block_portable(const int8_t *, int32_t *, size_t, size_t &) {}
inline void u8_to_i32_block_portable(const uint8_t *, int32_t *, size_t, size_t &) {}

} // namespace util::detail

#endif // UTIL_DETAIL_CONVERT_PORTABLE_H_
