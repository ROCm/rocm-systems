// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>

namespace at
{
struct RecordFunction;
}

namespace torch_trace_collector::detail
{

inline constexpr std::size_t kMaxArgsLength          = 512;
inline constexpr std::size_t kTruncationSuffixLength = 4;
// Keep worst-case wire-format headroom shared with the Python producer even
// though current native schema names and values cannot contain every delimiter.
inline constexpr std::size_t kMaxEncodedArgumentsLength = (kMaxArgsLength * 3) + kTruncationSuffixLength;
inline constexpr std::size_t kMaxEncodedArgumentsSize = kMaxEncodedArgumentsLength + 1;

/**
 * Capture the encoded argument field from a borrowed RecordFunction.
 *
 * The record_function object remains owned by PyTorch and is used only for this
 * synchronous call. The return value includes the trailing null character and
 * is always at least one. When output is non-null and capacity is greater than
 * zero, the destination is always null-terminated.
 */
std::size_t capture_args(const at::RecordFunction& record_function, char* output, std::size_t capacity) noexcept;

}  // namespace torch_trace_collector::detail
