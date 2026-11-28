// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string_view>

namespace profiler_hub::pytorch
{
// True when @p text, the first 64 KiB of a file (decompressed if gzip), starts with '{'
// and has the key "traceEvents" and one of "schemaVersion", "deviceProperties",
// "distributedInfo", "baseTimeNanoseconds" or "cpu_op". It searches for the keys and does
// not parse; load_kineto is the real check.
bool
looks_like_kineto(std::string_view text);

// The value of the "schemaVersion" key in @p text, the same prefix looks_like_kineto
// sees. 0 when the key is absent or its value is not a non-negative integer.
std::uint32_t
schema_version_in_prefix(std::string_view text);
}  // namespace profiler_hub::pytorch
