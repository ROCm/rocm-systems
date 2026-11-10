// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace profiler_hub::pytorch
{
// True when @p text, the first 64 KiB of a file (decompressed if gzip), starts with '{'
// and has the key "traceEvents" and one of "schemaVersion", "deviceProperties",
// "distributedInfo", "baseTimeNanoseconds" or "cpu_op". It searches for the keys and does
// not parse; load_kineto is the real check.
bool
looks_like_kineto(std::string_view text);
}  // namespace profiler_hub::pytorch
