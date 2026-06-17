// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>

namespace rocprofsys::progress
{

/**
 * @brief Callback invoked by a producer (e.g. storage_parser) to report
 *        incremental byte progress to a consumer (e.g. progress::bar).
 *
 * The argument is the delta in bytes since the previous invocation. An
 * empty (default-constructed) `progress_callback` means "no progress
 * reporting requested" and producers should skip the per-record overhead
 * of computing deltas.
 */
using progress_callback = std::function<void(std::uint64_t /*delta_bytes*/)>;

}  // namespace rocprofsys::progress
