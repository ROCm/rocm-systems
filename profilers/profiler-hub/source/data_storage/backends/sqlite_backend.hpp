// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "basic_sqlite_backend.hpp"
#include "sqlite_api_policy.hpp"

namespace profiler_hub::data_storage
{

/**
 * Production SQLite backend: basic_sqlite_backend specialized on the real
 * sqlite3 C API policy. Existing call sites refer to this name and its nested
 * types unchanged.
 */
using sqlite_backend = basic_sqlite_backend<sqlite_api_policy>;

}  // namespace profiler_hub::data_storage
