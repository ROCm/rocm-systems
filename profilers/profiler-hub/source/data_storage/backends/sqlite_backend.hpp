// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "basic_sqlite_backend.hpp"
#include "sqlite_api_policy.hpp"
#include "common/traits.hpp"
#include "debug.hpp"
#include <fmt/core.h>

#include <sqlite3.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace profiler_hub::data_storage
{

/**
 * Production SQLite backend: basic_sqlite_backend specialized on the real
 * sqlite3 C API policy. Existing call sites refer to this name and its nested
 * types unchanged.
 */
using sqlite_backend = basic_sqlite_backend<sqlite_api_policy>;

}  // namespace profiler_hub::data_storage
