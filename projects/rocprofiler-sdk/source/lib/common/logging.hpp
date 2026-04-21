// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/defines.hpp"

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/log/vlog_is_on.h>

#include <fmt/format.h>  // usually used in conjunction with logging
#include <fmt/ranges.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#define ROCP_LOG_LEVEL_TRACE   4
#define ROCP_LOG_LEVEL_INFO    3
#define ROCP_LOG_LEVEL_WARNING 2
#define ROCP_LOG_LEVEL_ERROR   1
#define ROCP_LOG_LEVEL_NONE    0

// Abseil does not provide VLOG_IF or LOG_ASSERT. Define compatibility macros.
#ifndef VLOG_IF
#    define VLOG_IF(level, condition)                                                              \
        if(VLOG_IS_ON(level) && (condition)) VLOG(level)
#endif

#ifndef LOG_ASSERT
#    define LOG_ASSERT(condition) CHECK(condition)
#endif

#define ROCP_TRACE   VLOG(ROCP_LOG_LEVEL_TRACE)
#define ROCP_INFO    LOG(INFO)
#define ROCP_WARNING LOG(WARNING)
#define ROCP_ERROR   LOG(ERROR)
#define ROCP_FATAL   LOG(FATAL)
#define ROCP_DFATAL  DLOG(FATAL)

#define ROCP_TRACE_IF(CONDITION)   VLOG_IF(ROCP_LOG_LEVEL_TRACE, (CONDITION))
#define ROCP_INFO_IF(CONDITION)    LOG_IF(INFO, (CONDITION))
#define ROCP_WARNING_IF(CONDITION) LOG_IF(WARNING, (CONDITION))
#define ROCP_ERROR_IF(CONDITION)   LOG_IF(ERROR, (CONDITION))
#define ROCP_FATAL_IF(CONDITION)   LOG_IF(FATAL, (CONDITION))
#define ROCP_DFATAL_IF(CONDITION)  DLOG_IF(FATAL, (CONDITION))

#if defined(ROCPROFILER_CI)
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) ROCP_FATAL_IF(__VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    ROCP_FATAL
#else
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) ROCP_##NON_CI_LEVEL##_IF(__VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    ROCP_##NON_CI_LEVEL
#endif

namespace rocprofiler
{
namespace common
{
/// CHECK_NOTNULL compatibility wrapper (abseil does not provide one).
/// Returns the value so it can be used as an expression:
///   auto* p = CHECK_NOTNULL(some_ptr);
/// Takes by const& to support move-only types like unique_ptr.
template <typename T>
const T&
check_notnull_impl(const char* expr, const T& ptr)
{
    CHECK(ptr != nullptr) << "'" << expr << "' Must be non NULL";
    return ptr;
}

template <typename T>
T&
check_notnull_impl(const char* expr, T& ptr)
{
    CHECK(ptr != nullptr) << "'" << expr << "' Must be non NULL";
    return ptr;
}

struct logging_config
{
    bool        install_failure_handler = false;
    int32_t     loglevel                = 1;  // WARNING
    int32_t     vlog_level              = ROCP_LOG_LEVEL_WARNING;
    std::string vlog_modules            = {};
    std::string name                    = {};
    std::string logdir                  = {};
};

void
init_logging(std::string_view env_prefix, logging_config cfg = logging_config{});

void
update_logging(const logging_config& cfg);
}  // namespace common
}  // namespace rocprofiler

#define CHECK_NOTNULL(val) ::rocprofiler::common::check_notnull_impl(#val, (val))
