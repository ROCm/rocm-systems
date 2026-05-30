// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Internal header. Pulls in <perfetto.h> and is included only by
// perfetto_log_filter.cpp and the unit tests. Production callers
// should use perfetto_log_filter.hpp, which exposes only
// register_with_perfetto_logger() and is SDK-free.

#include <perfetto.h>

namespace rocprofsys::output::perfetto_log_filter
{

// Decision taken for each incoming perfetto log message.
enum class filter_action
{
    drop,     // kLogDebug, kLogInfo — silenced from user output.
    warning,  // kLogImportant — forwarded as LOG_WARNING.
    error,    // kLogError — forwarded as LOG_ERROR.
    unknown,  // future SDK enum additions — forwarded as LOG_WARNING
              // with a "unknown severity" prefix so the message is
              // never silently dropped.
};

// Pure classifier. Exposed so the routing can be unit-tested
// without exercising the actual logger.
[[nodiscard]] filter_action
classify(::perfetto::base::LogLev level);

// Drops kLogDebug + kLogInfo, forwards kLogImportant -> LOG_WARNING
// and kLogError -> LOG_ERROR. Signature matches the SDK typedef
// perfetto::base::LogMessageCallback (by value).
void
filter_fn(::perfetto::base::LogMessageCallbackArgs args);

}  // namespace rocprofsys::output::perfetto_log_filter
