// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/perfetto_log_filter.hpp"
#include "output/perfetto_log_filter_detail.hpp"

#include "logger/debug.hpp"

#include <mutex>

namespace rocprofsys::output::perfetto_log_filter
{

filter_action
classify(::perfetto::base::LogLev level)
{
    switch(level)
    {
        case ::perfetto::base::LogLev::kLogDebug:
        case ::perfetto::base::LogLev::kLogInfo: return filter_action::drop;
        case ::perfetto::base::LogLev::kLogImportant: return filter_action::warning;
        case ::perfetto::base::LogLev::kLogError: return filter_action::error;
    }
    return filter_action::unknown;
}

void
filter_fn(::perfetto::base::LogMessageCallbackArgs args)
{
    const char* file = (args.filename != nullptr) ? args.filename : "<unknown>";
    const char* msg  = (args.message != nullptr) ? args.message : "";

    switch(classify(args.level))
    {
        case filter_action::drop: return;
        case filter_action::warning:
            LOG_WARNING("[perfetto] {}:{} {}", file, args.line, msg);
            return;
        case filter_action::error:
            LOG_ERROR("[perfetto] {}:{} {}", file, args.line, msg);
            return;
        case filter_action::unknown:
            LOG_WARNING("[perfetto] unknown severity {}: {}:{} {}",
                        static_cast<int>(args.level), file, args.line, msg);
            return;
    }
}

void
register_with_perfetto_logger()
{
    static std::once_flag once;
    std::call_once(once, []() { ::perfetto::base::SetLogMessageCallback(&filter_fn); });
}

}  // namespace rocprofsys::output::perfetto_log_filter
