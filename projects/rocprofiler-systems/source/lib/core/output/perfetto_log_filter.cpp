// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/perfetto_log_filter.hpp"

#include "logger/debug.hpp"

#include <mutex>

namespace rocprofsys::output::perfetto_log_filter
{

void
filter_fn(::perfetto::base::LogMessageCallbackArgs args)
{
    const char* file = (args.filename != nullptr) ? args.filename : "<unknown>";
    const char* msg  = (args.message != nullptr) ? args.message : "";

    switch(args.level)
    {
        case ::perfetto::base::LogLev::kLogDebug:
        case ::perfetto::base::LogLev::kLogInfo: return;
        case ::perfetto::base::LogLev::kLogImportant:
            LOG_WARNING("[perfetto] {}:{} {}", file, args.line, msg);
            return;
        case ::perfetto::base::LogLev::kLogError:
            LOG_ERROR("[perfetto] {}:{} {}", file, args.line, msg);
            return;
    }

    // Warn-and-drop on future SDK enum additions instead of silently
    // missing the message.
    LOG_WARNING("[perfetto] unknown severity {}: {}:{} {}", static_cast<int>(args.level),
                file, args.line, msg);
}

void
install()
{
    static std::once_flag once;
    std::call_once(once, []() { ::perfetto::base::SetLogMessageCallback(&filter_fn); });
}

}  // namespace rocprofsys::output::perfetto_log_filter
