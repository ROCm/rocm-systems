// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/log_filter.hpp"

#include "logger/debug.hpp"

#include <perfetto.h>

#include <cstdint>
#include <shared_mutex>

namespace rocprofsys::core::log_filter
{

namespace
{
std::shared_mutex g_mutex;
bool              g_registered = false;

// Decision taken for each incoming perfetto log message.
enum class filter_action : std::uint8_t
{
    drop,     // kLogDebug, kLogInfo — silenced from user output.
    warning,  // kLogImportant — forwarded as LOG_WARNING.
    error,    // kLogError — forwarded as LOG_ERROR.
    unknown,  // future SDK enum additions — forwarded as LOG_WARNING
              // with a "unknown severity" prefix so the message is
              // never silently dropped.
};

[[nodiscard]] filter_action
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
    std::shared_lock lock(g_mutex);
    if(!g_registered) return;

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
}  // namespace

void
register_with_perfetto_logger()
{
    std::unique_lock lock(g_mutex);
    if(g_registered) return;
    g_registered = true;
    ::perfetto::base::SetLogMessageCallback(&filter_fn);
}

void
unregister_from_perfetto_logger()
{
    std::unique_lock lock(g_mutex);
    ::perfetto::base::SetLogMessageCallback(nullptr);
    g_registered = false;
}

}  // namespace rocprofsys::core::log_filter
