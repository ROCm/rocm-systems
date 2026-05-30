// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/perfetto_log_filter.hpp"
#include "output/perfetto_log_filter_detail.hpp"

#include "logger/debug.hpp"

#include <atomic>
#include <mutex>

namespace rocprofsys::output::perfetto_log_filter
{

namespace
{
// Cleared by unregister_from_perfetto_logger() before the spdlog
// logger's function-local static can be destroyed. A perfetto worker
// thread already inside filter_fn at unregister time sees this flag
// on its next entry and short-circuits before touching the logger.
std::atomic<bool> g_callback_active{ false };
}  // namespace

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
    // Shutdown gate. acquire pairs with the release store in
    // unregister_from_perfetto_logger() so a worker thread cannot
    // reach the logger after finalize has cleared the flag.
    if(!g_callback_active.load(std::memory_order_acquire)) return;

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
    std::call_once(once, []() {
        g_callback_active.store(true, std::memory_order_release);
        ::perfetto::base::SetLogMessageCallback(&filter_fn);
    });
}

void
unregister_from_perfetto_logger()
{
    // Close the gate first so any in-flight callback that has already
    // passed the SDK dispatch but not yet executed the load sees the
    // disabled state on its next entry. Then drop the SDK callback so
    // no new invocations occur.
    g_callback_active.store(false, std::memory_order_release);
    ::perfetto::base::SetLogMessageCallback(nullptr);
}

}  // namespace rocprofsys::output::perfetto_log_filter
