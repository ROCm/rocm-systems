// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>

#include <cstdlib>
#include <mutex>
#include <string_view>

namespace rocprofsys::sampling
{

// Production FatalErrorPolicy: serialize via FatalMutex (DEC-5 order-index 0)
// and terminate with exit code 1. The mutex serializes the LOG_CRITICAL +
// _Exit pair so a second concurrent fatal does not interleave its log line
// before the first thread exits.
class real_fatal_error_policy
{
public:
    template <class... Args>
    [[noreturn]] void fatal(char const* file, int line, std::string_view fmt_str,
                            Args const&... args) noexcept
    {
        // Mutex must live outside the template so every fatal<Args...> instantiation
        // serializes through the same lock — DEC-5 single-mutex semantics.
        std::lock_guard<std::mutex> const lock{ fatal_mutex() };
        auto const formatted = fmt::vformat(fmt_str, fmt::make_format_args(args...));
        LOG_CRITICAL("[{}:{}] fatal sampling error: {}", file, line, formatted);
        ::_Exit(1);
    }

private:
    static std::mutex& fatal_mutex() noexcept
    {
        static std::mutex m;
        return m;
    }
};

}  // namespace rocprofsys::sampling
