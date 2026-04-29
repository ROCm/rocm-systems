// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger/debug.hpp"

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
    [[noreturn]] void fatal(char const* file, int line, std::string_view fmt,
                            Args const&... /*args*/) noexcept
    {
        static std::mutex                 fatal_mtx;
        std::lock_guard<std::mutex> const lock{ fatal_mtx };
        LOG_CRITICAL("[{}:{}] fatal sampling error: {}", file, line, fmt);
        ::_Exit(1);
    }
};

}  // namespace rocprofsys::sampling
