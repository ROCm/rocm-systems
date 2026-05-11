// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/color.hpp"

#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace rocprofsys
{
inline namespace common
{
namespace diagnostic
{
namespace color
{
bool
stderr_is_tty() noexcept
{
    return ::isatty(2) != 0;
}

bool
stdout_is_tty() noexcept
{
    return ::isatty(1) != 0;
}

bool
color_supported_for(int fd) noexcept
{
    // NO_COLOR wins over CLICOLOR_FORCE per the no-color.org spec wording.
    if(const char* nc = std::getenv("NO_COLOR"); nc != nullptr)
    {
        return false;
    }

    if(const char* force = std::getenv("CLICOLOR_FORCE");
       force != nullptr && std::strcmp(force, "1") == 0)
    {
        return true;
    }

    return ::isatty(fd) != 0;
}
}  // namespace color
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
