// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/color.hpp"

#include <cstdlib>
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
no_color_env() noexcept
{
    return std::getenv("NO_COLOR") != nullptr;
}
}  // namespace color
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
